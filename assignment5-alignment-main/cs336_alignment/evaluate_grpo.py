import argparse
import json
from datetime import datetime
from pathlib import Path

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.vllm_utils import VLLMServer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="allenai/OLMo-2-0425-1B",
    )
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--limit", type=int, default=32)
    args = parser.parse_args()

    assert args.limit > 0

    # 每次评测单独保存，避免覆盖之前的结果。
    output_dir = Path("runs") / datetime.now().strftime(
        "eval_%Y%m%d_%H%M%S_%f"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    examples = []
    with Path("data/gsm8k/test.jsonl").open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            examples.append({
                "question": record["question"],
                "ground_truth": record["answer"].split("####")[-1].strip(),
            })
            if len(examples) >= args.limit:
                break

    assert examples, "评测数据为空"

    prompt_template = Path(
        "cs336_alignment/prompts/r1_zero.prompt"
    ).read_text(encoding="utf-8")

    prompts = [
        prompt_template.format(question=example["question"])
        for example in examples
    ]

    # 快速评测使用 greedy decoding，减少采样随机性。
    sampling_params = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 512,
        "n": 1,
        "seed": 0,
        "stop": ["</answer>"],
        "include_stop_str_in_output": True,
    }

    config = {
        "model": args.model,
        "gpu": args.gpu,
        "data_path": "data/gsm8k/test.jsonl",
        "selection": "first_n",
        "num_examples": len(examples),
        "prompt_template": prompt_template,
        "sampling_params": sampling_params,
        "reward_fn": "r1_zero_reward_fn",
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    server = VLLMServer(
        model_id=args.model,
        gpu=args.gpu,
        port=args.port,
        seed=0,
        gpu_memory_utilization=0.8,
    )

    try:
        server.start()
        completions = server.generate_completions(
            prompts=prompts,
            sampling_params=sampling_params,
        )
        assert len(completions) == len(examples)

        results = []
        for index, (example, completion) in enumerate(
            zip(examples, completions)
        ):
            scores = r1_zero_reward_fn(
                completion.text,
                example["ground_truth"],
            )
            results.append({
                "index": index,
                **example,
                "response": completion.text,
                "finish_reason": completion.finish_reason,
                **scores,
            })

        with (output_dir / "responses.jsonl").open(
            "w", encoding="utf-8"
        ) as file:
            for result in results:
                file.write(
                    json.dumps(result, ensure_ascii=False) + "\n"
                )

        summary = {
            "num_examples": len(results),
            # 正确答案且格式合格，才计为成功。
            "accuracy": sum(r["reward"] for r in results) / len(results),
            "format_rate": (
                sum(r["format_reward"] for r in results) / len(results)
            ),
        }

        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(summary, flush=True)
        print(f"Results saved to: {output_dir}", flush=True)

    finally:
        server.stop()


if __name__ == "__main__":
    main()
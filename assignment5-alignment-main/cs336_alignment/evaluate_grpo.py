import argparse
import json
from datetime import datetime
from pathlib import Path

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.vllm_utils import VLLMServer


def load_eval_examples(
    data_path: str,
    limit: int,
) -> list[dict[str, str]]:
    assert limit > 0

    examples = []
    with Path(data_path).open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            examples.append({
                "question": record["question"],
                "ground_truth": record["answer"].split("####")[-1].strip(),
            })
            if len(examples) >= limit:
                break

    if len(examples) != limit:
        raise ValueError(
            f"要求 {limit} 道题，但文件中只有 {len(examples)} 道"
        )

    return examples


def evaluate_policy(
    server: VLLMServer,
    examples: list[dict[str, str]],
    prompt_template: str,
    output_dir: Path,
    seed: int = 0,
    batch_size: int = 32,
) -> dict[str, float | int]:
    """评测已有 server 的当前权重，不启动服务、不更新参数。"""
    assert examples

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    prompts = [
        prompt_template.format(question=example["question"])
        for example in examples
    ]

    # 与训练采样参数分开，避免改变训练的 n 和 temperature。
    sampling_params = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 512,
        "n": 1,
        "seed": seed,
        "stop": ["</answer>"],
        "include_stop_str_in_output": True,
    }

    (output_dir / "eval_config.json").write_text(
        json.dumps(
            {
                "num_examples": len(examples),
                "prompt_template": prompt_template,
                "sampling_params": sampling_params,
                "batch_size": batch_size,
                "reward_fn": "r1_zero_reward_fn",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    completions = server.generate_completions(
        prompts=prompts,
        sampling_params=sampling_params,
        batch_size=batch_size,
    )
    assert len(completions) == len(examples)

    total_reward = 0.0
    total_format_reward = 0.0
    total_response_tokens = 0

    with (output_dir / "responses.jsonl").open(
        "w", encoding="utf-8"
    ) as file:
        for index, (example, completion) in enumerate(
            zip(examples, completions)
        ):
            scores = r1_zero_reward_fn(
                completion.text,
                example["ground_truth"],
            )
            response_tokens = len(completion.token_ids)

            total_reward += scores["reward"]
            total_format_reward += scores["format_reward"]
            total_response_tokens += response_tokens

            result = {
                "index": index,
                **example,
                "response": completion.text,
                "response_tokens": response_tokens,
                "finish_reason": completion.finish_reason,
                **scores,
            }
            file.write(json.dumps(result, ensure_ascii=False) + "\n")

    count = len(examples)
    summary = {
        "num_examples": count,
        "accuracy": total_reward / count,
        "format_rate": total_format_reward / count,
        "mean_response_tokens": total_response_tokens / count,
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Evaluation: {summary}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--data", default="data/gsm8k/test.jsonl")
    parser.add_argument(
        "--prompt",
        default="cs336_alignment/prompts/r1_zero.prompt",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    args = parser.parse_args()

    examples = load_eval_examples(args.data, args.limit)
    prompt_template = Path(args.prompt).read_text(encoding="utf-8")

    output_dir = Path("runs") / datetime.now().strftime(
        "eval_%Y%m%d_%H%M%S_%f"
    )

    server = VLLMServer(
        model_id=args.model,
        gpu=args.gpu,
        port=args.port,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    try:
        server.start()
        evaluate_policy(
            server=server,
            examples=examples,
            prompt_template=prompt_template,
            output_dir=output_dir,
            seed=args.seed,
            batch_size=args.batch_size,
        )

        # 独立评测额外记录模型、数据来源等信息。
        (output_dir / "config.json").write_text(
            json.dumps(vars(args), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Results saved to: {output_dir}", flush=True)
    finally:
        server.stop()


if __name__ == "__main__":
    main()
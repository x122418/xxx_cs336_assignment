import json
import argparse
import random, torch
from pathlib import Path
from datetime import datetime

from cs336_alignment.grpo_utils import grpo_train_step
from cs336_alignment.drgrpo_grader import (
    question_only_reward_fn,
    r1_zero_reward_fn,
)
PROMPT_CONFIGS = {
    "question_only": {
        "path": "cs336_alignment/prompts/question_only.prompt",
        "reward_fn": question_only_reward_fn,
        "stop": None,
    },
    "r1_zero": {
        "path": "cs336_alignment/prompts/r1_zero.prompt",
        "reward_fn": r1_zero_reward_fn,
        "stop": "</answer>",
    },
    "r1_zero_three_shot": {
        "path": "cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt",
        "reward_fn": r1_zero_reward_fn,
        "stop": "</answer>",
    },
}


PROMPTS_PER_BATCH = 32
GROUP_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 32


TRAIN_DEVICE = "cuda:5"
INFERENCE_GPU = 4
PORT = 18080

NUM_STEPS = 200
N_TRAIN_EXAMPLES = 6400
N_VAL_EXAMPLES = 1024
EVAL_EVERY = 10



from cs336_alignment.vllm_utils import VLLMServer
from cs336_alignment.drgrpo_grader import (question_only_reward_fn, r1_zero_reward_fn)

from cs336_alignment.checkpoint import get_model_and_tokenizer
from cs336_alignment.vllm_utils import VLLMServer
from cs336_alignment.evaluate_grpo_prompt import (
    evaluate_policy,
    load_eval_examples,
)

MODEL_ID = "allenai/OLMo-2-0425-1B"

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--train-device", default=TRAIN_DEVICE)
parser.add_argument("--inference-gpu", type=int, default=INFERENCE_GPU)
parser.add_argument("--port", type=int, default=PORT)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument(
    "--prompt-type",
    choices=list(PROMPT_CONFIGS),
    default="r1_zero",
)
args = parser.parse_args()

prompt_config = PROMPT_CONFIGS[args.prompt_type]
reward_fn = prompt_config["reward_fn"]
stop = prompt_config["stop"]

seed = args.seed
train_device = args.train_device

random.seed(seed)
torch.manual_seed(seed)
torch.cuda.set_device(train_device)

def load_training_examples():
    examples = []
    with Path("data/gsm8k/train.jsonl").open(
        encoding="utf-8"
    ) as file:
        for line in file:
            record = json.loads(line)
            examples.append(
                {
                    'question': record['question'],
                    'ground_truth': (
                        record['answer'].split("####")[-1].strip()
                    ),
                }
            )
    return examples

def main():
    torch.manual_seed(seed)
    torch.cuda.set_device(train_device)

    run_dir = Path("runs") / (
        f"grpo_prompt_{args.prompt_type}_lr{args.lr:g}_seed{seed}_"
        + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    config = {
        "model_id": MODEL_ID,
        "train_device": train_device,
        "learning_rate": args.lr,
        "inference_gpu": args.inference_gpu,
        "num_steps": NUM_STEPS,
        "prompts_per_batch": PROMPTS_PER_BATCH,
        "group_size": GROUP_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "max_grad_norm": 1.0,
        "seed": seed,
        "port": args.port,
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 512,
        "prompt_type": args.prompt_type,
        "prompt_path": prompt_config["path"],
        "reward_fn": reward_fn.__name__,
        "stop": stop,
        "baseline": "mean",
        "advantage_normalizer": "std",
        "n_train_examples": N_TRAIN_EXAMPLES,
        "n_val_examples": N_VAL_EXAMPLES,
        "eval_every": EVAL_EVERY,
        "eval_temperature": 0.0,
        "eval_seed": 0,
    }


    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Run directory: {run_dir}", flush=True)

    model, tokenizer = get_model_and_tokenizer(
        MODEL_ID, 
        train_device,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.use_cache = False
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    print(
        f"seed={seed}, actual_lr={optimizer.param_groups[0]['lr']}",
        flush=True,
    )

    print("HF model and optimizer initialized.", flush=True)

    examples = load_training_examples()[:N_TRAIN_EXAMPLES]
    assert len(examples) == N_TRAIN_EXAMPLES

    val_examples = load_eval_examples(
        "data/gsm8k/test.jsonl",
        N_VAL_EXAMPLES,
    )


    prompt_template = Path(
        prompt_config["path"]
    ).read_text(encoding="utf-8")

    rng = random.Random(seed)

    server = VLLMServer(
        model_id=MODEL_ID,
        gpu=args.inference_gpu,
        port=args.port,
        seed=seed,
        gpu_memory_utilization=0.25,
    )

    def run_validation(eval_step):
        summary = evaluate_policy(
            server=server,
            examples=val_examples,
            prompt_template=prompt_template,
            output_dir=run_dir / "eval" / f"step_{eval_step:04d}",
            seed=0,
            batch_size=32,
            reward_fn=reward_fn,
            stop=stop,
        )

        record = {
            "step": eval_step,
            "val_reward": summary["accuracy"],
            "val_format_reward": summary["format_rate"],
            "val_mean_response_tokens": summary["mean_response_tokens"],
            "num_examples": summary["num_examples"],
        }

        with (run_dir / "val_metrics.jsonl").open(
            "a", encoding="utf-8"
        ) as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


    try:
        server.start()
        print("vLLM server ready.", flush = True)

        server.init_weight_sync(policy_device=train_device)
        server.sync_policy_weights(model)
        print("Initial policy weights synchronized.", flush=True)
        run_validation(0)


        sampling_params = {
            "temperature": 1.0,
            "top_p": 1.0,
            'max_tokens': 512,
            "n": 1,
            "seed": seed,
        }
        if stop is not None:
            sampling_params["stop"] = [stop]
            sampling_params["include_stop_str_in_output"] = True

        for step in range(NUM_STEPS):
            batch = rng.sample(examples, PROMPTS_PER_BATCH)
        
            repeated_prompts = []
            repeated_ground_truths = []

            for example in batch:
                prompt = prompt_template.format(question = example['question'])

                repeated_prompts.extend([prompt] * GROUP_SIZE)
                repeated_ground_truths.extend(
                    [example['ground_truth']] * GROUP_SIZE
                )
            prompts = [
                prompt_template.format(question=example["question"])
                for example in batch
            ]
            sampling_params["n"] = GROUP_SIZE

            completions = server.generate_completions(
                prompts = prompts,
                sampling_params=sampling_params,
            )
            for j, completion in enumerate(completions):
                if not completion.text:
                    print(
                        "Empty completion:",
                        {
                            "index": j,
                            "text": repr(completion.text),
                            "token_ids": completion.token_ids,
                            "finish_reason": completion.finish_reason,
                            "eos_token_id": tokenizer.eos_token_id,
                        },
                        flush=True,
                    )

            rollout_responses = [completion.text for completion in completions]

            assert len(rollout_responses) == len(batch) * GROUP_SIZE


            if (step + 1) % 40 == 0:
                rollout_dir = run_dir / "rollouts"
                rollout_dir.mkdir(parents=True, exist_ok=True)

                rollout_path = rollout_dir / f"step_{step + 1:04d}.jsonl"

                with rollout_path.open("w", encoding="utf-8") as file:
                    for j, completion in enumerate(completions):
                        example = batch[j // GROUP_SIZE]
                        scores = reward_fn(
                            completion.text,
                            repeated_ground_truths[j],
                        )

                        result = {
                            "step": step + 1,
                            "prompt_index": j // GROUP_SIZE,
                            "sample_index": j % GROUP_SIZE,
                            "question": example["question"],
                            "prompt": repeated_prompts[j],
                            "ground_truth": repeated_ground_truths[j],
                            "response": completion.text,
                            "response_tokens": len(completion.token_ids),
                            "finish_reason": completion.finish_reason,
                            **scores,
                        }
                        file.write(json.dumps(result, ensure_ascii=False) + "\n")

            loss, metadata = grpo_train_step(
                model = model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                max_grad_norm = 1,
                reward_fn = reward_fn,
                repeated_prompts=repeated_prompts,
                rollout_responses=rollout_responses,
                repeated_ground_truths=repeated_ground_truths,
                group_size = GROUP_SIZE,
            )
            print(
                f"Step {step+1}/{NUM_STEPS} | "
                f"loss = {loss.item():.6f} | {metadata}",
                flush = True,
            )
            record = {
                "step": step + 1,
                "loss": loss.item(),
                **metadata,
            }

            with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")            


            server.sync_policy_weights(model)
            print("Policy weights synchronized.", flush=True)

            if (step + 1) % EVAL_EVERY == 0:
                run_validation(step + 1)

        checkpoint_dir = run_dir / "final_checkpoint"
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"Checkpoint saved: {checkpoint_dir}", flush=True)


    finally:
        server.stop()

if __name__ == "__main__":
    main()
import json
import random, torch
from pathlib import Path
from datetime import datetime

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.grpo_utils import grpo_train_step

NUM_STEPS = 50
PROMPTS_PER_BATCH = 32
GROUP_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 32

from cs336_alignment.vllm_utils import VLLMServer
from cs336_alignment.drgrpo_grader import (question_only_reward_fn, r1_zero_reward_fn)

from cs336_alignment.checkpoint import get_model_and_tokenizer
from cs336_alignment.vllm_utils import VLLMServer
from cs336_alignment.evaluate_grpo import (
    evaluate_policy,
    load_eval_examples,
)

MODEL_ID = "allenai/OLMo-2-0425-1B"
# 改成你实际可用的两张卡
TRAIN_DEVICE = "cuda:6"
INFERENCE_GPU = 4
PORT = 18080

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
    torch.manual_seed(0)
    torch.cuda.set_device(TRAIN_DEVICE)
    run_dir = Path("runs") / datetime.now().strftime("grpo_%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)

    config = {
        "model_id": MODEL_ID,
        "train_device": TRAIN_DEVICE,
        "inference_gpu": INFERENCE_GPU,
        "num_steps": NUM_STEPS,
        "prompts_per_batch": PROMPTS_PER_BATCH,
        "group_size": GROUP_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": 1e-5,
        "max_grad_norm": 1.0,
        "seed": 0,
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 512,
        "prompt_path": "cs336_alignment/prompts/r1_zero.prompt",
        "reward_fn": "r1_zero_reward_fn",
        "baseline": "mean",
        "advantage_normalizer": "std",
    }

    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Run directory: {run_dir}", flush=True)

    model, tokenizer = get_model_and_tokenizer(
        MODEL_ID, 
        TRAIN_DEVICE,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.use_cache = False
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )

    print("HF model and optimizer initialized.", flush=True)

    examples = load_training_examples()
    prompt_template = Path(
        "cs336_alignment/prompts/r1_zero.prompt"
    ).read_text(encoding="utf-8")

    rng = random.Random(0)

    server = VLLMServer(
        model_id = MODEL_ID,
        gpu = INFERENCE_GPU,
        port = PORT,
        seed = 0,
        gpu_memory_utilization=0.25
    )

    try:
        server.start()
        print("vLLM server ready.", flush = True)

        server.init_weight_sync(policy_device=TRAIN_DEVICE)
        server.sync_policy_weights(model)
        print("Initial policy weights synchronized.", flush=True)



        sampling_params = {
            "temperature": 1.0,
            "top_p": 1.0,
            'max_tokens': 512,
            "n": 1,
            "seed": 0,
            "stop": ["</answer>"],
            "include_stop_str_in_output": True,
        }

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
            rollout_responses = [completion.text for completion in completions]

            assert len(rollout_responses) == len(batch) * GROUP_SIZE

            loss, metadata = grpo_train_step(
                model = model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                max_grad_norm = 1,
                reward_fn = r1_zero_reward_fn,
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

        checkpoint_dir = run_dir / "final_checkpoint"
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"Checkpoint saved: {checkpoint_dir}", flush=True)


    finally:
        server.stop()

if __name__ == "__main__":
    main()
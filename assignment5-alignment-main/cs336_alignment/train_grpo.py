import json
import argparse
import random, torch
import subprocess
import time
from pathlib import Path
from datetime import datetime

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.grpo_utils import grpo_train_step

PROMPTS_PER_BATCH = 32
GROUP_SIZE = 8
TRAIN_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 1


TRAIN_DEVICE = "cuda:1"
INFERENCE_GPU = 0
PORT = 18080

NUM_STEPS = 200
N_TRAIN_EXAMPLES = 6400
N_VAL_EXAMPLES = 1024
EVAL_EVERY = 10

from cs336_alignment.vllm_utils import VLLMServer
from cs336_alignment.drgrpo_grader import (question_only_reward_fn, r1_zero_reward_fn)
from cs336_alignment.grpo_utils import (tokenize_prompt_and_output, get_response_log_probs)

from cs336_alignment.checkpoint import get_model_and_tokenizer
from cs336_alignment.vllm_utils import VLLMServer
from cs336_alignment.evaluate_grpo import (
    evaluate_policy,
    load_eval_examples,
)

MODEL_ID = "allenai/OLMo-2-0425-1B"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--policy-mode",
    choices=['onpolicy', 'offpolicy'],
    default="onpolicy"
)
parser.add_argument("--train-batch-size", type=int, default=None)
parser.add_argument("--gradient-accumulation-steps", type=int, default=None)

parser.add_argument(
    "--importance_method",
    choices=['none', 'noclip', 'grpo', 'gspo'],
    default=None
)
parser.add_argument('--cliprange', type=float, default=None)
parser.add_argument("--num_steps", type=int, default=NUM_STEPS)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--train-device", default=TRAIN_DEVICE)
parser.add_argument("--inference-gpu", type=int, default=INFERENCE_GPU)
parser.add_argument("--port", type=int, default=PORT)
parser.add_argument("--lr", type=float, default=1e-5)
args = parser.parse_args()

rollout_batch_size = PROMPTS_PER_BATCH * GROUP_SIZE
is_onpolicy = args.policy_mode == "onpolicy"

train_batch_size = args.train_batch_size
if train_batch_size is None:
    train_batch_size = rollout_batch_size if is_onpolicy else 8

gradient_accumulation_steps = args.gradient_accumulation_steps
if gradient_accumulation_steps is None:
    gradient_accumulation_steps = 32 if is_onpolicy else 1
if args.importance_method is None:
    args.importance_method = "none" if is_onpolicy else "grpo"

if rollout_batch_size % train_batch_size != 0:
    parser.error("rollout batch size 必须能被 train batch size 整除")
if train_batch_size % GROUP_SIZE != 0:
    parser.error("train batch size 必须包含完整的回答组")
if train_batch_size % gradient_accumulation_steps != 0:
    parser.error("train batch size 必须能被 gradient accumulation steps 整除")
if is_onpolicy and train_batch_size != rollout_batch_size:
    parser.error("onpolicy 要求每轮 rollout 只调用一次训练更新")

old_log_probs_batch_size = 8

seed = args.seed
train_device = args.train_device

print(
    f"mode={args.policy_mode}, "
    f"importance={args.importance_method}, "
    f"rollout_batch={rollout_batch_size}, "
    f"train_batch={train_batch_size}, "
    f"grad_accum={gradient_accumulation_steps}, "
    f"microbatch={train_batch_size // gradient_accumulation_steps}",
    flush=True,
)

cliprange = args.cliprange
if cliprange is None:
    cliprange = {
        "grpo": 0.2,
        "gspo": 3e-4,
    }.get(args.importance_method)

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

def log_gpu_memory(stage):
    # 等训练 GPU 上已提交的计算完成，使记录更准确。
    torch.cuda.synchronize(train_device)
    gib = 1024 ** 3

    stats = {
        "stage": stage,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_allocated_GiB": round(
            torch.cuda.memory_allocated(train_device) / gib, 3
        ),
        "train_reserved_GiB": round(
            torch.cuda.memory_reserved(train_device) / gib, 3
        ),
        "train_peak_allocated_GiB": round(
            torch.cuda.max_memory_allocated(train_device) / gib, 3
        ),
    }
    print("[MEMORY] " + json.dumps(stats), flush=True)

    # 同时观察独立的 vLLM 进程。
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        print(
            result.stdout if result.returncode == 0 else result.stderr,
            flush=True,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[MEMORY] nvidia-smi unavailable: {exc}", flush=True)


def main():
    torch.manual_seed(seed)
    torch.cuda.set_device(train_device)

    run_dir = Path("runs") / (
        f"grpo_{args.policy_mode}_{args.importance_method}_"
        f"lr{args.lr:g}_seed{seed}_"
        + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
        
    run_dir.mkdir(parents=True, exist_ok=False)

    config = {
        "model_id": MODEL_ID,
        "train_device": train_device,
        "learning_rate": args.lr,
        "inference_gpu": args.inference_gpu,
        "policy_mode": args.policy_mode,

        "num_steps": args.num_steps,
        "prompts_per_batch": PROMPTS_PER_BATCH,
        "rollout_batch_size": PROMPTS_PER_BATCH * GROUP_SIZE,
        "train_batch_size": train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "group_size": GROUP_SIZE,

        "importance_reweighting_method": args.importance_method,
        "cliprange": cliprange,
        "loss_normalization": "sequence",       
        "max_grad_norm": 1.0,
        "seed": seed,
        "port": args.port,
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 512,
        "prompt_path": "cs336_alignment/prompts/r1_zero.prompt",
        "reward_fn": "r1_zero_reward_fn",
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
        "cs336_alignment/prompts/r1_zero.prompt"
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
        log_gpu_memory("before_server_start")

        server.start()
        log_gpu_memory("after_server_start")
        print("vLLM server ready.", flush = True)

        server.init_weight_sync(policy_device=train_device)
        log_gpu_memory("after_init_weight_sync")

        server.sync_policy_weights(model)
        log_gpu_memory("after_initial_weight_sync")
        print("Initial policy weights synchronized.", flush=True)
        run_validation(0)
        log_gpu_memory("after_initial_validation")


        sampling_params = {
            "temperature": 1.0,
            "top_p": 1.0,
            'max_tokens': 512,
            "n": 1,
            "seed": seed,
            "stop": ["</answer>"],
            "include_stop_str_in_output": True,
        }

        train_call_count = 0
        for step in range(args.num_steps):
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

            # Rollout
            completions = server.generate_completions(
                prompts = prompts,
                sampling_params=sampling_params,
            )
            rollout_responses = [completion.text for completion in completions]
            if step == 0:
                log_gpu_memory("after_first_rollout_generation")


            assert len(rollout_responses) == len(batch) * GROUP_SIZE

            old_log_probs = None
            if args.importance_method != 'none':
                tokenized = tokenize_prompt_and_output(
                    repeated_prompts, rollout_responses, tokenizer
                )

                old_log_probs_parts = []
                was_training = model.training
                model.eval()

                with torch.no_grad():
                    for start in range(0, len(rollout_responses), old_log_probs_batch_size):
                        end = start + old_log_probs_batch_size
                        result = get_response_log_probs(
                            model,
                            tokenized['input_ids'][start:end].to(train_device),
                            tokenized['labels'][start:end].to(train_device), 
                            return_token_entropy=False
                        )
                        old_log_probs_parts.append(
                            result['log_probs'].detach().cpu()  # [TRAIN_BATCH_SIZE, S]
                        )
                    model.train(was_training)
                    old_log_probs = torch.cat(old_log_probs_parts, dim = 0)


            if (step + 1) % 40 == 0:
                rollout_dir = run_dir / "rollouts"
                rollout_dir.mkdir(parents=True, exist_ok=True)

                rollout_path = rollout_dir / f"step_{step + 1:04d}.jsonl"

                with rollout_path.open("w", encoding="utf-8") as file:
                    for j, completion in enumerate(completions):
                        example = batch[j // GROUP_SIZE]
                        scores = r1_zero_reward_fn(
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

            
            for start in range(0, len(rollout_responses), train_batch_size):
                end = start + train_batch_size

                minibatch_prompts = repeated_prompts[start:end]
                minibatch_responses = rollout_responses[start:end]
                minibatch_truths = repeated_ground_truths[start:end]

                minibatch_old_probs = None
                if old_log_probs is not None:
                    minibatch_tokenized = tokenize_prompt_and_output(
                        minibatch_prompts, minibatch_responses, tokenizer
                    )
                    sequence_length = minibatch_tokenized['input_ids'].shape[1]

                    minibatch_old_probs = (
                        old_log_probs[start:end, :sequence_length]
                    )
                loss, metadata = grpo_train_step(
                    model = model,
                    tokenizer = tokenizer,
                    optimizer = optimizer,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    max_grad_norm = 1.0,
                    reward_fn=r1_zero_reward_fn,
                    repeated_prompts=minibatch_prompts,
                    rollout_responses=minibatch_responses,
                    repeated_ground_truths=minibatch_truths,
                    group_size=GROUP_SIZE,
                    importance_reweighting_method=args.importance_method,
                    old_log_probs=minibatch_old_probs,
                    cliprange=cliprange,
                    loss_normalization = "sequence",
                )

                train_call_count += 1
                num_minibatches = rollout_batch_size // train_batch_size
                record = {
                    "step": train_call_count,           # 累计训练调用次数；通常每次对应一次参数更新
                    "rollout_iterations": step + 1,     # 一个rollout_iteration产生  M 条rollout
                    "minibatch_index": start // train_batch_size,
                    # 当前采样轮内的 minibatch 编号（从 0 开始，当前为 0～31）
                    "loss": loss.item(),
                    **metadata,
                }
                print(
                    f"Rollout {step + 1}/{args.num_steps} | "
                    f"Minibatch {start // train_batch_size + 1}/{num_minibatches} | "
                    f"loss={loss.item():.6f}",
                    flush=True,
                )

                with (run_dir / "metrics.jsonl").open(
                    "a", encoding="utf-8"
                ) as file:
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")

            # 内层 minibatch 循环已结束，尚未同步权重
            if step == 0:
                log_gpu_memory("after_first_rollout_training")
            server.sync_policy_weights(model)
            if step == 0:
                log_gpu_memory("after_first_rollout_weight_sync")

            print("Policy weights synchronized.", flush=True)

            if (step + 1) % EVAL_EVERY == 0 or (step + 1) == args.num_steps:
                run_validation(step + 1)

        checkpoint_dir = run_dir / "final_checkpoint"
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"Checkpoint saved: {checkpoint_dir}", flush=True)


    finally:
        server.stop()

if __name__ == "__main__":
    main()
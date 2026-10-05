import json
import random, torch
from pathlib import Path

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.grpo_utils import grpo_train_step

# 仅用于之后跑通流程，不是正式实验配置
NUM_STEPS = 3
PROMPTS_PER_BATCH = 2
GROUP_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4

from cs336_alignment.vllm_utils import VLLMServer
from cs336_alignment.drgrpo_grader import (question_only_reward_fn, r1_zero_reward_fn)

from cs336_alignment.checkpoint import get_model_and_tokenizer
from cs336_alignment.vllm_utils import VLLMServer

MODEL_ID = "allenai/OLMo-2-0425-1B"
# 改成你实际可用的两张卡
TRAIN_DEVICE = "cuda:0"
INFERENCE_GPU = 1
PORT = 18080

def load_training_examples():
    examples = []
    with Path("data/gsm8k/train.jsonal").open(
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
        gpu_memory_utilization=0.8
    )

    try:
        server.start()
        print("vLLM server ready.", flush = True)

        server.init_weight_sync(policy_device=TRAIN_DEVICE)

        server.sync_policy_weights(model)
        print("Policy weights synchronized.", flush=True)

    finally:
        server.stop()

if __name__ == "__main__":
    main()
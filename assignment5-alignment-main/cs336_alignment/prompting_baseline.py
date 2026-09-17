import json
from pathlib import Path

from cs336_alignment.vllm_utils import VLLMServer
from cs336_alignment.drgrpo_grader import (question_only_reward_fn, r1_zero_reward_fn)

MODEL_ID = "allenai/OLMo-2-0425-1B"
GPU_iD = 1
PORT = 18080

PROMPT_CONFIGS = {
    "question_only": {
        "prompt_path": "cs336_alignment/prompts/question_only.prompt",
        "reward_fn": question_only_reward_fn,
        "stop": None,
    },
    "r1_zero": {
        "prompt_path": "cs336_alignment/prompts/r1_zero.prompt",
        "reward_fn": r1_zero_reward_fn,
        "stop": ["</answer>"],
    },
    "r1_zero_three_shot": {
        "prompt_path": (
            "cs336_alignment/prompts/"
            "r1_zero_three_shot_gsm8k.prompt"
        ),
        "reward_fn": r1_zero_reward_fn,
        "stop": ["</answer>"],
    },
}

def load_first_example():
    data_path = Path("data/gsm8k/test.jsonl")

    with data_path.open("r", encoding="utf-8") as file:
        example = json.loads(next(file))

    question = example["question"]
    ground_truth = example["answer"].split("####")[-1].strip()

    return question, ground_truth

def run_one_prompt(server, prompt_name, question, ground_truth):
    config = PROMPT_CONFIGS[prompt_name]

    prompt_template = Path(
        config["prompt_path"]
    ).read_text(encoding="utf-8")

    prompt = prompt_template.format(question = question)

    sampling_params={
    "temperature": 1.0,
    "max_tokens": 512,
    "n":1,
    "seed":0,
}
    if config['stop'] is not None:
        sampling_params['stop'] = config['stop']
        sampling_params['include_stop_str_in_output'] = True

    completion = server.generate_completions(
        prompts=[prompt],
        sampling_params = sampling_params,
    )[0]

    reward = config['reward_fn'](
        response = completion.text,
        ground_truth = ground_truth,
    )

    return {
        "prompt_name": prompt_name,
        "question": question,
        "ground_truth": ground_truth,
        "prompt": prompt,
        "response": completion.text,
        "finish_reason": completion.finish_reason,
        "reward": reward,
    }


def main():
    question, ground_truth = load_first_example()

    server = VLLMServer(
        model_id=MODEL_ID,
        gpu = GPU_iD,
        port = PORT,
        seed=0,
        gpu_memory_utilization=0.8
    )
    try:
        server.start()
        for prompt_name in PROMPT_CONFIGS:
            result = run_one_prompt(
                server=server,
                prompt_name=prompt_name,
                question=question,
                ground_truth=ground_truth,
            )
            print("=" * 80)
            print("Prompt type:", result["prompt_name"])
            print("Question:")
            print(result["question"])
            print("Ground truth:", result["ground_truth"])
            print("Response:")
            print(result["response"])
            print("Finish reason:", result["finish_reason"])
            print("Reward:", result["reward"])

    finally:
        server.stop()


if __name__ == "__main__":
    main()
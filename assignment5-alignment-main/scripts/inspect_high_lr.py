import json
from pathlib import Path

run_names = [
    "grpo_lr3e-05_seed0_20261007_133315_529385",
    "grpo_lr3e-05_seed1_20261007_141515_653795",
]

for name in run_names:
    run = Path("runs") / name
    val_rows = [
        json.loads(line)
        for line in (run / "val_metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    final = max(val_rows, key=lambda row: row["step"])
    print("\nRUN:", name)
    print("Final validation:", final)

    path = run / "eval/step_0200/responses.jsonl"
    with path.open(encoding="utf-8") as file:
        for index, line in enumerate(file):
            if index >= 5:
                break

            row = json.loads(line)
            print("\nQuestion:", row["question"])
            print("Ground truth:", row["ground_truth"])
            print("Response:", row["response"])
            print(
                "Format:", row["format_reward"],
                "Reward:", row["reward"],
                "Tokens:", row.get("response_tokens"),
                "Finish:", row["finish_reason"],
            )
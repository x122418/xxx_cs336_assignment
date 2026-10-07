import json
import re
from pathlib import Path

for run in sorted(Path("runs").glob("grpo_lr*_seed*")):
    match = re.match(r"grpo_lr(.+)_seed(\d+)_", run.name)
    if match is None:
        continue

    val_path = run / "val_metrics.jsonl"
    rows = [
        json.loads(line)
        for line in val_path.read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        continue

    final = max(rows, key=lambda row: row["step"])
    best = max(rows, key=lambda row: row["val_reward"])

    print(run.name)
    print("LR label:", match[1], "seed:", match[2])
    print(
        f"Final step {final['step']}: {final['val_reward']:.2%} | "
        f"Best: {best['val_reward']:.2%} at step {best['step']}"
    )
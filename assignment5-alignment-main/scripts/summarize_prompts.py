import json
from pathlib import Path

paths = [
    Path("runs/grpo_seed0_20261007_011350_002768"),
    *sorted(Path("runs").glob("grpo_prompt_*")),
]

for run in paths:
    config_path = run / "config.json"
    val_path = run / "val_metrics.jsonl"
    train_path = run / "metrics.jsonl"

    if not all(p.exists() for p in [config_path, val_path, train_path]):
        continue

    config = json.loads(config_path.read_text())
    vals = [
        json.loads(line)
        for line in val_path.read_text().splitlines()
        if line.strip()
    ]
    train = [
        json.loads(line)
        for line in train_path.read_text().splitlines()
        if line.strip()
    ]
    if not vals:
        continue

    final = max(vals, key=lambda row: row["step"])
    if final["step"] != 200:
        continue  # 排除两轮 smoke

    initial = next(row for row in vals if row["step"] == 0)
    best = max(vals, key=lambda row: row["val_reward"])
    prompt = config.get("prompt_type", "r1_zero")

    print("\nRun:", run.name)
    print(
        "Prompt:", prompt,
        "Seed:", config["seed"],
        "LR:", config["learning_rate"],
        "Train records:", len(train),
        "Val records:", len(vals),
    )
    print(
        f"Initial: {initial['val_reward']:.2%} | "
        f"Final: {final['val_reward']:.2%} | "
        f"Improvement: "
        f"{100 * (final['val_reward'] - initial['val_reward']):.2f} pp"
    )
    print(
        f"Best: {best['val_reward']:.2%} at step {best['step']} | "
        f"Final format: {final['val_format_reward']:.2%} | "
        f"Final length: {final['val_mean_response_tokens']:.1f} tokens"
    )
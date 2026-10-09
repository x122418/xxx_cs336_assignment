import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUNS = {
    "GRPO_constant": "grpo_variant_grpo_constant_lr1e-05_seed0_20261008_101245_916311",
    "Dr_GRPO": "grpo_variant_dr_grpo_lr1e-05_seed0_20261008_103623_879591",
    "RFT": "grpo_variant_rft_lr1e-05_seed0_20261008_105923_805893",
    "MaxRL": "grpo_variant_maxrl_lr1e-05_seed0_20261008_111811_498820",
}


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]
    return sorted(rows, key=lambda row: row["step"])


def get_metric(row, key):
    # 兼容指标直接保存在记录中，或放在 metadata 中。
    if key in row:
        return row[key]
    return row.get("metadata", {}).get(key)


def plot_metric(data, key, title, ylabel, path, scale=1.0):
    fig, ax = plt.subplots(figsize=(9, 5))

    for name, rows in data.items():
        points = []
        for row in rows:
            value = get_metric(row, key)
            if value is not None:
                points.append((row["step"], value * scale))

        if not points:
            print(f"WARNING: {name} missing metric {key}")
            continue

        steps, values = zip(*points)
        ax.plot(steps, values, label=name, linewidth=1.6)

    ax.set(
        title=title,
        xlabel="Rollout iteration",
        ylabel=ylabel,
    )
    ax.grid(alpha=0.25)
    if ax.lines:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    output_dir = Path("analysis") / (
        "grpo_variants_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    train_data = {}
    val_data = {}

    for name, directory in RUNS.items():
        run_dir = Path("runs") / directory
        train_rows = read_jsonl(run_dir / "metrics.jsonl")
        val_rows = read_jsonl(run_dir / "val_metrics.jsonl")

        if not train_rows or not val_rows:
            raise ValueError(f"{name}: empty training or validation log")

        train_data[name] = train_rows
        val_data[name] = val_rows

        initial = val_rows[0]
        final = val_rows[-1]
        best = max(val_rows, key=lambda row: row["val_reward"])

        print(
            f"{name}: train={len(train_rows)}, val={len(val_rows)}, "
            f"last_train_step={train_rows[-1]['step']}\n"
            f"  Initial={initial['val_reward']:.2%}, "
            f"Final={final['val_reward']:.2%}, "
            f"Best={best['val_reward']:.2%} at step {best['step']}"
        )

    plots = [
        (
            val_data, "val_reward",
            "Validation accuracy — seed 0", "Accuracy (%)",
            "validation_accuracy.png", 100,
        ),
        (
            val_data, "val_mean_response_tokens",
            "Validation response length — seed 0", "Mean response tokens",
            "validation_length.png", 1,
        ),
        (
            train_data, "mean_reward",
            "Train reward — seed 0", "Mean reward",
            "train_reward.png", 1,
        ),
        (
            train_data, "active_token_entropy",
            "Token entropy on retained sequences — seed 0",
            "Active-token entropy",
            "active_token_entropy.png", 1,
        ),
    ]

    for data, key, title, ylabel, filename, scale in plots:
        plot_metric(
            data, key, title, ylabel,
            output_dir / filename, scale,
        )

    print(f"\nSaved figures to: {output_dir}")


if __name__ == "__main__":
    main()
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUNS = {
    "Question-only": "grpo_prompt_question_only_lr1e-05_seed0_20261007_163752_980657",
    "R1 zero-shot": "grpo_seed0_20261007_011350_002768",
    "R1 three-shot": "grpo_prompt_r1_zero_three_shot_lr1e-05_seed0_20261007_171348_024447",
}

METRICS = [
    ("val", "val_reward", "Validation reward"),
    ("train", "mean_reward", "Train reward"),
    ("train", "zero_variance_group_fraction", "Zero-variance group fraction"),
    ("train", "token_entropy", "Response token entropy"),
    ("val", "val_mean_response_tokens", "Validation response length"),
    ("val", "val_format_reward", "Format / answer-extraction rate"),
]


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main():
    output_dir = Path("analysis") / datetime.now().strftime(
        "prompt_comparison_%Y%m%d_%H%M%S_%f"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    data = {}
    summary = []

    for label, name in RUNS.items():
        run = Path("runs") / name
        train = read_jsonl(run / "metrics.jsonl")
        val = read_jsonl(run / "val_metrics.jsonl")

        for records in [train, val]:
            steps = [row["step"] for row in records]
            assert len(steps) == len(set(steps)), f"{label}: duplicate steps"

        train = sorted(train, key=lambda row: row["step"])
        val = sorted(val, key=lambda row: row["step"])
        assert val[0]["step"] == 0 and val[-1]["step"] == 200

        data[label] = {"train": train, "val": val}

        initial = val[0]["val_reward"]
        final = val[-1]["val_reward"]
        best = max(val, key=lambda row: row["val_reward"])

        summary.append({
            "prompt": label,
            "run": name,
            "initial_reward": initial,
            "final_reward": final,
            "improvement_pp": 100 * (final - initial),
            "best_reward": best["val_reward"],
            "best_step": best["step"],
            "final_length": val[-1]["val_mean_response_tokens"],
            "final_format_rate": val[-1]["val_format_reward"],
        })

    for source, key, title in METRICS:
        fig, ax = plt.subplots(figsize=(7, 4))

        for label, records in data.items():
            rows = records[source]
            missing = [row["step"] for row in rows if key not in row]
            if missing:
                raise ValueError(f"{label}: {key} missing at {missing}")

            ax.plot(
                [row["step"] for row in rows],
                [row[key] for row in rows],
                label=label,
                linewidth=1.5,
            )

        ax.set_title(f"{title} — seed 0")
        ax.set_xlabel("Rollout iteration / optimizer step")
        ax.set_ylabel(title)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{key}.png", dpi=160)
        plt.close(fig)

    # 去掉各自起点后的增益曲线，作为原始reward图的补充。
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, records in data.items():
        rows = records["val"]
        initial = rows[0]["val_reward"]
        ax.plot(
            [row["step"] for row in rows],
            [100 * (row["val_reward"] - initial) for row in rows],
            label=label,
        )
    ax.set_title("Validation improvement from own baseline — seed 0")
    ax.set_xlabel("Rollout iteration / optimizer step")
    ax.set_ylabel("Improvement (percentage points)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "validation_improvement.png", dpi=160)
    plt.close(fig)

    with (output_dir / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
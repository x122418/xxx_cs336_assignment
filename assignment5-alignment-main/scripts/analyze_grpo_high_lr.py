import csv
import json
import math
import statistics
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 服务器无图形界面
import matplotlib.pyplot as plt

# 只分析四个完整运行，排除之前的 smoke 和失败启动。
RUN_NAMES = [
    "grpo_lr3e-05_seed0_20261007_133315_529385",
    "grpo_lr3e-05_seed1_20261007_141515_653795",
]

METRICS = [
    ("train", "loss", "Policy loss"),
    ("train", "grad_norm", "Gradient norm (before clipping)"),
    ("train", "token_entropy", "Response token entropy"),
    ("train", "mean_reward", "Train reward"),
    ("train", "mean_format_reward", "Train format reward"),
    ("val", "val_reward", "Validation reward"),
    ("val", "val_format_reward", "Validation format reward"),
    ("val", "val_mean_response_tokens", "Validation response length"),
    ("train", "zero_variance_group_fraction", "Zero-variance group fraction"),
]

def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def index_records(records, label):
    indexed = {}
    for record in records:
        step = record["step"]
        if step in indexed:
            raise ValueError(f"{label}: duplicate step {step}")

        for key, value in record.items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"{label}: step {step}, {key}={value}")

        indexed[step] = record
    return indexed

def main():
    output_dir = Path("analysis") / datetime.now().strftime(
        "standard_grpo_%Y%m%d_%H%M%S_%f"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    runs = []
    checks = []
    summary_rows = []

    for name in RUN_NAMES:
        run_dir = Path("runs") / name
        config = {
            "seed": int(name.split("_seed")[1].split("_")[0]),
            "num_steps": 200,
            "learning_rate": 3e-5,  # 来自目录标签，仍需核对启动日志
        }
        seed = config["seed"]

        train = index_records(
            read_jsonl(run_dir / "metrics.jsonl"), f"seed {seed} train"
        )
        val = index_records(
            read_jsonl(run_dir / "val_metrics.jsonl"), f"seed {seed} val"
        )

        expected_steps = config["num_steps"]
        if set(train) != set(range(1, expected_steps + 1)):
            raise ValueError(f"seed {seed}: incomplete training steps")
        if 0 not in val or expected_steps not in val:
            raise ValueError(f"seed {seed}: missing initial/final validation")

        for source, key, _ in METRICS:
            records = train if source == "train" else val
            missing = [step for step, row in records.items() if key not in row]
            if missing:
                checks.append(
                    f"WARNING seed {seed}: {key} missing at steps {missing}"
                )

        initial = val[0]["val_reward"]
        final = val[expected_steps]["val_reward"]
        # 并列最高时取最早的 step。
        best_step = max(sorted(val), key=lambda step: val[step]["val_reward"])

        summary_rows.append({
            "seed": seed,
            "run": name,
            "initial_accuracy": initial,
            "final_accuracy": final,
            "improvement_percentage_points": 100 * (final - initial),
            "best_accuracy": val[best_step]["val_reward"],
            "best_step": best_step,
            "final_format_rate": val[expected_steps]["val_format_reward"],
            "final_response_tokens": val[expected_steps][
                "val_mean_response_tokens"
            ],
        })

        checks.append(
            f"seed {seed}: {len(train)} train records, "
            f"{len(val)} validation records, final step {expected_steps}"
        )
        runs.append({
            "seed": seed,
            "config": config,
            "train": train,
            "val": val,
        })

    if len({run["seed"] for run in runs}) != len(runs):
        raise ValueError("Duplicate seeds")

    # 检查主要实验配置；缺失的配置明确记录，不能默认一致。
    compare_keys = [
        "model_id", "num_steps", "prompts_per_batch", "group_size",
        "gradient_accumulation_steps", "learning_rate", "max_grad_norm",
        "temperature", "top_p", "max_tokens", "prompt_path", "reward_fn",
        "baseline", "advantage_normalizer", "n_train_examples",
        "n_val_examples", "eval_every", "eval_temperature", "eval_seed",
    ]
    for key in compare_keys:
        values = [run["config"].get(key, "<missing>") for run in runs]
        if any(value == "<missing>" for value in values):
            checks.append(f"WARNING config {key}: {values}")
        elif any(value != values[0] for value in values[1:]):
            checks.append(f"WARNING config differs: {key}: {values}")

    # 每个指标一张图：四个 seed + 相同步数上的均值及 min/max。
    for source, key, title in METRICS:
        fig, ax = plt.subplots(figsize=(7, 4))
        series = []

        for run in runs:
            points = {
                step: row[key]
                for step, row in run[source].items()
                if key in row
            }
            series.append(points)
            steps = sorted(points)
            ax.plot(
                steps, [points[step] for step in steps],
                alpha=0.55, linewidth=1,
                label=f"seed {run['seed']}",
            )

        common_steps = sorted(set.intersection(*(set(s) for s in series)))
        if common_steps:
            values = [
                [s[step] for s in series] for step in common_steps
            ]
            ax.plot(
                common_steps,
                [statistics.mean(v) for v in values],
                color="black", linewidth=2, label="Mean",
            )
            ax.fill_between(
                common_steps,
                [min(v) for v in values],
                [max(v) for v in values],
                color="gray", alpha=0.15, label="Min–max",
            )

        ax.set_title(title)
        ax.set_xlabel("Rollout iteration / optimizer step")
        ax.set_ylabel(title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"{key}.png", dpi=160)
        plt.close(fig)

    with (output_dir / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    final_scores = [row["final_accuracy"] for row in summary_rows]
    aggregate = {
        "num_seeds": len(final_scores),
        "final_accuracy_mean": statistics.mean(final_scores),
        "final_accuracy_sample_std": statistics.stdev(final_scores),
        "mean_improvement_percentage_points": statistics.mean(
            row["improvement_percentage_points"] for row in summary_rows
        ),
    }
    (output_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    (output_dir / "checks.txt").write_text(
        "\n".join(checks) + "\n", encoding="utf-8"
    )

    print("\n".join(checks))
    print(json.dumps(aggregate, indent=2))
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
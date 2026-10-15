import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


manifests = sorted(
    Path("analysis").glob("offpolicy_curves_*/selected_runs.json")
)
if not manifests:
    raise FileNotFoundError("找不到 selected_runs.json")

manifest = manifests[-1]
offpolicy = json.loads(manifest.read_text(encoding="utf-8"))

# 明确选择四个正式运行，排除 smoke test。
onpolicy = [
    "runs/grpo_seed0_20261007_011350_002768",
    "runs/grpo_seed1_20261007_014530_820840",
    "runs/grpo_seed2_20261007_021641_625630",
    "runs/grpo_seed3_20261007_024703_016301",
]

out = Path("analysis") / datetime.now().strftime(
    "offpolicy_comparison_%Y%m%d_%H%M%S_%f"
)
out.mkdir(parents=True, exist_ok=False)

(out / "selected_runs.json").write_text(
    json.dumps(
        {"onpolicy": onpolicy, "offpolicy": offpolicy},
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=180)
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


# 1. Clip fraction：每轮先累加计数，再计算比例。
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)

for ax, method, title in zip(
    axes,
    ["grpo", "gspo"],
    ["GRPO: active-token clipping", "GSPO: active-sequence clipping"],
):
    for seed in range(4):
        run = Path(offpolicy[f"{method}/seed{seed}"])
        rows = read_jsonl(run / "metrics.jsonl")
        grouped = defaultdict(lambda: [0, 0])

        for r in rows:
            iteration = int(r["rollout_iterations"])
            count = r.get("clip_count")
            total = r.get("clip_total")

            if count is None or total is None:
                raise ValueError(f"缺少 clip 计数：{run}, step={r['step']}")
            if not 0 <= count <= total:
                raise ValueError(f"clip 计数异常：{run}, step={r['step']}")

            grouped[iteration][0] += count
            grouped[iteration][1] += total

        x = sorted(grouped)
        y = [
            100 * grouped[t][0] / grouped[t][1]
            if grouped[t][1] > 0 else np.nan
            for t in x
        ]
        ax.plot(x, y, label=f"seed {seed}", linewidth=1.2)

    ax.set_title(title)
    ax.set_xlabel("Rollout iteration")
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend()

axes[0].set_ylabel("Directional clip fraction (%)")
fig.suptitle("Zero-denominator iterations are missing, not zero")
save(fig, "clip_fraction")


# 2. 验证准确率：仅在四个 seed 都有记录的 step 上计算均值和 SD。
# 不为中断运行补零、不前向填充，也不偷偷改成幸存者均值。
groups = {"On-policy": [Path(p) for p in onpolicy]}
labels = {
    "grpo": "Off-policy GRPO",
    "gspo": "Off-policy GSPO",
    "noclip": "Off-policy no clipping",
    "none": "Off-policy no reweighting",
}

for method, label in labels.items():
    groups[label] = [
        Path(offpolicy[f"{method}/seed{s}"]) for s in range(4)
    ]

fig, ax = plt.subplots(figsize=(10, 6))

for label, runs in groups.items():
    curves = []
    for run in runs:
        rows = read_jsonl(run / "val_metrics.jsonl")
        curve = {int(r["step"]): 100 * r["val_reward"] for r in rows}
        if len(curve) != len(rows):
            raise ValueError(f"重复验证 step：{run}")
        curves.append(curve)

    common = sorted(set.intersection(*(set(c) for c in curves)))
    values = np.array([[c[t] for t in common] for c in curves])
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=1)

    display_label = label
    if common[-1] < 200:
        display_label += f" (4 seeds through {common[-1]})"

    line, = ax.plot(common, mean, label=display_label)
    ax.fill_between(
        common, mean - std, mean + std,
        color=line.get_color(), alpha=0.12,
    )

    print(
        f"{label}: step={common[-1]}, "
        f"mean={mean[-1]:.2f}%, sample SD={std[-1]:.2f} pp"
    )

ax.set(
    xlabel="Rollout iteration",
    ylabel="Validation accuracy (%)",
    title="Mean ± sample SD across 4 seeds",
    xlim=(0, 200),
    ylim=(0, 65),
)
ax.grid(alpha=0.25)
ax.legend()
save(fig, "onpolicy_vs_offpolicy")

print("使用运行列表：", manifest)
print("图像保存到：", out)
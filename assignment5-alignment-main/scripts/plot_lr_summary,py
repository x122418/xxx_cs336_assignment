from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 对应 seed 0、1；基线只用相同的两个 seed。
lrs = [3e-6, 1e-5, 3e-5]
seed0 = [40.8203125, 48.73046875, 0.0]
seed1 = [40.72265625, 50.5859375, 0.0]
means = [(a + b) / 2 for a, b in zip(seed0, seed1)]

output_dir = Path("analysis") / datetime.now().strftime(
    "lr_summary_%Y%m%d_%H%M%S_%f"
)
output_dir.mkdir(parents=True, exist_ok=False)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(lrs, seed0, "o--", alpha=0.5, label="Seed 0")
ax.plot(lrs, seed1, "o--", alpha=0.5, label="Seed 1")
ax.plot(lrs, means, "o-", color="black", linewidth=2, label="Mean")
ax.set_xscale("log")
ax.set_xticks(lrs)
ax.set_xticklabels(["3e-6", "1e-5", "3e-5"])
ax.set_xlabel("Learning rate")
ax.set_ylabel("Final validation accuracy (%)")
ax.set_title("Standard GRPO: fixed 200-step budget")
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(output_dir / "final_accuracy_vs_lr.png", dpi=160)
plt.close(fig)

print(output_dir / "final_accuracy_vs_lr.png")
# Plot training dynamics from runs/*/log_history.json (written by grpo_clipping_experiment.py).
# Usage:  python plot_pilot.py                     # all runs under runs/
#         python plot_pilot.py runs/token_wide_seed0 runs/seq_narrow_seed0
# Output: pilot_dynamics.png / .pdf  (5 panels, mirroring ARLArena Fig. 2)
import glob
import json
import os
import sys

import matplotlib.pyplot as plt

# (panel title, candidate substrings in TRL metric names, in priority order)
# Metric names differ across TRL versions, so we match the first one present.
PANELS = [
    ("Train accuracy (correctness reward)", ["rewards/correctness_reward/mean", "rewards/correctness_reward"]),
    ("KL to reference", ["kl"]),
    ("Gradient norm", ["grad_norm"]),
    ("Clipped-token fraction", ["clip_ratio/region_mean", "clip_ratio"]),
    ("Boxed-answer rate",["rewards/format_reward/mean", "rewards/format_reward"]),
]
EVAL_KEYS = ["eval_rewards/correctness_reward/mean", "eval_rewards/correctness_reward"]


def series(logs, keys):
    for k in keys:
        pts = [(r["step"], r[k]) for r in logs if k in r and "step" in r]
        if pts:
            return k, pts
    return None, []


def ema(ys, a=0.3):
    out, m = [], None
    for y in ys:
        m = y if m is None else a * y + (1 - a) * m
        out.append(m)
    return out


def main():
    dirs = sys.argv[1:] or sorted(glob.glob("runs/*"))
    runs = {}
    for d in dirs:
        p = os.path.join(d, "log_history.json")
        if os.path.exists(p):
            runs[os.path.basename(d.rstrip("/"))] = json.load(open(p))
    if not runs:
        sys.exit("no runs/*/log_history.json found")

    fig, axes = plt.subplots(1, 6, figsize=(22, 3.6))
    for i, (name, logs) in enumerate(runs.items()):
        c = f"C{i}"
        for ax, (title, keys) in zip(axes, PANELS):
            k, pts = series(logs, keys)
            if not pts:
                continue
            x, y = zip(*pts)
            if "format" in k:  # max format reward is 0.1 -> rescale to 0-1
                y = [v / 0.1 for v in y]
            ax.plot(x, y, color=c, alpha=0.25, lw=1)
            ax.plot(x, ema(y), color=c, lw=2, label=name)
            ax.set_title(title, fontsize=10)
            if title.startswith("Gradient"):
                ax.set_yscale("log")
        k, pts = series(logs, EVAL_KEYS)
        if pts:
            x, y = zip(*pts)
            axes[5].plot(x, y, "o-", color=c, lw=2, label=name)
        axes[5].set_title("Held-out GSM8K accuracy", fontsize=10)

    for ax in axes:
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"pilot_dynamics.{ext}", dpi=200, bbox_inches="tight")
    print("saved pilot_dynamics.png / .pdf")

    # print which metric keys were found, useful to check the TRL version mapping
    any_logs = next(iter(runs.values()))
    keys = sorted({k for r in any_logs for k in r})
    print("metric keys in log:", keys)


if __name__ == "__main__":
    main()

# Grouped bar chart: per-task success rate, pi vs gn16. Throwaway analysis script.
import glob
import json

import matplotlib.pyplot as plt
import numpy as np

# task label -> (pi glob, gn16 glob)
TASKS = [
    (
        "bagel_on_plate",
        "eval/0625_2k_bagel_plate_banana_bowl_linked/overnight2_runs/*/episode_results_rank0.jsonl",
        "eval/0625_1k_bagel_plate_banana_bowl_gn16/*/*/episode_results_rank0.jsonl",
    ),
    (
        "objects_in_box",
        "eval/0625_2k_bin_mug_marker_bowl_linked_pi/runs_jsonl/*/episode_results_rank0.jsonl",
        "eval/0625_1k_bin_mug_marker_bowl_gn16/*/*/episode_results_rank0.jsonl",
    ),
    (
        "two_bin_sort",
        "eval/0625_1.6k_two_bin_pi/*/*/episode_results_rank0.jsonl",
        "eval/06_25_1k_two_bin_gn16/**/episode_results_rank0.jsonl",
    ),
    (
        "mustard_on_box",
        "eval/0625_1k_mustard_raisin_pi/chunk*/episode_results_rank0.jsonl",
        "eval/0625_1k_mustard_raisin_box_gn16/*/*/episode_results_rank0.jsonl",
    ),
]


def rate(pattern):
    rows = [json.loads(line) for f in sorted(glob.glob(pattern, recursive=True)) for line in open(f) if line.strip()]
    n = len(rows)
    s = sum(bool(r["success"]) for r in rows)
    return s / n, n


labels = [t[0] for t in TASKS]
pi = [rate(t[1]) for t in TASKS]
gn = [rate(t[2]) for t in TASKS]
for lbl, (pp, pn), (gp, gn_) in zip(labels, pi, gn):
    print(f"{lbl:26s} pi {pp:.1%} (n={pn})   gn16 {gp:.1%} (n={gn_})")

x = np.arange(len(TASKS))
width = 0.38
figure, ax = plt.subplots(figsize=(11, 5.5))
b1 = ax.bar(x - width / 2, [100 * p for p, _ in pi], width, color="steelblue", label="pi")
b2 = ax.bar(x + width / 2, [100 * p for p, _ in gn], width, color="indianred", label="gn16")

for bars, data in ((b1, pi), (b2, gn)):
    for rect, (p, n) in zip(bars, data):
        ax.annotate(
            f"{p:.1%}\n(n={n})", (rect.get_x() + rect.get_width() / 2, rect.get_height()),
            textcoords="offset points", xytext=(0, 3), ha="center", fontsize=10,
        )

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylabel("success rate (%)", fontsize=13)
ax.set_ylim(0, 75)
ax.set_title("Policy success rate by task: pi vs gn16", fontsize=15, fontweight="bold")
ax.tick_params(axis="y", labelsize=11)
ax.legend(fontsize=12)
ax.grid(alpha=0.3, axis="y")
figure.tight_layout()
out = "eval/success_rate_pi_vs_gn16.png"
figure.savefig(out, dpi=150, bbox_inches="tight")
print(f"[INFO] wrote {out}")

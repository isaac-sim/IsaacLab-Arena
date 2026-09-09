# Overlay the wrist-camera posterior marginals of two datasets/models on shared axes.
# Throwaway comparison script (not part of the package).
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde

from isaaclab_arena.analysis.sensitivity.analyzer import SensitivityAnalyzer
from isaaclab_arena.analysis.sensitivity.episode_results_reader import dataset_from_episode_results

RUNS = [
    ("pi", "eval/0625_1k_mustard_raisin_pi_combined.jsonl", "steelblue"),
    ("gn16", "eval/0625_1k_mustard_raisin_box_gn16_combined.jsonl", "indianred"),
]
AXES = [f"droid_abs_joint_pos.camera_extrinsics_wrist_camera[{i}]" for i in range(3)]
# Camera ROS frame x_right / y_down / z_forward -> horizontal / vertical / depth.
AXIS_LABELS = ["horizontal (x)", "vertical (y)", "depth (z)"]
GRID = np.linspace(-0.05, 0.05, 200)  # shared ±0.05 sweep range for both runs

# Fit each run once and sample its success-conditioned posterior.
samples_by_run = {}
for label, path, _ in RUNS:
    torch.manual_seed(0)
    dataset = dataset_from_episode_results(path)
    analyzer = SensitivityAnalyzer(dataset)
    analyzer.fit()
    samples = analyzer.sample_posterior(dataset.default_observation()).cpu().numpy()
    cols = dataset.factor_columns
    n_succ = int(dataset.x[:, 0].sum())
    samples_by_run[label] = (samples, cols, n_succ)

figure, axes = plt.subplots(1, 3, figsize=(18, 4.5), squeeze=False)
for axis_index, axis_name in enumerate(AXES):
    ax = axes[0][axis_index]
    for label, _, color in RUNS:
        samples, cols, n_succ = samples_by_run[label]
        values = samples[:, cols[axis_name]].squeeze(-1)
        density = gaussian_kde(values)(GRID)
        ax.plot(GRID, density, color=color, linewidth=2, label=label)
        ax.fill_between(GRID, 0, density, color=color, alpha=0.12)
    ax.axhline(1.0 / 0.1, color="grey", linestyle="--", linewidth=1.3, label="prior (uniform)")
    ax.set_xlabel(f"wrist camera {AXIS_LABELS[axis_index]} displacement", fontsize=13)
    ax.set_ylabel("posterior density", fontsize=13)
    ax.set_xlim(-0.05, 0.05)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11)

figure.suptitle(
    "Wrist-camera posterior marginals (conditioned on success): pi vs gn16",
    fontsize=16,
    fontweight="bold",
)
figure.tight_layout(rect=[0, 0, 1, 0.95])
out = "eval/compare_pi_vs_gn16_camera.png"
figure.savefig(out, dpi=150, bbox_inches="tight")
print(f"[INFO] wrote {out}")

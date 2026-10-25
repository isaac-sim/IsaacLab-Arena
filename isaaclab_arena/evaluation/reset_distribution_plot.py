# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Overlay empirical reset occupancy on the first recorded scene configuration."""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path


def position_counts(positions: np.ndarray, bin_size_m: float = 0.01):
    """Return environment-aligned XY cell origins and the sample count in each cell."""
    values = np.asarray(positions, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Positions must be a nonempty finite N by 2 array")
    if not np.isfinite(bin_size_m) or bin_size_m <= 0:
        raise ValueError("Position bin size must be finite and positive")
    cells, counts = np.unique(np.floor(values / bin_size_m).astype(np.int64), axis=0, return_counts=True)
    return cells * bin_size_m, counts


def joint_counts(positions: np.ndarray, bin_size: float = 0.01):
    """Count joint offsets from the first sample in bins centered on multiples of bin_size."""
    values = np.asarray(positions, dtype=float)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError("Joint positions must be a nonempty finite N by J array")
    if not np.isfinite(bin_size) or bin_size <= 0:
        raise ValueError("Joint bin size must be finite and positive")
    delta = values - values[0]
    indices = np.floor(delta / bin_size + 0.5).astype(np.int64)
    lo, hi = int(indices.min()), int(indices.max())
    if hi - lo > 10000:
        raise ValueError("Too many joint bins; increase the joint bin size")
    counts = np.zeros((hi - lo + 1, values.shape[1]), dtype=int)
    for joint in range(values.shape[1]):
        counts[:, joint] = np.bincount(indices[:, joint] - lo, minlength=len(counts))
    return (np.arange(lo, hi + 1) - 0.5) * bin_size, counts, delta.min(0), delta.max(0)


def _load(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No reset samples: {path}")
    names = {row["run_name"] for row in rows}
    if len(names) != 1 or any(row.get("coordinate_frame") != "environment" for row in rows):
        raise ValueError("Each input must contain one run in environment coordinates")
    entities = {}
    for row in rows:
        for kind in ("articulation", "rigid_object"):
            for name, state in row["state"].get(kind, {}).items():
                entity = entities.setdefault((kind, name), {"poses": [], "joints": []})
                pose = np.asarray(state["root_pose"], dtype=float)
                if pose.shape != (7,) or not np.isfinite(pose).all():
                    raise ValueError(f"Invalid root pose: {name}")
                entity["poses"].append(pose)
                if "joint_position" in state:
                    entity["joints"].append(state["joint_position"])
    return rows[0]["run_name"], rows, entities


def render_reset_distribution(sample_paths, output_dir: Path, *, position_bin_m=0.01, joint_bin=0.01):
    """Write position and joint occupancy plots to a new directory, without modifying source runs.

    Args:
        sample_paths: One reset_samples.jsonl per run, in display order.
        output_dir: New directory for plots and their measurement conventions.
        position_bin_m: Width of environment-aligned XY bins in metres.
        joint_bin: Joint offset bin width in native units (radians or metres).

    Returns:
        Paths to the generated PNG images.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.cm import ScalarMappable, get_cmap
    from matplotlib.colors import BoundaryNorm
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    runs = [_load(path) for path in sample_paths]
    if not runs:
        raise ValueError("At least one sample file is required")
    # Validate numeric options before creating artifacts.
    position_counts([[0, 0]], position_bin_m)
    joint_counts([[0]], joint_bin)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    maximum = max(len(rows) for _, rows, _ in runs)
    cmap = get_cmap("viridis", maximum)
    norm = BoundaryNorm(np.arange(0.5, maximum + 1.5), maximum)
    fig = Figure(figsize=(6 * len(runs), 5.8), constrained_layout=True)
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, len(runs), squeeze=False)[0]
    metadata = {
        "reference": "first recorded reset for each entity; not a nominal configured pose",
        "position_bin_m": position_bin_m,
        "joint_bin_native_units": joint_bin,
        "colors": "sample count per bin, separately for each entity/joint",
        "range": "empirical min/max; not configured support or confidence interval",
        "runs": [],
    }
    for ax, (run_name, rows, entities), path in zip(axes, runs, sample_paths):
        metadata["runs"].append({
            "name": run_name,
            "source": str(Path(path).resolve()),
            "samples": len(rows),
            "first_sample": {k: rows[0][k] for k in ("rebuild_index", "env_id", "episode_in_env", "seed")},
        })
        all_xy = []
        for (kind, name), data in entities.items():
            xy = np.asarray(data["poses"])[:, :2]
            all_xy.extend(xy)
            origins, counts = position_counts(xy, position_bin_m)
            for origin, count in zip(origins, counts):
                ax.add_patch(
                    Rectangle(
                        origin, position_bin_m, position_bin_m, facecolor=cmap(norm(count)), edgecolor="none", alpha=0.9
                    )
                )
            low, high = xy.min(0), xy.max(0)
            ax.add_patch(Rectangle(low, *(high - low), fill=False, linestyle="--", edgecolor="#444444", lw=1.4))
            ax.scatter(*xy[0], marker="x", s=25, color="#e05c00", linewidth=1.2, zorder=5)
            label = name.replace("_hope_robolab", "").replace("_ycb_robolab", "").replace("_robolab", "")
            if kind == "articulation":
                label += " (base)"
            label = {"granola_bars": "granola box", "chocolate_pudding": "pudding box", "grey_bin": "bin"}.get(
                label, label
            )
            rightmost = name in ("counter", "grey_bin_robolab", "chocolate_pudding_ycb_robolab")
            ax.annotate(
                label,
                xy[0],
                xytext=(-7 if rightmost else 7, 8),
                ha="right" if rightmost else "left",
                textcoords="offset points",
                fontsize=9,
            )
        if all_xy:
            all_xy = np.asarray(all_xy)
            pad = max(0.06, float(np.ptp(all_xy, axis=0).max()) * 0.16)
            ax.set_xlim(all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad)
            ax.set_ylim(all_xy[:, 1].min() - pad, all_xy[:, 1].max() + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{run_name}\n{len(rows)} reset samples")
        ax.set_xlabel("Environment X (m)")
        ax.set_ylabel("Environment Y (m)")
        ax.grid(alpha=0.18)
    fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap), ax=list(axes), label="Occurrences per XY bin", ticks=range(1, maximum + 1)
    )
    fig.suptitle(
        f"Reset occupancy over initial configuration · {position_bin_m * 100:g} cm bins\n"
        "Orange ×: first reset   |   Dashed box: observed range   |   Robot markers: base positions",
        fontsize=13,
    )
    outputs = [output_dir / "positions.png"]
    fig.savefig(outputs[0], dpi=180)
    for run_index, (run_name, rows, entities) in enumerate(runs):
        robots = [(name, data) for (kind, name), data in entities.items() if kind == "articulation" and data["joints"]]
        if not robots:
            continue
        fig = Figure(figsize=(13, 3.8 * len(robots)), constrained_layout=True)
        FigureCanvasAgg(fig)
        axes = fig.subplots(len(robots), 1, squeeze=False)[:, 0]
        for ax, (name, data) in zip(axes, robots):
            edges, counts, low, high = joint_counts(data["joints"], joint_bin)
            edges = np.append(edges, edges[-1] + joint_bin)
            joints = np.arange(counts.shape[1])
            ax.pcolormesh(
                np.arange(counts.shape[1] + 1) - 0.5, edges, np.ma.masked_equal(counts, 0), cmap=cmap, norm=norm
            )
            ax.vlines(joints, low, high, color="#333333", lw=1.0)
            ax.scatter(joints, np.zeros_like(joints), marker="x", s=26, color="#e05c00", zorder=5)
            ax.set_xticks(joints)
            ax.set_xlabel("Joint index in recorded scene state")
            ax.set_ylabel("Offset from first reset\n(native joint units)")
            ax.set_title(name)
            ax.grid(axis="x", alpha=0.15)
        fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            ax=list(axes),
            label="Occurrences per joint bin",
            ticks=range(1, maximum + 1),
        )
        fig.suptitle(
            f"{run_name}: robot joint reset distribution\nOrange ×: first reset   |   Black line: observed range   |  "
            f" Bin width: {joint_bin:g}"
        )
        output = output_dir / f"joints_{run_index}.png"
        fig.savefig(output, dpi=170)
        outputs.append(output)
    (output_dir / "plot_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return outputs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--position-bin-m", type=float, default=0.01)
    parser.add_argument("--joint-bin", type=float, default=0.01)
    args = parser.parse_args()
    for path in render_reset_distribution(
        args.samples, args.output, position_bin_m=args.position_bin_m, joint_bin=args.joint_bin
    ):
        print(path)

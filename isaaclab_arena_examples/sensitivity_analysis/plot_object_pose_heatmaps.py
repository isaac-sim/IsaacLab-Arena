# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Plot reset/settled object occupancy and settled-position success rates."""

from __future__ import annotations

import argparse
import math
import matplotlib
import numpy as np
import re
from collections import defaultdict
from pathlib import Path

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from isaaclab_arena.visualization.episode_results_files import (  # noqa: E402
    find_episode_results_files,
    read_episode_results,
)

RESET_POSITIONS_KEY = "initial_reset_positions"
SETTLED_POSITIONS_KEY = "initial_rest_positions"
DEFAULT_GRID_SIZE_M = 0.05


def grid_edges(values: np.ndarray, grid_size_m: float) -> np.ndarray:
    """Return grid-aligned edges that contain all finite values."""
    assert grid_size_m > 0, f"grid_size_m must be positive, got {grid_size_m}"
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    assert values.size, "Cannot construct grid edges without finite values"
    low = math.floor(float(values.min()) / grid_size_m) * grid_size_m
    high = math.ceil(float(values.max()) / grid_size_m) * grid_size_m
    if math.isclose(low, high, abs_tol=grid_size_m * 1e-9):
        high = low + grid_size_m
    num_cells = max(1, int(round((high - low) / grid_size_m)))
    return low + np.arange(num_cells + 1, dtype=float) * grid_size_m


def bin_xy(
    xy: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    successes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Bin x/y positions into episode counts and optional per-cell success rates."""
    xy = np.asarray(xy, dtype=float)
    assert xy.ndim == 2 and xy.shape[1] == 2, f"xy must have shape (N, 2), got {xy.shape}"
    counts, _, _ = np.histogram2d(xy[:, 1], xy[:, 0], bins=(y_edges, x_edges))
    if successes is None:
        return counts, None

    successes = np.asarray(successes, dtype=float)
    assert successes.shape == (xy.shape[0],), f"successes must have shape ({xy.shape[0]},), got {successes.shape}"
    success_counts, _, _ = np.histogram2d(
        xy[:, 1],
        xy[:, 0],
        bins=(y_edges, x_edges),
        weights=successes,
    )
    success_rates = np.full(counts.shape, np.nan, dtype=float)
    np.divide(success_counts, counts, out=success_rates, where=counts > 0)
    return counts, success_rates


def _xy(position: object) -> tuple[float, float] | None:
    if not isinstance(position, list) or len(position) < 2:
        return None
    x, y = position[:2]
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return None
    if not math.isfinite(float(x)) or not math.isfinite(float(y)):
        return None
    return float(x), float(y)


def load_records_by_task(results_root: str | Path) -> dict[str, list[dict]]:
    """Load episode records beneath an Experiment output directory, grouped by task."""
    results_root = Path(results_root)
    files = [results_root] if results_root.is_file() else find_episode_results_files(results_root)
    assert files, f"No episode_results*.jsonl files found under {results_root}"

    records_by_task: dict[str, list[dict]] = defaultdict(list)
    issues = []
    for path in files:
        records, file_issues = read_episode_results(path, root=results_root)
        issues.extend(file_issues)
        for record in records:
            task_name = record.get("job_name")
            if not isinstance(task_name, str) or not task_name:
                task_name = path.parent.name
            records_by_task[task_name].append(record)
    for issue in issues:
        print(f"[WARNING] {issue.path}: {issue.message}")
    assert records_by_task, f"No valid episode records found under {results_root}"
    return dict(records_by_task)


def to_env_local_positions(records: list[dict]) -> list[dict]:
    """Subtract each episode's recorded environment origin from object positions."""
    local_records: list[dict] = []
    for record in records:
        env_origin = _xy(record.get("env_origin"))
        assert env_origin is not None, (
            f"Episode record for env {record.get('env_id')} is missing a valid env_origin; "
            "rerun the experiment with environment-origin recording enabled"
        )
        local_record = dict(record)
        for key in (RESET_POSITIONS_KEY, SETTLED_POSITIONS_KEY):
            positions = record.get(key)
            if not isinstance(positions, dict):
                continue
            local_positions = dict(positions)
            for name, value in positions.items():
                position = _xy(value)
                if position is None:
                    continue
                local_xy = np.asarray(position) - np.asarray(env_origin)
                local_positions[name] = [*local_xy.tolist(), *value[2:]]
            local_record[key] = local_positions
        local_records.append(local_record)
    return local_records


def object_names(records: list[dict]) -> list[str]:
    """Return sorted object names present in either recorded position mapping."""
    names: set[str] = set()
    for record in records:
        for key in (RESET_POSITIONS_KEY, SETTLED_POSITIONS_KEY):
            positions = record.get(key)
            if isinstance(positions, dict):
                names.update(name for name in positions if isinstance(name, str))
    return sorted(names)


def collect_object_samples(
    records: list[dict],
    object_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return reset x/y, settled x/y, and settled-sample success arrays."""
    reset_xy: list[tuple[float, float]] = []
    settled_xy: list[tuple[float, float]] = []
    settled_successes: list[bool] = []
    for record in records:
        reset_positions = record.get(RESET_POSITIONS_KEY)
        if isinstance(reset_positions, dict):
            position = _xy(reset_positions.get(object_name))
            if position is not None:
                reset_xy.append(position)

        settled_positions = record.get(SETTLED_POSITIONS_KEY)
        success = record.get("success")
        if isinstance(settled_positions, dict) and isinstance(success, bool):
            position = _xy(settled_positions.get(object_name))
            if position is not None:
                settled_xy.append(position)
                settled_successes.append(success)

    return (
        np.asarray(reset_xy, dtype=float).reshape(-1, 2),
        np.asarray(settled_xy, dtype=float).reshape(-1, 2),
        np.asarray(settled_successes, dtype=bool),
    )


def plot_object_heatmaps(
    task_name: str,
    object_name: str,
    records: list[dict],
    output_path: str | Path,
    grid_size_m: float = DEFAULT_GRID_SIZE_M,
) -> Path:
    """Plot reset occupancy, settled occupancy, and settled-position success rate."""
    reset_xy, settled_xy, settled_successes = collect_object_samples(records, object_name)
    combined_xy = np.concatenate([reset_xy, settled_xy], axis=0)
    assert combined_xy.size, f"No reset or settled positions found for {task_name}/{object_name}"

    x_edges = grid_edges(combined_xy[:, 0], grid_size_m)
    y_edges = grid_edges(combined_xy[:, 1], grid_size_m)
    reset_counts, _ = bin_xy(reset_xy, x_edges, y_edges)
    settled_counts, success_rates = bin_xy(settled_xy, x_edges, y_edges, settled_successes)
    assert success_rates is not None

    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharex=True, sharey=True)
    extent = (x_edges[0], x_edges[-1], y_edges[0], y_edges[-1])
    panels = (
        (reset_counts, "Reset-position occupancy", "Episodes", "Blues", None, None),
        (settled_counts, "Settled-position occupancy", "Episodes", "Blues", None, None),
        (np.ma.masked_invalid(success_rates), "Success rate by settled position", "Success rate", "RdYlGn", 0, 1),
    )
    for axis, (values, title, colorbar_label, cmap, vmin, vmax) in zip(axes, panels, strict=True):
        image = axis.imshow(
            values,
            origin="lower",
            extent=extent,
            interpolation="nearest",
            aspect="equal",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(title)
        axis.set_xlabel("Environment-local x [m]")
        axis.set_ylabel("Environment-local y [m]")
        axis.grid(False)
        colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label(colorbar_label)

    figure.suptitle(
        f"{task_name} — {object_name} — {grid_size_m * 100:g} cm grid\n"
        f"{len(reset_xy)} reset samples, {len(settled_xy)} settled samples",
        fontweight="bold",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.92])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_").lower()


def generate_heatmaps(
    results_root: str | Path,
    output_dir: str | Path,
    grid_size_m: float = DEFAULT_GRID_SIZE_M,
) -> list[Path]:
    """Generate one three-panel heatmap figure per task and recorded object."""
    records_by_task = load_records_by_task(results_root)
    output_dir = Path(output_dir)
    written: list[Path] = []
    for task_name, records in sorted(records_by_task.items()):
        records = to_env_local_positions(records)
        for object_name in object_names(records):
            output_path = output_dir / f"{_slug(task_name)}__{_slug(object_name)}.png"
            written.append(
                plot_object_heatmaps(
                    task_name,
                    object_name,
                    records,
                    output_path,
                    grid_size_m,
                )
            )
            print(f"[INFO] Wrote {output_path}")
    assert written, f"No object positions found under {results_root}"
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment_output",
        type=Path,
        required=True,
        help="Experiment output directory containing per-task episode_results*.jsonl files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Visualization directory (default: <experiment_output>/object_pose_heatmaps).",
    )
    parser.add_argument(
        "--grid_size_m",
        type=float,
        default=DEFAULT_GRID_SIZE_M,
        help="Square x/y grid size in metres (default: 0.05).",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or (args.experiment_output / "object_pose_heatmaps")
    generate_heatmaps(args.experiment_output, output_dir, args.grid_size_m)


if __name__ == "__main__":
    main()

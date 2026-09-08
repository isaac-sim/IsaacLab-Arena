# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Plot reset/settled object occupancy and settled-position success rates."""

from __future__ import annotations

import argparse
import json
import math
import matplotlib
import numpy as np
import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from isaaclab_arena.evaluation.arena_experiment_result import ARENA_EXPERIMENT_RESULT_FILENAME  # noqa: E402
from isaaclab_arena.visualization.episode_results_files import (  # noqa: E402
    find_episode_results_files,
    read_episode_results,
)

RESET_POSITIONS_KEY = "reset_positions"
SETTLED_POSITIONS_KEY = "settled_positions"
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
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Bin x/y positions, optionally weighting each sample."""
    xy = np.asarray(xy, dtype=float)
    assert xy.ndim == 2 and xy.shape[1] == 2, f"xy must have shape (N, 2), got {xy.shape}"
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        assert weights.shape == (xy.shape[0],), f"weights must have shape ({xy.shape[0]},), got {weights.shape}"
    histogram, _, _ = np.histogram2d(
        xy[:, 1],
        xy[:, 0],
        bins=(y_edges, x_edges),
        weights=weights,
    )
    return histogram


def _xy(position: object) -> tuple[float, float] | None:
    if not isinstance(position, list) or len(position) < 2:
        return None
    x, y = position[:2]
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return None
    if not math.isfinite(float(x)) or not math.isfinite(float(y)):
        return None
    return float(x), float(y)


def _load_experiment_result_records(result_path: Path) -> dict[str, list[dict]]:
    """Load embedded episodes from one canonical Arena Experiment result."""
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert isinstance(result, dict) and isinstance(
        result.get("runs"), dict
    ), f"{result_path} must contain a JSON object named 'runs'"
    records_by_run: dict[str, list[dict]] = {}
    for run_name, run in result["runs"].items():
        assert isinstance(run_name, str) and isinstance(run, dict), f"Invalid Run entry in {result_path}"
        rebuilds = run.get("rebuilds")
        assert isinstance(rebuilds, list), f"Run {run_name!r} must contain a rebuilds list"
        records: list[dict] = []
        for rebuild in rebuilds:
            assert isinstance(rebuild, dict) and isinstance(
                rebuild.get("episodes"), list
            ), f"Run {run_name!r} contains an invalid rebuild"
            for episode in rebuild["episodes"]:
                assert isinstance(episode, dict), f"Run {run_name!r} contains a non-object episode"
                records.append(episode)
        records_by_run[run_name] = records
    assert records_by_run, f"No Runs found in {result_path}"
    return records_by_run


def load_records_by_task(results_root: str | Path) -> dict[str, list[dict]]:
    """Load episode records from an Experiment JSON or output directory, grouped by Run."""
    results_root = Path(results_root)
    if results_root.name == ARENA_EXPERIMENT_RESULT_FILENAME:
        return _load_experiment_result_records(results_root)

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
                local_positions[name] = [position[0] - env_origin[0], position[1] - env_origin[1], *value[2:]]
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
    reset_counts = bin_xy(reset_xy, x_edges, y_edges)
    settled_counts = bin_xy(settled_xy, x_edges, y_edges)
    success_counts = bin_xy(settled_xy, x_edges, y_edges, settled_successes)
    success_rates = np.full(settled_counts.shape, np.nan, dtype=float)
    np.divide(success_counts, settled_counts, out=success_rates, where=settled_counts > 0)

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
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_").lower()


def _split_task_and_policy(run_name: str, policies: Sequence[str]) -> tuple[str, str]:
    """Split a Run name using the longest requested policy suffix."""
    for policy in sorted(policies, key=len, reverse=True):
        suffix = f"_{policy}"
        if run_name.endswith(suffix):
            task_name = run_name[: -len(suffix)]
            assert task_name, f"Run {run_name!r} has no task name before policy suffix {suffix!r}"
            return task_name, policy
    raise AssertionError(f"Run {run_name!r} does not end in one of the requested policy suffixes: {list(policies)}")


def generate_heatmaps(
    results_root: str | Path,
    output_dir: str | Path,
    grid_size_m: float = DEFAULT_GRID_SIZE_M,
    policies: Sequence[str] | None = None,
) -> list[Path]:
    """Generate one three-panel heatmap figure per task and recorded object."""
    records_by_task = load_records_by_task(results_root)
    output_dir = Path(output_dir)
    written: list[Path] = []
    for run_name, records in sorted(records_by_task.items()):
        task_name, policy = _split_task_and_policy(run_name, policies) if policies else (run_name, None)
        records = to_env_local_positions(records)
        policy_output_dir = output_dir / policy if policy is not None else output_dir
        for object_name in object_names(records):
            output_path = policy_output_dir / f"{_slug(task_name)}__{_slug(object_name)}.png"
            written.append(
                plot_object_heatmaps(
                    f"{task_name} [{policy}]" if policy is not None else task_name,
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
        help="arena_experiment_result.json or an output directory containing episode_results*.jsonl files.",
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
    parser.add_argument(
        "--policies",
        nargs="+",
        default=None,
        help="Policy suffixes used to group plots into per-policy subdirectories, e.g. pi0 cosmos.",
    )
    args = parser.parse_args()
    default_output_root = args.experiment_output.parent if args.experiment_output.is_file() else args.experiment_output
    output_dir = args.output_dir or (default_output_root / "object_pose_heatmaps")
    generate_heatmaps(args.experiment_output, output_dir, args.grid_size_m, args.policies)


if __name__ == "__main__":
    main()

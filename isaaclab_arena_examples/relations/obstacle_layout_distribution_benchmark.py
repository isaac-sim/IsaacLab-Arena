# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Generate and analyze layout distributions around fixed tabletop obstacles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import numpy as np
import platform
import random
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

Method = Literal["arena", "robolab", "random_rejection"]
TableBounds = tuple[float, float, float, float]
Size = tuple[float, float]
Layout = dict[str, tuple[float, float]]

TABLE_BOUNDS: TableBounds = (-0.5, 0.5, -0.5, 0.5)
MOVABLE_SIZES: dict[str, Size] = {
    "movable-small": (0.06, 0.06),
    "movable-medium": (0.09, 0.09),
    "movable-large": (0.12, 0.12),
}


@dataclass(frozen=True)
class FixedObstacle:
    """One fixed axis-aligned obstacle."""

    name: str
    x: float
    y: float
    width: float
    depth: float


OBSTACLES: tuple[FixedObstacle, ...] = (
    FixedObstacle("obstacle-northwest", -0.27, 0.25, 0.20, 0.18),
    FixedObstacle("obstacle-northeast", 0.22, 0.22, 0.16, 0.24),
    FixedObstacle("obstacle-center", -0.02, 0.00, 0.18, 0.16),
    FixedObstacle("obstacle-southwest", -0.25, -0.25, 0.14, 0.22),
    FixedObstacle("obstacle-southeast", 0.26, -0.24, 0.22, 0.14),
)


@dataclass(frozen=True)
class CoverageSummary:
    """Per-object marginal coverage statistics for one method."""

    method: Method
    object_name: str
    sample_count_per_repetition: int
    repetitions: int
    feasible_cells: int
    coverage: float
    coverage_iqr: float
    normalized_entropy: float
    normalized_entropy_iqr: float
    coverage_curve_auc: float
    coverage_curve_auc_iqr: float
    js_divergence_to_rejection: float
    js_divergence_to_rejection_iqr: float


def _git_output(repository_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository_root), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout


def _source_revision(repository_root: Path) -> str | None:
    try:
        commit = _git_output(repository_root, "rev-parse", "HEAD").strip()
        if not _git_output(repository_root, "status", "--porcelain"):
            return commit
        digest = hashlib.sha256(_git_output(repository_root, "diff", "--binary", "HEAD").encode())
        for relative_path in sorted(
            _git_output(repository_root, "ls-files", "--others", "--exclude-standard").splitlines()
        ):
            digest.update(relative_path.encode())
            digest.update((repository_root / relative_path).read_bytes())
        return f"{commit}+dirty.{digest.hexdigest()[:12]}"
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _overlap(
    first_xy: tuple[float, float],
    first_size: Size,
    second_xy: tuple[float, float],
    second_size: Size,
) -> bool:
    return (
        abs(first_xy[0] - second_xy[0]) < (first_size[0] + second_size[0]) / 2
        and abs(first_xy[1] - second_xy[1]) < (first_size[1] + second_size[1]) / 2
    )


def _fits_around_obstacles(
    xy: tuple[float, float],
    size: Size,
    table_bounds: TableBounds,
    obstacles: tuple[FixedObstacle, ...],
) -> bool:
    """Check yaw-zero containment and fixed-obstacle clearance."""
    x, y = xy
    width, depth = size
    xmin, xmax, ymin, ymax = table_bounds
    outside_table = x - width / 2 < xmin or x + width / 2 > xmax or y - depth / 2 < ymin or y + depth / 2 > ymax
    return not outside_table and not any(
        _overlap(xy, size, (obstacle.x, obstacle.y), (obstacle.width, obstacle.depth)) for obstacle in obstacles
    )


def validate_layout(
    layout: Layout,
    table_bounds: TableBounds = TABLE_BOUNDS,
    movable_sizes: dict[str, Size] = MOVABLE_SIZES,
    obstacles: tuple[FixedObstacle, ...] = OBSTACLES,
) -> bool:
    """Check exact yaw-zero containment and non-overlap."""
    if set(layout) != set(movable_sizes):
        return False
    for name, xy in layout.items():
        if not _fits_around_obstacles(xy, movable_sizes[name], table_bounds, obstacles):
            return False
    items = list(layout.items())
    return not any(
        _overlap(first_xy, movable_sizes[first_name], second_xy, movable_sizes[second_name])
        for index, (first_name, first_xy) in enumerate(items)
        for second_name, second_xy in items[index + 1 :]
    )


def _layout_key(layout: Layout) -> tuple[tuple[str, float, float], ...]:
    return tuple((name, round(x, 6), round(y, 6)) for name, (x, y) in sorted(layout.items()))


def _sample_rejection(seed: int) -> Layout:
    generator = random.Random(seed)
    xmin, xmax, ymin, ymax = TABLE_BOUNDS
    return {
        name: (
            generator.uniform(xmin + width / 2, xmax - width / 2),
            generator.uniform(ymin + depth / 2, ymax - depth / 2),
        )
        for name, (width, depth) in MOVABLE_SIZES.items()
    }


def _build_arena_assets():
    from isaaclab_arena.relations.benchmark.synthetic_benchmark import BenchmarkAsset
    from isaaclab_arena.relations.relations import IsAnchor, On
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose

    xmin, xmax, ymin, ymax = TABLE_BOUNDS
    table = BenchmarkAsset(
        "table",
        AxisAlignedBoundingBox(min_point=(xmin, ymin, -0.1), max_point=(xmax, ymax, 0.0)),
    )
    table.add_relation(IsAnchor())
    table.set_initial_pose(Pose.identity())
    assets = [table]
    for obstacle in OBSTACLES:
        asset = BenchmarkAsset(
            obstacle.name,
            AxisAlignedBoundingBox(
                min_point=(-obstacle.width / 2, -obstacle.depth / 2, -0.05),
                max_point=(obstacle.width / 2, obstacle.depth / 2, 0.05),
            ),
        )
        asset.add_relation(IsAnchor())
        asset.set_initial_pose(Pose(position_xyz=(obstacle.x, obstacle.y, 0.05), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
        assets.append(asset)
    for name, (width, depth) in MOVABLE_SIZES.items():
        asset = BenchmarkAsset(
            name,
            AxisAlignedBoundingBox(
                min_point=(-width / 2, -depth / 2, -0.05),
                max_point=(width / 2, depth / 2, 0.05),
            ),
        )
        asset.add_relation(On(table, clearance_m=0.0, edge_margin_m=0.0))
        assets.append(asset)
    return assets


def _sample_arena(seed: int, count: int, max_iterations: int) -> list[Layout]:
    from isaaclab_arena.relations.object_placer import ObjectPlacer
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams

    assets = _build_arena_assets()
    placer = ObjectPlacer(
        ObjectPlacerParams(
            solver_params=RelationSolverParams(
                max_iters=max_iterations,
                clearance_m=0.0,
                verbose=False,
                save_position_history=False,
            ),
            placement_seed=seed,
            max_placement_attempts=1,
            random_yaw_init=False,
            apply_positions_to_objects=False,
            verbose=False,
        )
    )
    movable_assets = {asset.name: asset for asset in assets if asset.name in MOVABLE_SIZES}
    results = placer.place(assets, num_envs=count)
    return [
        {
            name: (float(result.positions[asset][0]), float(result.positions[asset][1]))
            for name, asset in movable_assets.items()
        }
        for result in results
        if result.success
    ]


def _load_robolab(robolab_root: Path):
    sys.path.insert(0, str(robolab_root.resolve()))
    from robolab.scene_gen.llm_scene_gen.predicates import ObjectState, PlaceOnBasePredicate
    from robolab.scene_gen.llm_scene_gen.spatial_solver import SpatialSolver

    return ObjectState, PlaceOnBasePredicate, SpatialSolver


def _sample_robolab(seed: int, robolab_api, max_iterations: int) -> Layout | None:
    ObjectState, PlaceOnBasePredicate, SpatialSolver = robolab_api
    random.seed(seed)
    np.random.seed(seed % (2**32))
    states = {
        obstacle.name: ObjectState(
            name=obstacle.name,
            x=obstacle.x,
            y=obstacle.y,
            yaw=0.0,
            is_placed=True,
        )
        for obstacle in OBSTACLES
    }
    states.update(
        {name: ObjectState(name=name, predicates=[PlaceOnBasePredicate(name, yaw=0.0)]) for name in MOVABLE_SIZES}
    )
    dimensions = {obstacle.name: (obstacle.width, obstacle.depth, 0.1) for obstacle in OBSTACLES}
    dimensions.update({name: (*size, 0.1) for name, size in MOVABLE_SIZES.items()})
    solver = SpatialSolver(table_bounds=TABLE_BOUNDS, collision_margin=0.0)
    success, _ = solver.solve(
        states,
        dimensions,
        max_iterations=max_iterations,
        fixed_objects=[obstacle.name for obstacle in OBSTACLES],
        allow_relaxation=False,
    )
    if not success:
        return None
    return {
        name: (float(states[name].x), float(states[name].y))
        for name in MOVABLE_SIZES
        if states[name].x is not None and states[name].y is not None
    }


def _generate_repetition(
    method: Method,
    seed: int,
    target_layouts: int,
    max_attempts_per_layout: int,
    max_iterations: int,
    robolab_api=None,
) -> dict:
    unique: dict[tuple[tuple[str, float, float], ...], Layout] = {}
    attempted = accepted = 0
    maximum_attempts = target_layouts * max_attempts_per_layout
    start = time.perf_counter()
    while attempted < maximum_attempts and len(unique) < target_layouts:
        if method == "arena":
            batch_size = min(target_layouts - len(unique), maximum_attempts - attempted)
            candidates = _sample_arena(seed + attempted, batch_size, max_iterations)
            attempted += batch_size
        elif method == "robolab":
            candidate = _sample_robolab(seed + attempted, robolab_api, max_iterations)
            attempted += 1
            candidates = [] if candidate is None else [candidate]
        else:
            candidates = [_sample_rejection(seed + attempted)]
            attempted += 1
        for layout in candidates:
            if validate_layout(layout):
                accepted += 1
                unique.setdefault(_layout_key(layout), layout)
    elapsed_ms = (time.perf_counter() - start) * 1e3
    return {
        "seed": seed,
        "target_layouts": target_layouts,
        "target_reached": len(unique) == target_layouts,
        "elapsed_ms": elapsed_ms,
        "attempted_layouts": attempted,
        "accepted_layouts": accepted,
        "unique_layouts": len(unique),
        "layouts": [{name: [xy[0], xy[1]] for name, xy in layout.items()} for layout in unique.values()],
    }


def generate(args: argparse.Namespace) -> int:
    """Generate exact-K externally valid layouts."""
    robolab_api = _load_robolab(args.robolab_root) if args.method == "robolab" else None
    seed_stride = args.target_layouts * args.max_attempts_per_layout
    for warmup_index in range(args.warmup):
        _generate_repetition(
            args.method,
            args.seed - (warmup_index + 1) * seed_stride,
            min(args.target_layouts, 32),
            args.max_attempts_per_layout,
            args.max_iterations,
            robolab_api,
        )
    seeds = [args.seed + repeat * seed_stride for repeat in range(args.repetitions)]
    samples = [
        _generate_repetition(
            args.method,
            seed,
            args.target_layouts,
            args.max_attempts_per_layout,
            args.max_iterations,
            robolab_api,
        )
        for seed in seeds
    ]
    benchmark_root = Path(__file__).resolve().parents[2]
    source_root = args.robolab_root if args.method == "robolab" else benchmark_root
    payload = {
        "schema_version": 1,
        "workload": "fixed-obstacle-tabletop",
        "method": args.method,
        "source_revision": _source_revision(source_root),
        "benchmark_revision": _source_revision(benchmark_root),
        "table_bounds": TABLE_BOUNDS,
        "obstacles": [asdict(obstacle) for obstacle in OBSTACLES],
        "movable_sizes": MOVABLE_SIZES,
        "target_layouts": args.target_layouts,
        "repetitions": args.repetitions,
        "warmup": args.warmup,
        "seeds": seeds,
        "max_attempts_per_layout": args.max_attempts_per_layout,
        "max_iterations": args.max_iterations,
        "timing_scope": "complete-exact-k-generation",
        "host": platform.node(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "collision_model": {
            "arena": "aabb",
            "robolab": "max-xy-radius-circle",
            "random_rejection": "exact-aabb",
        }[args.method],
        "samples": samples,
    }
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    elapsed = [sample["elapsed_ms"] for sample in samples]
    rates = [sample["unique_layouts"] * 1e3 / sample["elapsed_ms"] for sample in samples]
    print(
        f"{args.method}: targets={sum(sample['target_reached'] for sample in samples)}/{len(samples)}, "
        f"median_ms={statistics.median(elapsed):.3f}, median_valid_layouts_per_s={statistics.median(rates):.3f}"
    )
    return 0


def _rectangle_has_unblocked_area(
    bounds: tuple[float, float, float, float],
    blocked: list[tuple[float, float, float, float]],
) -> bool:
    """Return whether an axis-aligned cell has positive area outside blocked rectangles."""
    xmin, xmax, ymin, ymax = bounds
    if xmin >= xmax or ymin >= ymax:
        return False
    x_breaks = {xmin, xmax}
    clipped = []
    for block_xmin, block_xmax, block_ymin, block_ymax in blocked:
        clipped_xmin = max(xmin, block_xmin)
        clipped_xmax = min(xmax, block_xmax)
        clipped_ymin = max(ymin, block_ymin)
        clipped_ymax = min(ymax, block_ymax)
        if clipped_xmin < clipped_xmax and clipped_ymin < clipped_ymax:
            clipped.append((clipped_xmin, clipped_xmax, clipped_ymin, clipped_ymax))
            x_breaks.update((clipped_xmin, clipped_xmax))
    ordered_x = sorted(x_breaks)
    for left, right in zip(ordered_x, ordered_x[1:]):
        if left >= right:
            continue
        midpoint = (left + right) / 2
        intervals = sorted(
            (block_ymin, block_ymax)
            for block_xmin, block_xmax, block_ymin, block_ymax in clipped
            if block_xmin < midpoint < block_xmax
        )
        covered_until = ymin
        for interval_min, interval_max in intervals:
            if interval_min > covered_until:
                return True
            covered_until = max(covered_until, interval_max)
        if covered_until < ymax:
            return True
    return False


def _feasible_mask(
    object_name: str,
    grid_size: int,
    table_bounds: TableBounds = TABLE_BOUNDS,
    movable_sizes: dict[str, Size] = MOVABLE_SIZES,
    obstacles: tuple[FixedObstacle, ...] = OBSTACLES,
) -> np.ndarray:
    xmin, xmax, ymin, ymax = table_bounds
    width, depth = movable_sizes[object_name]
    allowed = (xmin + width / 2, xmax - width / 2, ymin + depth / 2, ymax - depth / 2)
    blocked = [
        (
            obstacle.x - (obstacle.width + width) / 2,
            obstacle.x + (obstacle.width + width) / 2,
            obstacle.y - (obstacle.depth + depth) / 2,
            obstacle.y + (obstacle.depth + depth) / 2,
        )
        for obstacle in obstacles
    ]
    x_step = (xmax - xmin) / grid_size
    y_step = (ymax - ymin) / grid_size
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    for y_index in range(grid_size):
        for x_index in range(grid_size):
            cell = (
                max(allowed[0], xmin + x_index * x_step),
                min(allowed[1], xmin + (x_index + 1) * x_step),
                max(allowed[2], ymin + y_index * y_step),
                min(allowed[3], ymin + (y_index + 1) * y_step),
            )
            mask[y_index, x_index] = _rectangle_has_unblocked_area(cell, blocked)
    return mask


def _histogram(layouts: list[dict], object_name: str, grid_size: int, feasible: np.ndarray) -> np.ndarray:
    xmin, xmax, ymin, ymax = TABLE_BOUNDS
    histogram = np.zeros((grid_size, grid_size), dtype=np.float64)
    for layout in layouts:
        x, y = layout[object_name]
        x_index = min(grid_size - 1, max(0, int((x - xmin) / (xmax - xmin) * grid_size)))
        y_index = min(grid_size - 1, max(0, int((y - ymin) / (ymax - ymin) * grid_size)))
        assert feasible[y_index, x_index], f"valid {object_name} observation mapped outside feasible mask"
        histogram[y_index, x_index] += 1
    return histogram


def _coverage_entropy(histogram: np.ndarray, feasible: np.ndarray) -> tuple[float, float]:
    values = histogram[feasible]
    coverage = float(np.count_nonzero(values) / len(values))
    probabilities = values / values.sum()
    positive = probabilities[probabilities > 0]
    entropy = float(-(positive * np.log(positive)).sum() / math.log(len(values)))
    return coverage, entropy


def _coverage_auc(
    layouts: list[dict],
    object_name: str,
    grid_size: int,
    feasible: np.ndarray,
) -> float:
    prefixes = sorted({min(len(layouts), count) for count in (32, 128, 512, len(layouts))})
    coverages = [
        _coverage_entropy(_histogram(layouts[:count], object_name, grid_size, feasible), feasible)[0]
        for count in prefixes
    ]
    if len(prefixes) == 1:
        return coverages[0]
    x_values = np.log(np.asarray(prefixes, dtype=np.float64))
    return float(np.trapezoid(coverages, x_values) / (x_values[-1] - x_values[0]))


def _js_divergence(first: np.ndarray, second: np.ndarray, feasible: np.ndarray) -> float:
    first_values = first[feasible] + 1e-12
    second_values = second[feasible] + 1e-12
    first_values /= first_values.sum()
    second_values /= second_values.sum()
    midpoint = (first_values + second_values) / 2
    return float(
        0.5 * np.sum(first_values * np.log(first_values / midpoint))
        + 0.5 * np.sum(second_values * np.log(second_values / midpoint))
    )


def _median_iqr(values: list[float]) -> tuple[float, float]:
    return float(np.median(values)), float(np.percentile(values, 75) - np.percentile(values, 25))


def _check_compatible(runs: dict[Method, dict]) -> None:
    reference = runs["random_rejection"]
    for method, run in runs.items():
        for field in (
            "workload",
            "table_bounds",
            "obstacles",
            "movable_sizes",
            "target_layouts",
            "repetitions",
            "seeds",
            "max_attempts_per_layout",
            "max_iterations",
            "timing_scope",
            "benchmark_revision",
        ):
            if run.get(field) != reference.get(field):
                raise ValueError(f"{method} has incompatible {field}")
        if not run.get("benchmark_revision"):
            raise ValueError(f"{method} is missing benchmark_revision provenance")


def _write_heatmap_plot(heatmaps: dict[str, list], output: Path) -> None:
    """Write a comparison plot for every method and movable object."""
    import matplotlib.pyplot as plt

    methods = ("random_rejection", "arena", "robolab")
    object_names = tuple(MOVABLE_SIZES)
    figure, axes = plt.subplots(len(methods), len(object_names), figsize=(10, 9), sharex=True, sharey=True)
    for row, method in enumerate(methods):
        for column, object_name in enumerate(object_names):
            axis = axes[row, column]
            image = axis.imshow(
                heatmaps[f"{method}:{object_name}"],
                origin="lower",
                extent=TABLE_BOUNDS,
                cmap="viridis",
            )
            axis.set_title(f"{method.replace('_', ' ')} · {object_name.removeprefix('movable-')}")
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    for axis in axes[-1]:
        axis.set_xlabel("X position (m)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Y position (m)")
    figure.suptitle("Per-object marginal position distributions around fixed obstacles")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def analyze(args: argparse.Namespace) -> int:
    """Analyze equal-count per-object marginal free-space coverage."""
    loaded = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    runs = {run["method"]: run for run in loaded}
    if set(runs) != {"arena", "robolab", "random_rejection"}:
        raise ValueError("analysis requires arena, robolab, and random_rejection inputs")
    _check_compatible(runs)
    count = min(len(sample["layouts"]) for run in runs.values() for sample in run["samples"])
    if count < 2:
        raise ValueError("analysis requires at least two valid layouts per repetition")

    summaries = []
    heatmaps = {}
    for object_name in MOVABLE_SIZES:
        feasible = _feasible_mask(object_name, args.grid_size)
        reference_samples = runs["random_rejection"]["samples"]
        for method, run in runs.items():
            metrics = []
            aggregate = np.zeros_like(feasible, dtype=np.float64)
            for repeat_index, (sample, reference_sample) in enumerate(
                zip(run["samples"], reference_samples, strict=True)
            ):
                generator = np.random.default_rng(run["seeds"][repeat_index] % (2**32))
                layouts = [sample["layouts"][index] for index in generator.permutation(len(sample["layouts"]))[:count]]
                reference_layouts = [
                    reference_sample["layouts"][index]
                    for index in generator.permutation(len(reference_sample["layouts"]))[:count]
                ]
                histogram = _histogram(layouts, object_name, args.grid_size, feasible)
                aggregate += histogram
                coverage, entropy = _coverage_entropy(histogram, feasible)
                auc = _coverage_auc(layouts, object_name, args.grid_size, feasible)
                split_count = count // 2
                first = _histogram(layouts[:split_count], object_name, args.grid_size, feasible)
                if method == "random_rejection":
                    second = _histogram(layouts[split_count : 2 * split_count], object_name, args.grid_size, feasible)
                else:
                    second = _histogram(reference_layouts[:split_count], object_name, args.grid_size, feasible)
                metrics.append((coverage, entropy, auc, _js_divergence(first, second, feasible)))
            coverage_stats, entropy_stats, auc_stats, js_stats = (
                _median_iqr(list(metric_values)) for metric_values in zip(*metrics, strict=True)
            )
            summaries.append(
                CoverageSummary(
                    method=method,
                    object_name=object_name,
                    sample_count_per_repetition=count,
                    repetitions=len(metrics),
                    feasible_cells=int(feasible.sum()),
                    coverage=coverage_stats[0],
                    coverage_iqr=coverage_stats[1],
                    normalized_entropy=entropy_stats[0],
                    normalized_entropy_iqr=entropy_stats[1],
                    coverage_curve_auc=auc_stats[0],
                    coverage_curve_auc_iqr=auc_stats[1],
                    js_divergence_to_rejection=js_stats[0],
                    js_divergence_to_rejection_iqr=js_stats[1],
                )
            )
            heatmaps[f"{method}:{object_name}"] = (aggregate / aggregate.sum()).tolist()
    payload = {
        "schema_version": 1,
        "metric_scope": "per-object marginal center-space feasible with respect to table and fixed obstacles",
        "grid_size": args.grid_size,
        "rarefied_layouts_per_repetition": count,
        "inputs": [str(path) for path in args.inputs],
        "summaries": [asdict(summary) for summary in summaries],
        "heatmaps": heatmaps,
    }
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.plot is not None:
        _write_heatmap_plot(heatmaps, args.plot)
    for summary in summaries:
        print(
            f"{summary.method:16} {summary.object_name:16} coverage={summary.coverage:.3f} "
            f"entropy={summary.normalized_entropy:.3f} js={summary.js_divergence_to_rejection:.4f}"
        )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate_parser = commands.add_parser("generate")
    generate_parser.add_argument("--method", choices=("arena", "robolab", "random_rejection"), required=True)
    generate_parser.add_argument("--robolab-root", type=Path)
    generate_parser.add_argument("--target-layouts", type=int, default=1000)
    generate_parser.add_argument("--repetitions", type=int, default=5)
    generate_parser.add_argument("--warmup", type=int, default=1)
    generate_parser.add_argument("--seed", type=int, default=0)
    generate_parser.add_argument("--max-attempts-per-layout", type=int, default=1000)
    generate_parser.add_argument("--max-iterations", type=int, default=600)
    generate_parser.add_argument("--output", type=Path, required=True)
    analyze_parser = commands.add_parser("analyze")
    analyze_parser.add_argument("inputs", nargs=3, type=Path)
    analyze_parser.add_argument("--grid-size", type=int, default=32)
    analyze_parser.add_argument("--output", type=Path, required=True)
    analyze_parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    if args.command == "generate":
        if args.method == "robolab" and args.robolab_root is None:
            parser.error("--robolab-root is required for RoboLab")
        if (
            args.target_layouts <= 0
            or args.repetitions <= 0
            or args.warmup < 0
            or args.max_attempts_per_layout <= 0
            or args.max_iterations <= 0
        ):
            parser.error("generation counts must be positive and warmup non-negative")
    elif args.grid_size <= 1:
        parser.error("--grid-size must be greater than one")
    return args


def main() -> int:
    args = _parse_args()
    return generate(args) if args.command == "generate" else analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())

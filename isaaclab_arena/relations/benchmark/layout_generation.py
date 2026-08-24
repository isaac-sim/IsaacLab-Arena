# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Controlled layout-generation benchmark models, baselines, and analysis."""

from __future__ import annotations

import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal

from isaaclab_arena.relations.benchmark.provenance import collect_software_metadata

LayoutMethod = Literal["arena", "random_rejection", "explicit", "robolab"]
TableBounds = tuple[float, float, float, float]
ObjectSizes = dict[str, tuple[float, float]]

DEFAULT_TABLE_XY_BOUNDS: TableBounds = (-0.5, 0.5, -0.5, 0.5)
DEFAULT_OBJECT_XY_SIZE = 0.12
DEFAULT_OBJECT_NAMES = tuple(f"object-{index}" for index in range(5))
UNBIASED_SAMPLING_CAVEAT = "p >= 0.05 does not prove unbiased sampling."


def _check_finite(value: object, path: str = "value") -> None:
    """Reject non-finite floats recursively."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} must be finite")
    if isinstance(value, dict):
        for key, child in value.items():
            _check_finite(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _check_finite(child, f"{path}[{index}]")


def _strict_fields(cls: type, data: dict, label: str) -> dict:
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be an object")
    names = {field.name for field in fields(cls)}
    missing = names - data.keys()
    unknown = data.keys() - names
    if missing:
        raise ValueError(f"{label} is missing required fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"{label} contains unknown fields: {', '.join(sorted(unknown))}")
    _check_finite(data, label)
    return dict(data)


def _validate_geometry(table_xy_bounds: TableBounds, object_xy_sizes: ObjectSizes) -> None:
    _check_finite(table_xy_bounds, "table_xy_bounds")
    _check_finite(object_xy_sizes, "object_xy_sizes")
    if len(table_xy_bounds) != 4:
        raise ValueError("table_xy_bounds must contain xmin, xmax, ymin, ymax")
    xmin, xmax, ymin, ymax = table_xy_bounds
    if not xmin < xmax or not ymin < ymax:
        raise ValueError("table_xy_bounds must have positive extents")
    if not object_xy_sizes:
        raise ValueError("object_xy_sizes must not be empty")
    for name, size in object_xy_sizes.items():
        if not name or len(size) != 2 or size[0] <= 0.0 or size[1] <= 0.0:
            raise ValueError("object names must be non-empty and XY sizes must be positive")
        if size[0] > xmax - xmin or size[1] > ymax - ymin:
            raise ValueError(f"object {name!r} does not fit on the table")


@dataclass(frozen=True)
class LayoutPose:
    """One controlled object's center pose."""

    object_name: str
    x: float
    y: float
    z: float = 0.0
    yaw: float = 0.0

    def __post_init__(self) -> None:
        if not self.object_name:
            raise ValueError("object_name must not be empty")
        _check_finite(asdict(self), "layout pose")
        if self.yaw != 0.0:
            raise ValueError("controlled layouts require yaw=0")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> LayoutPose:
        return cls(**_strict_fields(cls, data, "layout pose"))


@dataclass(frozen=True)
class LayoutSample:
    """One bounded layout-generation attempt."""

    method: LayoutMethod
    seed: int
    success: bool
    attempts: int
    elapsed_ms: float | None
    poses: tuple[LayoutPose, ...]
    iterations: int | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.method not in ("arena", "random_rejection", "explicit", "robolab"):
            raise ValueError(f"unknown layout method: {self.method}")
        if self.attempts <= 0:
            raise ValueError("attempts must be positive")
        if self.elapsed_ms is not None and (not math.isfinite(self.elapsed_ms) or self.elapsed_ms < 0.0):
            raise ValueError("elapsed_ms must be finite and non-negative")
        if self.iterations is not None and self.iterations < 0:
            raise ValueError("iterations must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> LayoutSample:
        values = _strict_fields(cls, data, "layout sample")
        if not isinstance(values["poses"], (list, tuple)):
            raise ValueError("layout sample poses must be an array")
        values["poses"] = tuple(LayoutPose.from_dict(pose) for pose in values["poses"])
        return cls(**values)


@dataclass(frozen=True)
class LayoutGenerationRun:
    """Neutral envelope for controlled layout samples."""

    workload: str
    source_commit: str | None
    warmup: int
    master_seed: int
    max_iterations: int | None
    table_xy_bounds: TableBounds
    object_xy_sizes: ObjectSizes
    samples: tuple[LayoutSample, ...]

    def __post_init__(self) -> None:
        if not self.workload:
            raise ValueError("workload must not be empty")
        if self.warmup < 0:
            raise ValueError("warmup must be non-negative")
        if self.max_iterations is not None and self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        _validate_geometry(self.table_xy_bounds, self.object_xy_sizes)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> LayoutGenerationRun:
        values = _strict_fields(cls, data, "layout generation run")
        values["table_xy_bounds"] = tuple(values["table_xy_bounds"])
        if not isinstance(values["object_xy_sizes"], dict):
            raise ValueError("object_xy_sizes must be an object")
        values["object_xy_sizes"] = {name: tuple(size) for name, size in values["object_xy_sizes"].items()}
        values["samples"] = tuple(LayoutSample.from_dict(sample) for sample in values["samples"])
        return cls(**values)


@dataclass(frozen=True)
class LayoutGenerationSummary:
    """Statistical summary of one method's samples."""

    method: LayoutMethod
    sample_count: int
    successful_count: int
    success_rate: float
    median_successful_attempt_latency_ms: float | None
    p95_successful_attempt_latency_ms: float | None
    same_seed_deterministic: bool | None
    unique_layouts: int
    ks_statistic: float | None
    ks_p_value: float | None
    grid_p_value: float | None
    uniformity_interpretation: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_controlled_layout(
    poses: tuple[LayoutPose, ...] | list[LayoutPose],
    table_xy_bounds: TableBounds,
    object_xy_sizes: ObjectSizes,
) -> bool:
    """Validate object completeness, table containment, and AABB non-overlap."""
    _validate_geometry(table_xy_bounds, object_xy_sizes)
    by_name = {pose.object_name: pose for pose in poses}
    if len(by_name) != len(poses) or set(by_name) != set(object_xy_sizes):
        return False
    xmin, xmax, ymin, ymax = table_xy_bounds
    for name, pose in by_name.items():
        width, depth = object_xy_sizes[name]
        if pose.yaw != 0.0:
            return False
        if pose.x - width / 2 < xmin or pose.x + width / 2 > xmax:
            return False
        if pose.y - depth / 2 < ymin or pose.y + depth / 2 > ymax:
            return False
    names = tuple(object_xy_sizes)
    for left_index, left_name in enumerate(names):
        left = by_name[left_name]
        left_width, left_depth = object_xy_sizes[left_name]
        for right_name in names[left_index + 1 :]:
            right = by_name[right_name]
            right_width, right_depth = object_xy_sizes[right_name]
            separated = (
                abs(left.x - right.x) >= (left_width + right_width) / 2
                or abs(left.y - right.y) >= (left_depth + right_depth) / 2
            )
            if not separated:
                return False
    return True


def canonical_layout_key(poses: tuple[LayoutPose, ...] | list[LayoutPose]) -> tuple[tuple[str, float, float], ...]:
    """Return the six-decimal canonical controlled-layout identity."""
    return tuple(sorted((pose.object_name, round(pose.x, 6), round(pose.y, 6)) for pose in poses))


def make_object_sizes(
    num_objects: int = 5,
    object_size_m: float = DEFAULT_OBJECT_XY_SIZE,
) -> ObjectSizes:
    """Build square controlled-object geometry."""
    if num_objects <= 0 or not math.isfinite(object_size_m) or object_size_m <= 0.0:
        raise ValueError("num_objects and object_size_m must be positive")
    return {f"object-{index}": (object_size_m, object_size_m) for index in range(num_objects)}


def sample_random_rejection(
    *,
    seed: int,
    table_xy_bounds: TableBounds = DEFAULT_TABLE_XY_BOUNDS,
    object_xy_sizes: ObjectSizes | None = None,
    max_attempts: int = 100,
    clock=time.perf_counter,
) -> LayoutSample:
    """Uniformly sample full layouts and reject invalid ones."""
    sizes = object_xy_sizes or make_object_sizes()
    _validate_geometry(table_xy_bounds, sizes)
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    generator = random.Random(seed)
    xmin, xmax, ymin, ymax = table_xy_bounds
    start = clock()
    last_poses: tuple[LayoutPose, ...] = ()
    for attempt in range(1, max_attempts + 1):
        last_poses = tuple(
            LayoutPose(
                name,
                generator.uniform(xmin + width / 2, xmax - width / 2),
                generator.uniform(ymin + depth / 2, ymax - depth / 2),
            )
            for name, (width, depth) in sizes.items()
        )
        if validate_controlled_layout(last_poses, table_xy_bounds, sizes):
            return LayoutSample("random_rejection", seed, True, attempt, (clock() - start) * 1e3, last_poses)
    return LayoutSample(
        "random_rejection",
        seed,
        False,
        max_attempts,
        (clock() - start) * 1e3,
        last_poses,
        error="maximum layout attempts exhausted",
    )


def explicit_layout(
    *,
    seed: int = 0,
    table_xy_bounds: TableBounds = DEFAULT_TABLE_XY_BOUNDS,
    object_xy_sizes: ObjectSizes | None = None,
) -> LayoutSample:
    """Return the one authored capacity-baseline layout."""
    sizes = object_xy_sizes or make_object_sizes()
    _validate_geometry(table_xy_bounds, sizes)
    xmin, xmax, ymin, ymax = table_xy_bounds
    columns = math.ceil(math.sqrt(len(sizes)))
    rows = math.ceil(len(sizes) / columns)
    poses = tuple(
        LayoutPose(
            name,
            xmin + (column + 0.5) * (xmax - xmin) / columns,
            ymin + (row + 0.5) * (ymax - ymin) / rows,
        )
        for index, name in enumerate(sizes)
        for row, column in [(index // columns, index % columns)]
    )
    success = validate_controlled_layout(poses, table_xy_bounds, sizes)
    return LayoutSample(
        "explicit",
        seed,
        success,
        1,
        None,
        poses,
        error=None if success else "authored layout is invalid for requested geometry",
    )


def build_arena_controlled_scene(table_xy_bounds: TableBounds, object_xy_sizes: ObjectSizes):
    """Build controlled Arena assets with exact zero-margin table support."""
    from isaaclab_arena.relations.benchmark.synthetic_benchmark import BenchmarkAsset
    from isaaclab_arena.relations.relations import IsAnchor, On
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose

    _validate_geometry(table_xy_bounds, object_xy_sizes)
    xmin, xmax, ymin, ymax = table_xy_bounds
    table = BenchmarkAsset(
        "table",
        AxisAlignedBoundingBox(min_point=(xmin, ymin, -0.1), max_point=(xmax, ymax, 0.0)),
    )
    table.add_relation(IsAnchor())
    table.set_initial_pose(Pose.identity())
    objects = []
    for name, (width, depth) in object_xy_sizes.items():
        obj = BenchmarkAsset(
            name,
            AxisAlignedBoundingBox(
                min_point=(-width / 2, -depth / 2, -0.05),
                max_point=(width / 2, depth / 2, 0.05),
            ),
        )
        obj.add_relation(On(table, clearance_m=0.0, edge_margin_m=0.0))
        objects.append(obj)
    return [table, *objects]


def sample_arena_batch(
    *,
    seed: int,
    num_layouts: int,
    table_xy_bounds: TableBounds = DEFAULT_TABLE_XY_BOUNDS,
    object_xy_sizes: ObjectSizes | None = None,
    max_attempts_per_layout: int = 10,
    max_iterations: int = 600,
) -> tuple[LayoutSample, ...]:
    """Generate one native Arena batch and gate successes with the shared validator."""
    from isaaclab_arena.relations.benchmark.timing import time_call
    from isaaclab_arena.relations.object_placer import ObjectPlacer
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams

    if num_layouts <= 0 or max_attempts_per_layout <= 0 or max_iterations <= 0:
        raise ValueError("num_layouts, max_attempts_per_layout, and max_iterations must be positive")
    sizes = object_xy_sizes or make_object_sizes()
    assets = build_arena_controlled_scene(table_xy_bounds, sizes)
    placer = ObjectPlacer(
        ObjectPlacerParams(
            solver_params=RelationSolverParams(max_iters=max_iterations, verbose=False),
            placement_seed=seed,
            max_placement_attempts=max_attempts_per_layout,
            random_yaw_init=False,
            apply_positions_to_objects=False,
            verbose=False,
        )
    )
    elapsed_ms, results = time_call(lambda: placer.place(assets, num_envs=num_layouts), time.perf_counter)
    samples = []
    for index, result in enumerate(results):
        poses = tuple(
            LayoutPose(name, *result.positions[obj][:2], result.positions[obj][2], result.orientations.get(obj, 0.0))
            for obj in assets[1:]
            for name in [obj.name]
        )
        success = result.success and validate_controlled_layout(poses, table_xy_bounds, sizes)
        samples.append(
            LayoutSample(
                "arena",
                seed + index,
                success,
                result.attempts,
                elapsed_ms / num_layouts,
                poses,
                iterations=len(placer.last_loss_history),
                error=None if success else "Arena result failed the controlled validator",
            )
        )
    return tuple(samples)


def run_layout_generation(
    method: LayoutMethod,
    *,
    sample_count: int,
    master_seed: int = 0,
    warmup: int = 0,
    table_xy_bounds: TableBounds = DEFAULT_TABLE_XY_BOUNDS,
    object_xy_sizes: ObjectSizes | None = None,
    max_attempts: int = 100,
    max_iterations: int | None = None,
    workload: str = "controlled-tabletop",
) -> LayoutGenerationRun:
    """Run one controlled method and retain every bounded failure."""
    if sample_count <= 0 or warmup < 0:
        raise ValueError("sample_count must be positive and warmup must be non-negative")
    sizes = object_xy_sizes or make_object_sizes()
    if method == "arena":
        iterations = max_iterations or 600
        for index in range(warmup):
            sample_arena_batch(
                seed=master_seed - warmup + index,
                num_layouts=1,
                table_xy_bounds=table_xy_bounds,
                object_xy_sizes=sizes,
                max_attempts_per_layout=max_attempts,
                max_iterations=iterations,
            )
        samples = sample_arena_batch(
            seed=master_seed,
            num_layouts=sample_count,
            table_xy_bounds=table_xy_bounds,
            object_xy_sizes=sizes,
            max_attempts_per_layout=max_attempts,
            max_iterations=iterations,
        )
    elif method == "random_rejection":
        samples = tuple(
            sample_random_rejection(
                seed=master_seed + index,
                table_xy_bounds=table_xy_bounds,
                object_xy_sizes=sizes,
                max_attempts=max_attempts,
            )
            for index in range(sample_count)
        )
    elif method == "explicit":
        samples = (explicit_layout(seed=master_seed, table_xy_bounds=table_xy_bounds, object_xy_sizes=sizes),)
    else:
        raise ValueError("RoboLab runs are produced by the standalone RoboLab script")
    return LayoutGenerationRun(
        workload,
        collect_software_metadata().git_commit,
        warmup,
        master_seed,
        max_iterations,
        table_xy_bounds,
        sizes,
        samples,
    )


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _ks_uniform(values: list[float], lower: float, upper: float) -> tuple[float, float] | tuple[None, None]:
    if not values or upper <= lower:
        return None, None
    normalized = sorted((value - lower) / (upper - lower) for value in values)
    count = len(normalized)
    statistic = max(max((index + 1) / count - value, value - index / count) for index, value in enumerate(normalized))
    p_value = min(1.0, 2.0 * math.exp(-2.0 * count * statistic * statistic))
    return statistic, p_value


def _grid_uniformity(values: list[tuple[float, float]], bounds: TableBounds, grid_size: int = 4) -> float | None:
    if not values:
        return None
    xmin, xmax, ymin, ymax = bounds
    counts = [0] * (grid_size * grid_size)
    for x, y in values:
        column = min(grid_size - 1, int((x - xmin) / (xmax - xmin) * grid_size))
        row = min(grid_size - 1, int((y - ymin) / (ymax - ymin) * grid_size))
        counts[row * grid_size + column] += 1
    expected = len(values) / len(counts)
    statistic = sum((observed - expected) ** 2 / expected for observed in counts)
    degrees = len(counts) - 1
    z_score = ((statistic / degrees) ** (1 / 3) - (1 - 2 / (9 * degrees))) / math.sqrt(2 / (9 * degrees))
    return 0.5 * math.erfc(z_score / math.sqrt(2))


def summarize_layout_run(run: LayoutGenerationRun) -> LayoutGenerationSummary:
    """Summarize success, latency, determinism, diversity, and one-object uniformity."""
    method = run.samples[0].method if run.samples else "arena"
    successes = [sample for sample in run.samples if sample.success]
    latencies = [sample.elapsed_ms / sample.attempts for sample in successes if sample.elapsed_ms is not None]
    seed_keys: dict[int, set[tuple[tuple[str, float, float], ...]]] = {}
    for sample in successes:
        seed_keys.setdefault(sample.seed, set()).add(canonical_layout_key(sample.poses))
    repeated = [keys for seed, keys in seed_keys.items() if sum(sample.seed == seed for sample in run.samples) > 1]
    deterministic = all(len(keys) == 1 for keys in repeated) if repeated else None
    first_name = next(iter(run.object_xy_sizes))
    width, depth = run.object_xy_sizes[first_name]
    xmin, xmax, ymin, ymax = run.table_xy_bounds
    first_poses = [
        next(pose for pose in sample.poses if pose.object_name == first_name)
        for sample in successes
        if any(pose.object_name == first_name for pose in sample.poses)
    ]
    ks_statistic, ks_p_value = _ks_uniform([pose.x for pose in first_poses], xmin + width / 2, xmax - width / 2)
    grid_p_value = _grid_uniformity(
        [(pose.x, pose.y) for pose in first_poses],
        (xmin + width / 2, xmax - width / 2, ymin + depth / 2, ymax - depth / 2),
    )
    return LayoutGenerationSummary(
        method,
        len(run.samples),
        len(successes),
        len(successes) / len(run.samples) if run.samples else 0.0,
        statistics.median(latencies) if latencies else None,
        _percentile(latencies, 0.95) if latencies else None,
        deterministic,
        len({canonical_layout_key(sample.poses) for sample in successes}),
        ks_statistic,
        ks_p_value,
        grid_p_value,
        UNBIASED_SAMPLING_CAVEAT,
    )


def compare_layout_runs(
    runs: tuple[LayoutGenerationRun, ...] | list[LayoutGenerationRun],
) -> tuple[LayoutGenerationSummary, ...]:
    """Compare only runs with identical controlled workload, geometry, and seed."""
    if not runs:
        raise ValueError("at least one run is required")
    baseline = runs[0]
    for run in runs[1:]:
        if (
            run.workload != baseline.workload
            or run.table_xy_bounds != baseline.table_xy_bounds
            or run.object_xy_sizes != baseline.object_xy_sizes
            or run.master_seed != baseline.master_seed
        ):
            raise ValueError("layout runs have mismatched workload, geometry, or master seed")
    return tuple(summarize_layout_run(run) for run in runs)


def format_layout_markdown(runs: tuple[LayoutGenerationRun, ...] | list[LayoutGenerationRun]) -> str:
    """Render a neutral controlled-layout comparison report."""

    def number(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.3f}"

    summaries = compare_layout_runs(runs)
    rows = [
        (
            "| Method | Success | Median successful-attempt latency (ms) | P95 (ms) | Deterministic | Unique | KS p |"
            " Grid p |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for summary in summaries:
        rows.append(
            f"| {summary.method} | {summary.successful_count}/{summary.sample_count} | "
            f"{number(summary.median_successful_attempt_latency_ms)} | "
            f"{number(summary.p95_successful_attempt_latency_ms)} | "
            f"{'N/A' if summary.same_seed_deterministic is None else summary.same_seed_deterministic} | "
            f"{summary.unique_layouts} | {number(summary.ks_p_value)} | {number(summary.grid_p_value)} |"
        )
    return "\n".join([
        "# Controlled Tabletop Layout Generation",
        "",
        "All methods use the same table bounds, square yaw-zero objects, and shared XY validator.",
        "",
        *rows,
        "",
        f"Uniformity note: {UNBIASED_SAMPLING_CAVEAT}",
    ])


def write_layout_run(path: str | Path, run: LayoutGenerationRun) -> None:
    """Write strict finite neutral JSON."""
    Path(path).write_text(json.dumps(run.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_layout_run(path: str | Path) -> LayoutGenerationRun:
    """Load a strict neutral JSON run."""
    return LayoutGenerationRun.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

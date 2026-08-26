# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Exact-K unique-layout throughput models, runners, and reports."""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from isaaclab_arena.relations.benchmark.layout_generation import (
    DEFAULT_TABLE_XY_BOUNDS,
    LayoutMethod,
    LayoutPose,
    ObjectSizes,
    TableBounds,
    _check_finite,
    _strict_fields,
    _validate_geometry,
    canonical_layout_key,
    explicit_layout,
    make_object_sizes,
    sample_arena_batch,
    sample_random_rejection,
)
from isaaclab_arena.relations.benchmark.provenance import collect_source_revision
from isaaclab_arena.relations.benchmark.timing import get_peak_memory, reset_peak_memory

ThroughputMethod = Literal["arena", "random_rejection", "explicit", "robolab", "arena_scene"]


@dataclass(frozen=True)
class LayoutThroughputSample:
    """One repetition seeking exactly K unique valid layouts."""

    method: ThroughputMethod
    seed: int
    target_layouts: int
    elapsed_ms: float | None
    layouts_per_second: float | None
    validity_rate: float
    unique_layouts: int
    attempted_layouts: int
    accepted_layouts: int
    target_reached: bool
    timing_applicable: bool
    deterministic: bool
    iterations: int | None
    gpu_peak_allocated_bytes: int | None
    gpu_peak_reserved_bytes: int | None
    layouts: tuple[tuple[LayoutPose, ...], ...]
    error: str | None = None

    def __post_init__(self) -> None:
        if self.method not in ("arena", "random_rejection", "explicit", "robolab", "arena_scene"):
            raise ValueError(f"unknown throughput method: {self.method}")
        if self.target_layouts <= 0 or self.attempted_layouts <= 0:
            raise ValueError("target_layouts and attempted_layouts must be positive")
        if not 0.0 <= self.validity_rate <= 1.0:
            raise ValueError("validity_rate must be in [0, 1]")
        if not 0 <= self.unique_layouts <= self.accepted_layouts <= self.attempted_layouts:
            raise ValueError("throughput counts are inconsistent")
        if self.target_reached != (self.unique_layouts == self.target_layouts):
            raise ValueError("target_reached must mean exactly K unique layouts")
        if self.iterations is not None and self.iterations < 0:
            raise ValueError("iterations must be non-negative")
        if not self.timing_applicable and (self.elapsed_ms is not None or self.layouts_per_second is not None):
            raise ValueError("non-timed methods must use N/A timing")
        if self.timing_applicable:
            if self.elapsed_ms is None or not math.isfinite(self.elapsed_ms) or self.elapsed_ms < 0.0:
                raise ValueError("timed samples require finite non-negative elapsed_ms")
            expected_rate = self.unique_layouts * 1e3 / self.elapsed_ms if self.elapsed_ms > 0.0 else None
            if self.layouts_per_second != expected_rate:
                raise ValueError("layouts_per_second must be derived from elapsed_ms")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> LayoutThroughputSample:
        values = _strict_fields(cls, data, "layout throughput sample")
        values["layouts"] = tuple(tuple(LayoutPose.from_dict(pose) for pose in layout) for layout in values["layouts"])
        return cls(**values)


@dataclass(frozen=True)
class LayoutThroughputRun:
    """Neutral exact-K throughput run with complete compatibility metadata."""

    workload: str
    method: ThroughputMethod
    source_commit: str | None
    master_seed: int
    seeds: tuple[int, ...]
    repetitions: int
    target_layout_counts: tuple[int, ...]
    table_xy_bounds: TableBounds
    object_xy_sizes: ObjectSizes
    max_attempts_per_layout: int
    max_iterations: int | None
    warmup: int
    samples: tuple[LayoutThroughputSample, ...]
    solver_config: dict[str, object]

    def __post_init__(self) -> None:
        if not self.workload:
            raise ValueError("workload must not be empty")
        if self.repetitions <= 0 or len(self.seeds) != self.repetitions:
            raise ValueError("seeds must contain one seed per repetition")
        if not self.target_layout_counts or any(value <= 0 for value in self.target_layout_counts):
            raise ValueError("target_layout_counts must be positive")
        if len(set(self.target_layout_counts)) != len(self.target_layout_counts):
            raise ValueError("target_layout_counts must be unique")
        if self.max_attempts_per_layout <= 0 or self.warmup < 0:
            raise ValueError("attempt budget must be positive and warmup non-negative")
        if self.max_iterations is not None and self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        _validate_geometry(self.table_xy_bounds, self.object_xy_sizes)
        expected = {(seed, target) for seed in self.seeds for target in self.target_layout_counts}
        observed = {(sample.seed, sample.target_layouts) for sample in self.samples}
        if expected != observed or len(observed) != len(self.samples):
            raise ValueError("throughput samples must cover every seed/K pair exactly once")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> LayoutThroughputRun:
        if "solver_config" not in data:
            data = {**data, "solver_config": {}}
        values = _strict_fields(cls, data, "layout throughput run")
        values["seeds"] = tuple(values["seeds"])
        values["target_layout_counts"] = tuple(values["target_layout_counts"])
        values["table_xy_bounds"] = tuple(values["table_xy_bounds"])
        values["object_xy_sizes"] = {name: tuple(size) for name, size in values["object_xy_sizes"].items()}
        values["samples"] = tuple(LayoutThroughputSample.from_dict(sample) for sample in values["samples"])
        return cls(**values)


@dataclass(frozen=True)
class LayoutThroughputSummary:
    """Aggregate exact-K result for one method and target."""

    method: ThroughputMethod
    target_layouts: int
    repetitions: int
    targets_reached: int
    median_elapsed_ms: float | None
    median_layouts_per_second: float | None
    median_validity_rate: float
    median_unique_layouts: float
    attempted_layouts: int
    accepted_layouts: int


def _throughput_sample(
    method: ThroughputMethod,
    seed: int,
    target_layouts: int,
    elapsed_ms: float | None,
    attempted: int,
    accepted: int,
    layouts: list[tuple[LayoutPose, ...]],
    timing_applicable: bool,
    iterations: int | None = None,
    peak_allocated: int | None = None,
    peak_reserved: int | None = None,
) -> LayoutThroughputSample:
    unique = len(layouts)
    reached = unique == target_layouts
    rate = unique * 1e3 / elapsed_ms if timing_applicable and elapsed_ms is not None and elapsed_ms > 0.0 else None
    return LayoutThroughputSample(
        method,
        seed,
        target_layouts,
        elapsed_ms,
        rate,
        accepted / attempted,
        unique,
        attempted,
        accepted,
        reached,
        timing_applicable,
        True,
        iterations,
        peak_allocated,
        peak_reserved,
        tuple(layouts),
        None if reached else "bounded attempt budget exhausted before reaching K unique layouts",
    )


def run_controlled_throughput_sample(
    method: LayoutMethod,
    *,
    seed: int,
    target_layouts: int,
    table_xy_bounds: TableBounds = DEFAULT_TABLE_XY_BOUNDS,
    object_xy_sizes: ObjectSizes | None = None,
    max_attempts_per_layout: int = 100,
    max_iterations: int = 600,
    clock=time.perf_counter,
) -> LayoutThroughputSample:
    """Seek exactly K unique layouts while retaining bounded failures."""
    if target_layouts <= 0 or max_attempts_per_layout <= 0:
        raise ValueError("target_layouts and max_attempts_per_layout must be positive")
    sizes = object_xy_sizes or make_object_sizes()
    _validate_geometry(table_xy_bounds, sizes)
    if method == "explicit":
        authored = explicit_layout(seed=seed, table_xy_bounds=table_xy_bounds, object_xy_sizes=sizes)
        layouts = [authored.poses] if authored.success else []
        return _throughput_sample("explicit", seed, target_layouts, None, 1, int(authored.success), layouts, False)

    maximum_attempts = target_layouts * max_attempts_per_layout
    attempted = accepted = 0
    unique: dict[tuple[tuple[str, float, float], ...], tuple[LayoutPose, ...]] = {}
    reset_peak_memory()
    start = clock()
    iterations = None
    if method == "random_rejection":
        while attempted < maximum_attempts and len(unique) < target_layouts:
            sample = sample_random_rejection(
                seed=seed + attempted,
                table_xy_bounds=table_xy_bounds,
                object_xy_sizes=sizes,
                max_attempts=1,
            )
            attempted += 1
            if sample.success:
                accepted += 1
                unique.setdefault(canonical_layout_key(sample.poses), sample.poses)
    elif method == "arena":
        while attempted < maximum_attempts and len(unique) < target_layouts:
            batch_size = min(target_layouts - len(unique), maximum_attempts - attempted)
            samples = sample_arena_batch(
                seed=seed + attempted,
                num_layouts=batch_size,
                table_xy_bounds=table_xy_bounds,
                object_xy_sizes=sizes,
                max_attempts_per_layout=1,
                max_iterations=max_iterations,
            )
            attempted += len(samples)
            iterations = max((sample.iterations or 0 for sample in samples), default=0)
            for sample in samples:
                if sample.success:
                    accepted += 1
                    unique.setdefault(canonical_layout_key(sample.poses), sample.poses)
    else:
        raise ValueError("RoboLab throughput is produced by the standalone RoboLab script")
    elapsed_ms = (clock() - start) * 1e3
    peak_allocated, peak_reserved = get_peak_memory()
    return _throughput_sample(
        method,
        seed,
        target_layouts,
        elapsed_ms,
        attempted,
        accepted,
        list(unique.values()),
        True,
        iterations,
        peak_allocated,
        peak_reserved,
    )


def run_controlled_throughput(
    method: LayoutMethod,
    *,
    target_layout_counts: tuple[int, ...],
    repetitions: int = 3,
    master_seed: int = 0,
    warmup: int = 0,
    table_xy_bounds: TableBounds = DEFAULT_TABLE_XY_BOUNDS,
    object_xy_sizes: ObjectSizes | None = None,
    max_attempts_per_layout: int = 100,
    max_iterations: int | None = None,
    workload: str = "controlled-tabletop",
) -> LayoutThroughputRun:
    """Run the exact-K controlled throughput matrix."""
    if repetitions <= 0 or warmup < 0:
        raise ValueError("repetitions must be positive and warmup non-negative")
    sizes = object_xy_sizes or make_object_sizes()
    iterations = max_iterations or 600
    seed_stride = max(target_layout_counts) * max_attempts_per_layout
    seeds = tuple(master_seed + index * seed_stride for index in range(repetitions))
    for index in range(warmup):
        run_controlled_throughput_sample(
            method,
            seed=master_seed - (index + 1) * seed_stride,
            target_layouts=target_layout_counts[0],
            table_xy_bounds=table_xy_bounds,
            object_xy_sizes=sizes,
            max_attempts_per_layout=max_attempts_per_layout,
            max_iterations=iterations,
        )
    samples = tuple(
        run_controlled_throughput_sample(
            method,
            seed=seed,
            target_layouts=target,
            table_xy_bounds=table_xy_bounds,
            object_xy_sizes=sizes,
            max_attempts_per_layout=max_attempts_per_layout,
            max_iterations=iterations,
        )
        for seed in seeds
        for target in target_layout_counts
    )
    solver_config = {
        "name": "RelationSolver" if method == "arena" else method,
        "execution": "native-batch" if method == "arena" else "serial",
        "timing_scope": "complete-exact-k-generation" if method != "explicit" else "not-applicable",
        "clearance_m": 0.0 if method == "arena" else None,
        "collision_model": "aabb" if method == "arena" else None,
        "random_yaw": False,
    }
    return LayoutThroughputRun(
        workload,
        method,
        collect_source_revision(),
        master_seed,
        seeds,
        repetitions,
        target_layout_counts,
        table_xy_bounds,
        sizes,
        max_attempts_per_layout,
        iterations if method == "arena" else None,
        warmup,
        samples,
        solver_config,
    )


def check_throughput_compatibility(runs: tuple[LayoutThroughputRun, ...] | list[LayoutThroughputRun]) -> None:
    """Reject throughput runs whose workload-defining metadata differs."""
    if not runs:
        raise ValueError("at least one throughput run is required")
    if len(runs) > 1 and any(not run.solver_config for run in runs):
        raise ValueError("cross-method comparison requires solver_config metadata")
    baseline = runs[0]
    fields_to_match = (
        "workload",
        "table_xy_bounds",
        "object_xy_sizes",
        "max_attempts_per_layout",
        "seeds",
        "repetitions",
        "target_layout_counts",
    )
    for run in runs[1:]:
        mismatched = [name for name in fields_to_match if getattr(run, name) != getattr(baseline, name)]
        if mismatched:
            raise ValueError(f"incompatible throughput runs: mismatched {', '.join(mismatched)}")
    method_configs = {}
    for run in runs:
        config = (run.max_iterations, run.solver_config)
        if run.method in method_configs and method_configs[run.method] != config:
            raise ValueError(f"incompatible throughput runs: mismatched configuration for {run.method}")
        method_configs[run.method] = config


def summarize_throughput_run(run: LayoutThroughputRun) -> tuple[LayoutThroughputSummary, ...]:
    """Aggregate retained successes and failures by K."""
    summaries = []
    for target in run.target_layout_counts:
        samples = [sample for sample in run.samples if sample.target_layouts == target]
        elapsed = [sample.elapsed_ms for sample in samples if sample.elapsed_ms is not None]
        rates = [sample.layouts_per_second for sample in samples if sample.layouts_per_second is not None]
        summaries.append(
            LayoutThroughputSummary(
                run.method,
                target,
                len(samples),
                sum(sample.target_reached for sample in samples),
                statistics.median(elapsed) if elapsed else None,
                statistics.median(rates) if rates else None,
                statistics.median(sample.validity_rate for sample in samples),
                statistics.median(sample.unique_layouts for sample in samples),
                sum(sample.attempted_layouts for sample in samples),
                sum(sample.accepted_layouts for sample in samples),
            )
        )
    return tuple(summaries)


def format_throughput_markdown(runs: tuple[LayoutThroughputRun, ...] | list[LayoutThroughputRun]) -> str:
    """Render neutral exact-K throughput results without dropping failures."""

    def number(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.3f}"

    check_throughput_compatibility(runs)
    rows = [
        (
            "| Method | K | Targets reached | Median elapsed (ms) | Median layouts/s | Validity | Unique | Attempted |"
            " Accepted |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run in runs:
        for summary in summarize_throughput_run(run):
            rows.append(
                f"| {summary.method} | {summary.target_layouts} | "
                f"{summary.targets_reached}/{summary.repetitions} | {number(summary.median_elapsed_ms)} | "
                f"{number(summary.median_layouts_per_second)} | {summary.median_validity_rate:.3f} | "
                f"{summary.median_unique_layouts:g} | {summary.attempted_layouts} | {summary.accepted_layouts} |"
            )
    return "\n".join([
        "# Exact-K Unique Layout Throughput",
        "",
        "Failed bounded targets are retained. Explicit poses are a one-layout capacity baseline; timing is N/A.",
        (
            "Cross-method timing measures complete exact-K generation, not identical solver kernels. "
            "Each method retains its native collision model; exact settings are recorded in `solver_config`."
        ),
        "",
        *rows,
    ])


def write_throughput_run(path: str | Path, run: LayoutThroughputRun) -> None:
    """Write strict finite throughput JSON."""
    _check_finite(run.to_dict(), "layout throughput run")
    Path(path).write_text(json.dumps(run.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_throughput_run(path: str | Path) -> LayoutThroughputRun:
    """Load strict finite throughput JSON."""
    return LayoutThroughputRun.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

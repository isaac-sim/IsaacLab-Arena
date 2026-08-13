# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run construction, summaries, and report writers for relation benchmarks."""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from isaaclab_arena.relations.benchmark.models import (
    BenchmarkMeasurement,
    BenchmarkRun,
    BenchmarkScenario,
    BenchmarkTarget,
    CollisionModeName,
)
from isaaclab_arena.relations.benchmark.provenance import collect_software_metadata


@dataclass(frozen=True)
class _ScalingWorkload:
    """Parameters that must match before comparing one scaling axis."""

    target: BenchmarkTarget
    worker_id: str
    collision_mode: CollisionModeName
    num_objects: int | None
    num_envs: int | None
    graph_spec_path: str | None
    include_robot: bool | None
    max_iters: int
    convergence_threshold: float
    num_spheres: int
    placement_seed: int
    max_placement_attempts: int
    warmup_runs: int
    timed_runs: int
    final_loss_threshold: float
    min_valid_layout_rate: float

    @classmethod
    def from_measurement(
        cls,
        measurement: BenchmarkMeasurement,
        scaling_axis: Literal["batch", "objects"],
    ) -> _ScalingWorkload:
        """Build the workload identity for a measurement."""
        return cls(
            target=measurement.target,
            worker_id=measurement.worker_id,
            collision_mode=measurement.collision_mode,
            num_objects=None if scaling_axis == "objects" else measurement.num_objects,
            num_envs=None if scaling_axis == "batch" else measurement.num_envs,
            graph_spec_path=measurement.graph_spec_path,
            include_robot=measurement.include_robot,
            max_iters=measurement.max_iters,
            convergence_threshold=measurement.convergence_threshold,
            num_spheres=measurement.num_spheres,
            placement_seed=measurement.placement_seed,
            max_placement_attempts=measurement.max_placement_attempts,
            warmup_runs=measurement.warmup_runs,
            timed_runs=measurement.timed_runs,
            final_loss_threshold=measurement.final_loss_threshold,
            min_valid_layout_rate=measurement.min_valid_layout_rate,
        )


def _batch_ms(measurement: BenchmarkMeasurement) -> float | None:
    return {
        "solver": measurement.solve_ms,
        "placer": measurement.place_ms,
        "environment": measurement.bring_up_ms,
    }[measurement.target]


def requested_scenario_ids(
    scenarios: tuple[BenchmarkScenario, ...],
    targets: tuple[BenchmarkTarget, ...],
) -> tuple[str, ...]:
    """Return every expected result ID in execution order."""
    ids = tuple(scenario.scenario_id(target) for scenario in scenarios for target in targets)
    if len(ids) != len(set(ids)):
        raise ValueError("requested benchmark scenario IDs must be unique")
    return ids


def build_run(
    scenarios: tuple[BenchmarkScenario, ...],
    targets: tuple[BenchmarkTarget, ...],
    results: list[BenchmarkMeasurement],
    worker_assignments: dict[str, tuple[str, ...]] | None = None,
    worker_exit_codes: dict[str, int] | None = None,
    worker_errors: dict[str, str] | None = None,
) -> BenchmarkRun:
    """Build the canonical run envelope."""
    expected = requested_scenario_ids(scenarios, targets)
    exit_codes = worker_exit_codes or {"local": 0}
    if worker_assignments is None:
        worker_id = next(iter(exit_codes)) if len(exit_codes) == 1 else "local"
        worker_assignments = {worker_id: expected}
    return BenchmarkRun(
        requested_scenario_ids=expected,
        results=tuple(results),
        worker_assignments=worker_assignments,
        worker_exit_codes=exit_codes,
        software=collect_software_metadata(),
        worker_errors=worker_errors or {},
    )


def build_distributed_run(
    results: list[BenchmarkMeasurement],
    worker_assignments: dict[str, tuple[str, ...]],
    worker_exit_codes: dict[str, int],
    worker_errors: dict[str, str] | None = None,
) -> BenchmarkRun:
    """Build a run whose requested IDs are already worker-qualified."""
    expected = tuple(scenario_id for ids in worker_assignments.values() for scenario_id in ids)
    return BenchmarkRun(
        requested_scenario_ids=expected,
        results=tuple(results),
        worker_assignments=worker_assignments,
        worker_exit_codes=worker_exit_codes,
        software=collect_software_metadata(),
        worker_errors=worker_errors or {},
    )


def format_results_table(results: list[BenchmarkMeasurement]) -> str:
    """Render a compact text report."""
    header = (
        f"{'scenario':<28} {'worker':<10} {'target':<11} {'status':<7} {'mode':<5} {'objects':>7} "
        f"{'batch':>5} {'batch_ms':>10} {'step_ms':>9} {'iters':>7} {'layouts/s':>10} {'agg layouts/s':>13} "
        f"{'loss':>10} {'valid':>7}"
    )
    lines = [header, "-" * len(header)]
    for result in results:
        lines.append(
            f"{result.scenario_name:<28} {result.worker_id:<10} {result.target:<11} {result.status:<7} "
            f"{result.collision_mode:<5} {result.num_objects:>7} {result.num_envs:>5} "
            f"{_format_number(_batch_ms(result)):>10} {_format_number(result.solve_step_ms):>9} "
            f"{_format_iterations(result.iterations):>7} "
            f"{_format_number(result.throughput_envs_per_second):>10} "
            f"{_format_number(result.aggregate_throughput_envs_per_second):>13} "
            f"{_format_number(result.final_loss):>10} {_format_number(result.valid_layout_rate):>7}"
        )
        if result.error:
            lines.append(f"  error: {result.error}")
    return "\n".join(lines)


def format_scaling_summary(results: list[BenchmarkMeasurement]) -> str:
    """Summarize independent batch-size and object-count sweeps."""
    sections = [
        _format_batch_scaling(results),
        _format_object_scaling(results),
    ]
    return "\n\n".join(section for section in sections if section)


def _format_batch_scaling(results: list[BenchmarkMeasurement]) -> str:
    """Summarize batch-size scaling with object count held fixed."""
    groups: dict[_ScalingWorkload, list[BenchmarkMeasurement]] = {}
    for result in results:
        groups.setdefault(_ScalingWorkload.from_measurement(result, "batch"), []).append(result)

    lines = []
    for workload, measurements in groups.items():
        if len({measurement.num_envs for measurement in measurements}) <= 1:
            continue
        successful = [measurement for measurement in measurements if measurement.status == "ok"]
        ordered = sorted(measurements, key=lambda measurement: measurement.num_envs)
        baseline = ordered[0]
        baseline_throughput = baseline.throughput_envs_per_second
        highest = max((measurement.num_envs for measurement in successful), default=None)
        failures = sorted(
            (
                measurement.num_envs,
                measurement.error or "unknown failure",
            )
            for measurement in measurements
            if measurement.status == "failed"
        )
        assert workload.num_objects is not None
        workload_description = f"objects={workload.num_objects}"
        if workload.graph_spec_path is not None:
            robot = "yes" if workload.include_robot else "no"
            workload_description += f", graph={workload.graph_spec_path}, robot={robot}"
        highest_text = "-" if highest is None else str(highest)
        points = []
        for measurement in ordered:
            throughput = measurement.throughput_envs_per_second
            scale = (
                throughput / baseline_throughput if throughput is not None and baseline_throughput is not None else None
            )
            throughput_text = "-" if throughput is None else f"{throughput:.3f} layouts/s"
            scale_text = "" if scale is None else f", {scale:.2f}x"
            points.append(f"{measurement.num_envs}={throughput_text}{scale_text}, {measurement.status}")
        failure_text = "; ".join(f"{num_envs}: {reason}" for num_envs, reason in failures) or "-"
        lines.append(
            f"{workload.target}/{workload.worker_id} [{workload.collision_mode}, {workload_description}]: "
            f"throughput_vs_batch_{baseline.num_envs}: "
            + "; ".join(points)
            + f"; highest successful batch={highest_text}; failures={failure_text}"
        )
    return "\n".join(["Batch-size scaling (fixed object count)", *lines]) if lines else ""


def _format_object_scaling(results: list[BenchmarkMeasurement]) -> str:
    """Summarize object-count scaling with batch size held fixed."""
    groups: dict[_ScalingWorkload, list[BenchmarkMeasurement]] = {}
    for result in results:
        groups.setdefault(_ScalingWorkload.from_measurement(result, "objects"), []).append(result)

    lines = []
    for workload, measurements in groups.items():
        if len({measurement.num_objects for measurement in measurements}) <= 1:
            continue
        ordered = sorted(measurements, key=lambda measurement: measurement.num_objects)
        baseline_ms = _batch_ms(ordered[0])
        points = []
        for measurement in ordered:
            batch_ms = _batch_ms(measurement)
            scale = batch_ms / baseline_ms if batch_ms is not None and baseline_ms is not None else None
            timing = "-" if batch_ms is None else f"{batch_ms:.3f} ms"
            scale_text = "" if scale is None else f", {scale:.2f}x"
            points.append(f"{measurement.num_objects}={timing}{scale_text}, {measurement.status}")
        assert workload.num_envs is not None
        lines.append(
            f"{workload.target}/{workload.worker_id} [{workload.collision_mode}, batch={workload.num_envs}]: "
            f"latency_vs_objects_{ordered[0].num_objects}: "
            + "; ".join(points)
        )
    return "\n".join(["Object-count scaling (fixed batch size)", *lines]) if lines else ""


def _format_number(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _format_iterations(iterations: tuple[int, ...] | None) -> str:
    return "-" if not iterations else f"{statistics.median(iterations):.0f}"


def write_results_json(path: str | Path, run: BenchmarkRun) -> None:
    """Write the canonical benchmark envelope."""
    Path(path).write_text(json.dumps(run.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_results_csv(path: str | Path, run: BenchmarkRun) -> None:
    """Write the run's result rows as CSV."""
    rows = [result.to_dict() for result in run.results]
    fieldnames = list(rows[0]) if rows else []
    with Path(path).open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row["device"] = json.dumps(row["device"], sort_keys=True)
            writer.writerow(row)

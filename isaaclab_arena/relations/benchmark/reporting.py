# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run construction, capacity search, and report writers for relation benchmarks."""

from __future__ import annotations

import csv
import json
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from isaaclab_arena.relations.benchmark.metadata import collect_software_metadata
from isaaclab_arena.relations.benchmark.models import (
    BenchmarkMeasurement,
    BenchmarkRun,
    BenchmarkScenario,
    BenchmarkTarget,
    CollisionModeName,
)


@dataclass(frozen=True)
class _ScalingWorkload:
    """Parameters that must match before comparing batch sizes."""

    target: BenchmarkTarget
    worker_id: str
    collision_mode: CollisionModeName
    num_objects: int
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
    def from_measurement(cls, measurement: BenchmarkMeasurement) -> _ScalingWorkload:
        """Build the workload identity for a measurement."""
        return cls(
            target=measurement.target,
            worker_id=measurement.worker_id,
            collision_mode=measurement.collision_mode,
            num_objects=measurement.num_objects,
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


def _throughput_sort_key(measurement: BenchmarkMeasurement) -> tuple[float, int]:
    assert measurement.throughput_envs_per_second is not None
    return measurement.throughput_envs_per_second, measurement.num_envs


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


def search_capacity(
    probe: Callable[[int], bool],
    *,
    start_num_envs: int = 1,
    max_num_envs: int = 4096,
) -> int | None:
    """Find the largest viable batch by exponential growth and binary search."""
    assert 0 < start_num_envs <= max_num_envs
    if not probe(start_num_envs):
        return None
    if start_num_envs == max_num_envs:
        return start_num_envs
    low = start_num_envs
    high = min(start_num_envs * 2, max_num_envs)
    while probe(high):
        low = high
        if high == max_num_envs:
            return high
        high = min(high * 2, max_num_envs)
    while low + 1 < high:
        middle = (low + high) // 2
        if probe(middle):
            low = middle
        else:
            high = middle
    return low


def format_results_table(results: list[BenchmarkMeasurement]) -> str:
    """Render a compact text report."""
    header = (
        f"{'scenario':<28} {'worker':<10} {'target':<11} {'status':<7} {'mode':<5} {'envs':>5} "
        f"{'median_ms':>10} {'step_ms':>9} {'iters':>7} {'env/s':>10} {'agg env/s':>10} "
        f"{'loss':>10} {'valid':>7}"
    )
    lines = [header, "-" * len(header)]
    for result in results:
        median_ms = {
            "solver": result.solve_ms,
            "placer": result.place_ms,
            "environment": result.build_ms,
        }[result.target]
        lines.append(
            f"{result.scenario_name:<28} {result.worker_id:<10} {result.target:<11} {result.status:<7} "
            f"{result.collision_mode:<5} {result.num_envs:>5} "
            f"{_format_number(median_ms):>10} {_format_number(result.solve_step_ms):>9} "
            f"{_format_iterations(result.iterations):>7} "
            f"{_format_number(result.throughput_envs_per_second):>10} "
            f"{_format_number(result.aggregate_throughput_envs_per_second):>10} "
            f"{_format_number(result.final_loss):>10} {_format_number(result.valid_layout_rate):>7}"
        )
        if result.error:
            lines.append(f"  error: {result.error}")
    return "\n".join(lines)


def format_scaling_summary(results: list[BenchmarkMeasurement]) -> str:
    """Summarize comparable workloads measured at multiple batch sizes."""
    groups: dict[_ScalingWorkload, list[BenchmarkMeasurement]] = {}
    for result in results:
        groups.setdefault(_ScalingWorkload.from_measurement(result), []).append(result)

    lines = []
    for workload, measurements in groups.items():
        if len({measurement.num_envs for measurement in measurements}) <= 1:
            continue
        successful = [measurement for measurement in measurements if measurement.status == "ok"]
        throughput_results = [
            measurement for measurement in successful if measurement.throughput_envs_per_second is not None
        ]
        highest = max((measurement.num_envs for measurement in successful), default=None)
        best = max(
            throughput_results,
            key=_throughput_sort_key,
            default=None,
        )
        failures = sorted(
            (
                measurement.num_envs,
                measurement.error or "unknown failure",
            )
            for measurement in measurements
            if measurement.status == "failed"
        )
        workload_description = f"objects={workload.num_objects}"
        if workload.graph_spec_path is not None:
            robot = "yes" if workload.include_robot else "no"
            workload_description += f", graph={workload.graph_spec_path}, robot={robot}"
        highest_text = "-" if highest is None else str(highest)
        best_text = "-" if best is None else f"{best.num_envs} ({best.throughput_envs_per_second:.3f} env/s)"
        failure_text = "; ".join(f"{num_envs}: {reason}" for num_envs, reason in failures) or "-"
        lines.append(
            f"{workload.target}/{workload.worker_id} [{workload.collision_mode}, {workload_description}]: "
            f"highest successful={highest_text}, "
            f"best throughput={best_text}, failures={failure_text}"
        )
    return "\n".join(lines)


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

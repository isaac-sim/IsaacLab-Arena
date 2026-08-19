# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Baseline-versus-candidate comparison for solver performance runs."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from isaaclab_arena.relations.benchmark.models import BenchmarkMeasurement, BenchmarkRun

_COMPATIBILITY_FIELDS = (
    "target",
    "collision_mode",
    "num_objects",
    "num_envs",
    "max_iters",
    "convergence_threshold",
    "num_spheres",
    "placement_seed",
    "warmup_runs",
    "timed_runs",
    "final_loss_threshold",
    "asset_set_name",
    "background_treatment",
)


@dataclass(frozen=True)
class RegressionComparisonRow:
    """Performance changes for one matched solver scenario."""

    scenario_id: str
    scenario_name: str
    collision_mode: str
    num_objects: int
    num_envs: int
    baseline_iterations_per_second: float
    candidate_iterations_per_second: float | None
    iteration_rate_change_percent: float | None
    baseline_solve_ms: float
    candidate_solve_ms: float | None
    solve_time_change_percent: float | None
    baseline_peak_reserved_bytes: int | None
    candidate_peak_reserved_bytes: int | None
    peak_reserved_change_percent: float | None
    candidate_status: str
    candidate_error: str | None
    regressed: bool


@dataclass(frozen=True)
class RegressionComparison:
    """Complete comparison between two compatible benchmark runs."""

    baseline_commit: str | None
    candidate_commit: str | None
    baseline_dirty: bool | None
    candidate_dirty: bool | None
    device_name: str | None
    maximum_regression_percent: float
    baseline_run_succeeded: bool
    candidate_run_succeeded: bool
    rows: tuple[RegressionComparisonRow, ...]

    @property
    def correctness_passed(self) -> bool:
        """Whether the candidate run and every candidate scenario completed successfully."""
        return self.candidate_run_succeeded and all(row.candidate_status == "ok" for row in self.rows)

    @property
    def passed(self) -> bool:
        """Whether every candidate scenario passed its performance and correctness gates."""
        return self.correctness_passed and all(not row.regressed for row in self.rows)

    def to_dict(self) -> dict[str, object]:
        """Serialize the comparison."""
        return {**asdict(self), "correctness_passed": self.correctness_passed, "passed": self.passed}


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def load_benchmark_run(path: str | Path) -> BenchmarkRun:
    """Load one canonical benchmark JSON report."""
    payload = json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=_reject_nonfinite_json,
    )
    assert isinstance(payload, dict), "benchmark report must contain a JSON object"
    return BenchmarkRun.from_dict(payload)


def compare_benchmark_runs(
    baseline: BenchmarkRun,
    candidate: BenchmarkRun,
    *,
    maximum_regression_percent: float = 10.0,
) -> RegressionComparison:
    """Compare compatible solver runs using optimization iteration rate."""
    assert (
        math.isfinite(maximum_regression_percent) and maximum_regression_percent >= 0.0
    ), "maximum regression percent must be finite and non-negative"
    if not baseline.requested_scenario_ids:
        raise ValueError("benchmark comparison requires at least one scenario")
    if not baseline.succeeded:
        raise ValueError("baseline benchmark run did not complete successfully")
    if baseline.requested_scenario_ids != candidate.requested_scenario_ids:
        raise ValueError("baseline and candidate requested different scenario IDs")
    baseline_results = {result.scenario_id: result for result in baseline.results}
    candidate_results = {result.scenario_id: result for result in candidate.results}
    if baseline_results.keys() != candidate_results.keys():
        raise ValueError("baseline and candidate produced different scenario IDs")
    rows = tuple(
        _compare_measurements(
            baseline_results[scenario_id],
            candidate_results[scenario_id],
            maximum_regression_percent,
        )
        for scenario_id in baseline.requested_scenario_ids
    )
    baseline_device_values = [result.device.name for result in baseline.results]
    candidate_device_values = [result.device.name for result in candidate.results]
    device_values = baseline_device_values + candidate_device_values
    if any(name is not None and not isinstance(name, str) for name in device_values):
        raise ValueError("benchmark GPU names must be strings")
    baseline_device_names = set(baseline_device_values)
    candidate_device_names = set(candidate_device_values)
    if len(baseline_device_names) != 1 or len(candidate_device_names) != 1:
        raise ValueError("each benchmark run must use exactly one GPU model")
    baseline_device_name = next(iter(baseline_device_names))
    candidate_device_name = next(iter(candidate_device_names))
    if baseline_device_name != candidate_device_name:
        raise ValueError(
            f"benchmark runs used different GPU models: {baseline_device_name!r} and {candidate_device_name!r}"
        )
    return RegressionComparison(
        baseline_commit=baseline.software.git_commit,
        candidate_commit=candidate.software.git_commit,
        baseline_dirty=baseline.software.git_dirty,
        candidate_dirty=candidate.software.git_dirty,
        device_name=baseline_device_name,
        maximum_regression_percent=maximum_regression_percent,
        baseline_run_succeeded=baseline.succeeded,
        candidate_run_succeeded=candidate.succeeded,
        rows=rows,
    )


def _compare_measurements(
    baseline: BenchmarkMeasurement,
    candidate: BenchmarkMeasurement,
    maximum_regression_percent: float,
) -> RegressionComparisonRow:
    incompatible_fields = [
        field_name
        for field_name in _COMPATIBILITY_FIELDS
        if getattr(baseline, field_name) != getattr(candidate, field_name)
    ]
    if incompatible_fields:
        raise ValueError(f"scenario {baseline.scenario_id} differs in: {', '.join(incompatible_fields)}")
    if baseline.target != "solver":
        raise ValueError(f"regression scenario {baseline.scenario_id} is not a solver measurement")
    if baseline.status != "ok":
        raise ValueError(f"baseline scenario {baseline.scenario_id} did not pass: {baseline.error}")
    baseline_rate = baseline.solver_iterations_per_second
    baseline_solve_ms = baseline.solve_ms
    if not _is_positive_finite_number(baseline_rate) or not _is_positive_finite_number(baseline_solve_ms):
        raise ValueError(f"baseline scenario {baseline.scenario_id} has incomplete timing metrics")
    candidate_rate = candidate.solver_iterations_per_second
    candidate_solve_ms = candidate.solve_ms
    if candidate.status == "ok" and (
        not _is_positive_finite_number(candidate_rate) or not _is_positive_finite_number(candidate_solve_ms)
    ):
        raise ValueError(f"candidate scenario {candidate.scenario_id} has incomplete timing metrics")
    iteration_rate_change = _percentage_change(candidate_rate, baseline_rate)
    solve_time_change = _percentage_change(candidate_solve_ms, baseline_solve_ms)
    peak_reserved_change = _percentage_change(candidate.peak_reserved_bytes, baseline.peak_reserved_bytes)
    regressed = iteration_rate_change is None or iteration_rate_change < -maximum_regression_percent
    return RegressionComparisonRow(
        scenario_id=baseline.scenario_id,
        scenario_name=baseline.scenario_name,
        collision_mode=baseline.collision_mode,
        num_objects=baseline.num_objects,
        num_envs=baseline.num_envs,
        baseline_iterations_per_second=baseline_rate,
        candidate_iterations_per_second=candidate_rate,
        iteration_rate_change_percent=iteration_rate_change,
        baseline_solve_ms=baseline_solve_ms,
        candidate_solve_ms=candidate_solve_ms,
        solve_time_change_percent=solve_time_change,
        baseline_peak_reserved_bytes=baseline.peak_reserved_bytes,
        candidate_peak_reserved_bytes=candidate.peak_reserved_bytes,
        peak_reserved_change_percent=peak_reserved_change,
        candidate_status=candidate.status,
        candidate_error=candidate.error,
        regressed=regressed,
    )


def _percentage_change(candidate: int | float | None, baseline: int | float | None) -> float | None:
    if candidate is None or baseline is None or baseline == 0:
        return None
    if not _is_finite_number(candidate) or not _is_finite_number(baseline):
        raise ValueError("comparison metrics must be finite numbers")
    return 100.0 * (candidate - baseline) / baseline


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_positive_finite_number(value: object) -> bool:
    return _is_finite_number(value) and value > 0.0


def format_regression_markdown(comparison: RegressionComparison) -> str:
    """Render a concise benchmark comparison report."""
    lines = [
        "# Relation Solver Performance Regression",
        "",
        f"- Baseline commit: `{comparison.baseline_commit or 'unknown'}`"
        + (" (dirty)" if comparison.baseline_dirty else ""),
        f"- Candidate commit: `{comparison.candidate_commit or 'unknown'}`"
        + (" (dirty)" if comparison.candidate_dirty else ""),
        f"- GPU: {comparison.device_name or 'unknown'}",
        f"- Iteration-rate regression threshold: {comparison.maximum_regression_percent:g}%",
        f"- Result: {'PASS' if comparison.passed else 'REGRESSION'}",
        "",
        (
            "| Scenario | Mode | Objects | Batch | Baseline iter/s | Candidate iter/s | Iter/s change | "
            "Solve-time change | Peak reserved change | Result |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in comparison.rows:
        row_passed = row.candidate_status == "ok" and not row.regressed
        lines.append(
            f"| {row.scenario_name} | {row.collision_mode} | {row.num_objects} | {row.num_envs} | "
            f"{row.baseline_iterations_per_second:.3f} | {_format_number(row.candidate_iterations_per_second)} | "
            f"{_format_percent(row.iteration_rate_change_percent)} | "
            f"{_format_percent(row.solve_time_change_percent)} | "
            f"{_format_percent(row.peak_reserved_change_percent)} | "
            f"{'PASS' if row_passed else 'REGRESSION'} |"
        )
    failures = [row for row in comparison.rows if row.candidate_status != "ok"]
    if failures:
        lines.extend(["", "## Candidate Failures"])
        lines.extend(f"- `{row.scenario_id}`: {row.candidate_error or 'unknown error'}" for row in failures)
    return "\n".join(lines)


def write_regression_json(path: str | Path, comparison: RegressionComparison) -> None:
    """Write a machine-readable comparison report."""
    Path(path).write_text(json.dumps(comparison.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_regression_csv(path: str | Path, comparison: RegressionComparison) -> None:
    """Write one flat comparison row per scenario."""
    rows = [asdict(row) for row in comparison.rows]
    with Path(path).open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _format_number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _format_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.1f}%"

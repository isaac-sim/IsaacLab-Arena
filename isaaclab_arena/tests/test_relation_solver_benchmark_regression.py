# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for relation-solver baseline-versus-candidate comparisons."""

import json
from dataclasses import replace

import pytest

from isaaclab_arena.relations.benchmark import (
    BenchmarkMeasurement,
    BenchmarkRun,
    BenchmarkScenario,
    DeviceMetadata,
    SoftwareMetadata,
    compare_benchmark_runs,
    format_regression_markdown,
    load_benchmark_run,
)
from isaaclab_arena_examples.relations import compare_relation_solver_benchmarks as comparison_cli


def _measurement(
    *,
    iterations_per_second: float,
    solve_ms: float,
    status: str = "ok",
    device_name: str | None = "NVIDIA L40",
) -> BenchmarkMeasurement:
    scenario = BenchmarkScenario(
        name="regression-small",
        num_objects=6,
        num_envs=1,
        max_iters=200,
        timed_runs=5,
        final_loss_threshold=1e9,
        asset_set_name="regression-synthetic",
    )
    device = DeviceMetadata(
        physical_device="0",
        name=device_name,
        total_memory_bytes=48 * 2**30,
        free_memory_before_bytes=40 * 2**30,
        free_memory_after_bytes=39 * 2**30,
        minimum_free_memory_bytes=38 * 2**30,
        compute_capability="8.9",
    )
    return BenchmarkMeasurement.from_scenario(
        scenario,
        "solver",
        status=status,
        device=device,
        error=None if status == "ok" else "candidate failed",
        solve_ms_samples=(solve_ms,) * 5,
        solve_ms=solve_ms,
        solver_iterations_per_second_samples=(iterations_per_second,) * 5,
        solver_iterations_per_second=iterations_per_second,
        iterations=(200,) * 5,
        final_loss=0.1,
        peak_reserved_bytes=2 * 2**30,
    )


def _run(measurement: BenchmarkMeasurement, commit: str) -> BenchmarkRun:
    scenario_id = measurement.scenario_id
    return BenchmarkRun(
        requested_scenario_ids=(scenario_id,),
        results=(measurement,),
        worker_assignments={"local": (scenario_id,)},
        worker_exit_codes={"local": 0},
        software=SoftwareMetadata(
            git_commit=commit,
            git_dirty=False,
            python_version="3.11",
            pytorch_version="2.7",
            cuda_version="12.8",
        ),
    )


def test_benchmark_run_round_trips_for_regression_loading():
    run = _run(_measurement(iterations_per_second=100.0, solve_ms=2000.0), "baseline")

    assert BenchmarkRun.from_dict(run.to_dict()) == run


def test_comparison_flags_iteration_rate_regression():
    baseline = _run(_measurement(iterations_per_second=100.0, solve_ms=2000.0), "baseline")
    candidate = _run(_measurement(iterations_per_second=85.0, solve_ms=2350.0), "candidate")

    comparison = compare_benchmark_runs(baseline, candidate, maximum_regression_percent=10.0)

    assert not comparison.passed
    assert comparison.rows[0].regressed
    assert comparison.rows[0].iteration_rate_change_percent == pytest.approx(-15.0)
    assert "REGRESSION" in format_regression_markdown(comparison)


def test_comparison_rejects_different_workloads_and_gpu_models():
    baseline_measurement = _measurement(iterations_per_second=100.0, solve_ms=2000.0)
    baseline = _run(baseline_measurement, "baseline")
    candidate = _run(replace(baseline_measurement, max_iters=201), "candidate")
    with pytest.raises(ValueError, match="differs in: max_iters"):
        compare_benchmark_runs(baseline, candidate)

    candidate = _run(
        _measurement(iterations_per_second=100.0, solve_ms=2000.0, device_name="NVIDIA RTX 6000 Ada"),
        "candidate",
    )
    with pytest.raises(ValueError, match="different GPU models"):
        compare_benchmark_runs(baseline, candidate)

    candidate = _run(
        _measurement(iterations_per_second=100.0, solve_ms=2000.0, device_name=None),
        "candidate",
    )
    with pytest.raises(ValueError, match="different GPU models"):
        compare_benchmark_runs(baseline, candidate)


def test_candidate_failure_is_a_regression():
    baseline = _run(_measurement(iterations_per_second=100.0, solve_ms=2000.0), "baseline")
    candidate = _run(
        _measurement(iterations_per_second=100.0, solve_ms=2000.0, status="failed"),
        "candidate",
    )

    comparison = compare_benchmark_runs(baseline, candidate)

    assert not comparison.passed
    assert not comparison.correctness_passed
    assert comparison.rows[0].candidate_error == "candidate failed"


def test_comparison_validates_run_envelopes_and_finite_thresholds():
    baseline = _run(_measurement(iterations_per_second=100.0, solve_ms=2000.0), "baseline")
    failed_envelope = replace(baseline, worker_exit_codes={"local": 1})

    with pytest.raises(ValueError, match="baseline benchmark run did not complete"):
        compare_benchmark_runs(failed_envelope, baseline)
    comparison = compare_benchmark_runs(baseline, failed_envelope)
    assert not comparison.correctness_passed
    assert not comparison.passed
    with pytest.raises(AssertionError, match="finite and non-negative"):
        compare_benchmark_runs(baseline, baseline, maximum_regression_percent=float("inf"))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("software", None, "software must be an object"),
        ("worker_assignments", None, "worker_assignments must map"),
        ("results", [None], "results must be an array of objects"),
    ],
)
def test_benchmark_run_rejects_malformed_nested_data(field, value, message):
    payload = _run(_measurement(iterations_per_second=100.0, solve_ms=2000.0), "baseline").to_dict()
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        BenchmarkRun.from_dict(payload)


def test_benchmark_run_rejects_non_string_scenario_ids():
    payload = _run(_measurement(iterations_per_second=100.0, solve_ms=2000.0), "baseline").to_dict()
    payload["requested_scenario_ids"] = [[]]

    with pytest.raises(ValueError, match="array of strings"):
        BenchmarkRun.from_dict(payload)


def test_comparison_rejects_empty_runs_and_non_finite_metrics(tmp_path):
    empty_run = BenchmarkRun(
        requested_scenario_ids=(),
        results=(),
        worker_assignments={"local": ()},
        worker_exit_codes={"local": 0},
        software=SoftwareMetadata(
            git_commit="empty",
            git_dirty=False,
            python_version="3.11",
            pytorch_version="2.7",
            cuda_version="12.8",
        ),
    )
    with pytest.raises(ValueError, match="at least one scenario"):
        compare_benchmark_runs(empty_run, empty_run)

    baseline = _run(_measurement(iterations_per_second=100.0, solve_ms=2000.0), "baseline")
    candidate_measurement = replace(
        _measurement(iterations_per_second=100.0, solve_ms=2000.0),
        solver_iterations_per_second=float("nan"),
    )
    with pytest.raises(ValueError, match="incomplete timing metrics"):
        compare_benchmark_runs(baseline, _run(candidate_measurement, "candidate"))

    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text('{"metric": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON value"):
        load_benchmark_run(malformed_path)


def test_comparison_rejects_incomplete_or_malformed_candidate_metrics():
    baseline = _run(_measurement(iterations_per_second=100.0, solve_ms=2000.0), "baseline")
    candidate_measurement = _measurement(iterations_per_second=100.0, solve_ms=2000.0)
    with pytest.raises(ValueError, match="incomplete timing metrics"):
        compare_benchmark_runs(
            baseline,
            _run(replace(candidate_measurement, solver_iterations_per_second=None), "candidate"),
        )
    malformed_device = replace(candidate_measurement.device, name=[])
    with pytest.raises(ValueError, match="GPU names must be strings"):
        compare_benchmark_runs(
            baseline,
            _run(replace(candidate_measurement, device=malformed_device), "candidate"),
        )


def test_comparison_cli_writes_reports_and_gates_regressions(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    output_dir = tmp_path / "comparison"
    baseline = _run(_measurement(iterations_per_second=100.0, solve_ms=2000.0), "baseline")
    candidate = _run(_measurement(iterations_per_second=80.0, solve_ms=2500.0), "candidate")
    baseline_path.write_text(json.dumps(baseline.to_dict()), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate.to_dict()), encoding="utf-8")

    exit_code = comparison_cli.main([
        str(baseline_path),
        str(candidate_path),
        "--output-dir",
        str(output_dir),
    ])

    assert exit_code == 1
    assert {path.name for path in output_dir.iterdir()} == {
        "regression.csv",
        "regression.json",
        "regression.md",
    }


def test_report_only_does_not_hide_candidate_correctness_failure(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline = _run(_measurement(iterations_per_second=100.0, solve_ms=2000.0), "baseline")
    candidate = _run(
        _measurement(iterations_per_second=100.0, solve_ms=2000.0, status="failed"),
        "candidate",
    )
    baseline_path.write_text(json.dumps(baseline.to_dict()), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate.to_dict()), encoding="utf-8")

    assert comparison_cli.main([str(baseline_path), str(candidate_path), "--report-only"]) == 1

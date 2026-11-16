# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the placement success KPI exporter in ``isaaclab_arena_benchmarking/benchmarks/placement_success``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from isaaclab_arena.evaluation.arena_experiment_result import (
    ARENA_EXPERIMENT_RESULT_FILENAME,
    ARENA_SINGLE_RUN_PLACEMENT_DIAGNOSTICS_FILENAME,
)
from isaaclab_arena.relations.placement_diagnostics import (
    AggregatedPlacementDiagnostics,
    BatchValidationDiagnostics,
    PlacementDiagnostics,
)
from isaaclab_arena_benchmarking.benchmarks.placement_success.placement_success_exporter import (
    PlacementSuccessKpiExporter,
    RunPlacementCounts,
)
from isaaclab_arena_benchmarking.benchmarks.registry import export_kpis

END_TIMESTAMP_NS = 1792800000000000000
"""2026-10-23 UTC, in Unix epoch nanoseconds."""

EXPERIMENT_NAME = "placement_subset_experiment"


def test_run_rates_count_candidates_from_the_first_pool_batch_only():
    """The valid-candidate rate uses each build's first pool batch; later batches add only to the batch count."""
    first_batch = BatchValidationDiagnostics(num_candidates=100, num_valid=60, checks={})
    later_batch = BatchValidationDiagnostics(num_candidates=20, num_valid=5, checks={})
    placement_diagnostics = AggregatedPlacementDiagnostics([
        PlacementDiagnostics(10, [first_batch, later_batch], fallback_envs=[3]),
        PlacementDiagnostics(10, [first_batch]),
    ])

    run_counts = RunPlacementCounts("open_fridge", placement_diagnostics=placement_diagnostics)

    assert run_counts.valid_candidate_rate_first_batch == pytest.approx(0.6)
    assert run_counts.valid_env_rate_after_retries == pytest.approx(0.95)
    assert run_counts.batches_per_build == pytest.approx(1.5)


def test_exporter_writes_the_mean_over_scored_runs_with_equal_weights(tmp_path: Path):
    """Scored Runs are weighted equally whatever their sample size; a Run without diagnostics is not scored."""
    runs = ["pick_apple", "pick_banana", "no_relations"]
    diagnostics_per_run = {
        # 45 of 50 environments valid over two rebuilds, at 300 of 500 valid candidates.
        "pick_apple": [_diagnostics(25, 150, 250, 3, 2), _diagnostics(25, 150, 250, 2, 1)],
        # 25 of 25 environments valid over one rebuild, at 250 of 250 valid candidates.
        "pick_banana": [_diagnostics(25, 250, 250, 0, 1)],
    }
    _write_experiment(tmp_path, runs, diagnostics_per_run)

    exporter = PlacementSuccessKpiExporter()
    assert exporter.get_records_filename() == "kpis_placement_success.json"
    records_path = exporter.export_to_json(tmp_path)

    assert records_path == tmp_path / "kpis_placement_success.json"
    (record,) = json.loads(records_path.read_text(encoding="utf-8"))
    assert record == {
        "type": "placement_success",
        "timestamp_ns": END_TIMESTAMP_NS,
        "git_sha": "1a2b3c4",
        "git_branch": "main",
        "benchmark_id": EXPERIMENT_NAME,
        "valid_candidate_rate_first_batch": pytest.approx((0.6 + 1.0) / 2),
        "valid_env_rate_after_retries": pytest.approx((0.9 + 1.0) / 2),
        "batches_per_build": pytest.approx((1.5 + 1.0) / 2),
        "num_scored_runs": 2,
        "num_runs": 3,
    }


def test_export_kpis_prints_the_run_rates_after_a_complete_experiment(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """The per-Run placement rates are printed after the KPI file is written."""
    _write_experiment(tmp_path, ["pick_apple"], {"pick_apple": [_diagnostics(25, 150, 250, 3, 2)]})

    assert export_kpis(tmp_path, ["placement_success"]) == [tmp_path / "kpis_placement_success.json"]

    output = capsys.readouterr().out
    assert "Placement success per Run:" in output
    assert "  pick_apple: valid candidates 60.0%, valid environments 88.0%, 2.00 batches per build" in output


def test_exporter_writes_nothing_when_no_run_wrote_placement_diagnostics(tmp_path: Path):
    """An Experiment whose Runs solved no relations has no placement success record."""
    _write_experiment(tmp_path, ["no_relations"], {})

    assert PlacementSuccessKpiExporter().export_to_json(tmp_path) is None
    assert not (tmp_path / "kpis_placement_success.json").exists()


def test_export_kpis_writes_no_placement_file_when_a_run_failed_and_prints_the_run_rates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """A failed Run could have lowered the rates, so the Experiment gets no placement success file. The rates of
    the Runs that completed are printed, and the failed Run and a Run without diagnostics are named."""
    diagnostics_per_run = {"pick_apple": [_diagnostics(25, 150, 250, 3, 2)]}
    runs = ["pick_apple", "place_lime", "no_relations"]
    _write_experiment(tmp_path, runs, diagnostics_per_run, failed_runs=["place_lime"])

    assert export_kpis(tmp_path, ["placement_success"]) == []

    assert not (tmp_path / "kpis_placement_success.json").exists()
    output = capsys.readouterr().out
    assert "  pick_apple: valid candidates 60.0%, valid environments 88.0%, 2.00 batches per build" in output
    assert "  place_lime: failed" in output
    assert "  no_relations: no placement diagnostics" in output


def _write_experiment(
    experiment_output_directory: Path,
    run_names: list[str],
    diagnostics_per_run: dict[str, list[PlacementDiagnostics]],
    failed_runs: list[str] = (),
) -> None:
    """Write the parts of an Experiment output directory that placement success reads: the Experiment result and
    the diagnostics of each Run's builds, as the experiment runner writes them."""
    result = {
        "git_stamp": {"git_sha": "1a2b3c4", "git_branch": "main"},
        "experiment_name": EXPERIMENT_NAME,
        "runs": {
            run_name: {
                "status": "failed" if run_name in failed_runs else "completed",
                "end_timestamp_ns": END_TIMESTAMP_NS - index,
            }
            for index, run_name in enumerate(run_names)
        },
    }
    (experiment_output_directory / ARENA_EXPERIMENT_RESULT_FILENAME).write_text(json.dumps(result), encoding="utf-8")
    for run_name, diagnostics_per_build in diagnostics_per_run.items():
        (experiment_output_directory / run_name).mkdir()
        AggregatedPlacementDiagnostics(diagnostics_per_build).save(
            experiment_output_directory / run_name / ARENA_SINGLE_RUN_PLACEMENT_DIAGNOSTICS_FILENAME
        )


def _diagnostics(
    num_envs: int, num_valid: int, num_candidates: int, num_fallback_envs: int, num_batches: int
) -> PlacementDiagnostics:
    """Return diagnostics whose first batch has the given counts, followed by copies of it for later batches."""
    batch = BatchValidationDiagnostics(num_candidates, num_valid, {})
    return PlacementDiagnostics(num_envs, [batch] * num_batches, list(range(num_fallback_envs)))

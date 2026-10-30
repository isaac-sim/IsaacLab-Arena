# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the placement experiments and for reading placement success from an Experiment output directory."""

import json
import yaml
from pathlib import Path

import pytest

from isaaclab_arena.evaluation.arena_experiment_result import (
    ARENA_EXPERIMENT_RESULT_FILENAME,
    PLACEMENT_DIAGNOSTICS_FILENAME,
)
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_diagnostics import BatchValidationDiagnostics, PlacementDiagnostics
from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app
from isaaclab_arena_benchmarking.object_placement.placement_success import (
    RunPlacementRecord,
    format_placement_success,
    read_run_placement_records,
    summarize_placement_success,
)

REPO_ROOT = Path(TestConstants.repo_root)
EXPERIMENT_CONFIGS_DIR = REPO_ROOT / "isaaclab_arena_benchmarking" / "experiment_configs"


def _test_placement_experiment_run_writes_its_placement_diagnostics(simulation_app, tmp_path: Path) -> bool:
    from isaaclab_arena.evaluation.arena_experiment_config_loader import load_arena_experiment_from_config_file
    from isaaclab_arena.evaluation.experiment_runner import _write_arena_experiment_result
    from isaaclab_arena.evaluation.run_execution import execute_experiment

    # One Run of the RoboLab placement experiment, with its shared settings and two environments.
    experiment = yaml.safe_load((EXPERIMENT_CONFIGS_DIR / "robolab_placement_experiment.yaml").read_text())
    experiment["shared"]["environment_builder"]["num_envs"] = 2
    experiment["runs"] = {"banana_on_plate": experiment["runs"]["banana_on_plate"]}
    config_path = tmp_path / "banana_on_plate_placement_experiment.yaml"
    config_path.write_text(yaml.safe_dump(experiment))
    experiment_cfg = load_arena_experiment_from_config_file(config_path, device="cuda:0")
    output_dir = tmp_path / "output"

    run_results = execute_experiment(experiment_cfg, output_dir=output_dir)
    _write_arena_experiment_result(experiment_cfg, run_results, output_dir)

    assert (output_dir / "banana_on_plate" / "placement_diagnostics_rebuild0.json").is_file()
    [record] = read_run_placement_records(output_dir)
    assert (record.run_name, record.failed, record.num_builds, record.num_envs) == ("banana_on_plate", False, 1, 2)
    assert 0 < record.num_valid_candidates <= record.num_candidates
    return True


def test_placement_experiment_run_writes_its_placement_diagnostics(tmp_path):
    """A placement experiment Run writes its build's placement diagnostics, which give the Run's placement success."""
    assert run_function_with_persistent_simulation_app(
        _test_placement_experiment_run_writes_its_placement_diagnostics, tmp_path=tmp_path
    )


@pytest.mark.parametrize(
    "experiment_config, spec_dir",
    [
        ("robolab_placement_experiment.yaml", "isaaclab_arena_environments/robolab/tasks"),
        ("kitchen_bench_placement_experiment.yaml", "isaaclab_arena_environments/kitchen_bench"),
    ],
)
def test_placement_experiment_has_one_run_per_spec(experiment_config, spec_dir):
    """A placement experiment has one Run for every spec in its suite's folder, named after the spec."""
    runs = yaml.safe_load((EXPERIMENT_CONFIGS_DIR / experiment_config).read_text())["runs"]
    spec_per_run = {run_name: run["environment"]["type"] for run_name, run in runs.items()}
    spec_paths = sorted((REPO_ROOT / spec_dir).glob("*.yaml"))
    assert spec_per_run == {spec_path.stem: f"{spec_dir}/{spec_path.name}" for spec_path in spec_paths}


@pytest.mark.parametrize(
    "experiment_config", ["robolab_placement_experiment.yaml", "kitchen_bench_placement_experiment.yaml"]
)
def test_placement_experiment_runs_solve_separate_candidate_seeds(experiment_config):
    """No two Runs of a placement experiment share candidate seeds, so Runs that share a scene solve different
    candidates."""
    experiment = yaml.safe_load((EXPERIMENT_CONFIGS_DIR / experiment_config).read_text())
    # No spec sets its own placer parameters. A build solves at most max_placement_attempts batches, each of at most
    # num_envs * min_unique_layouts_per_env * max_placement_attempts candidates, seeded consecutively.
    params = ObjectPlacerParams()
    num_envs = experiment["shared"]["environment_builder"]["num_envs"]
    seeds_per_build = num_envs * params.min_unique_layouts_per_env * params.max_placement_attempts**2
    seeds = sorted(run["environment_builder"]["placement_seed"] for run in experiment["runs"].values())
    assert all(next_seed - seed >= seeds_per_build for seed, next_seed in zip(seeds, seeds[1:]))


def test_record_counts_the_first_pool_batch_of_each_build():
    """Candidates come from each build's first pool batch; later batches add only to the batch count."""
    first_batch = BatchValidationDiagnostics(num_candidates=100, num_valid=60, checks={})
    later_batch = BatchValidationDiagnostics(num_candidates=20, num_valid=15, checks={})
    record = RunPlacementRecord("open_fridge")

    record.add_build(PlacementDiagnostics(10, [first_batch, later_batch], fallback_envs=[3]))
    record.add_build(PlacementDiagnostics(10, [first_batch]))

    assert (record.num_valid_candidates, record.num_candidates) == (120, 200)
    assert (record.num_fallback_envs, record.num_envs, record.num_batches, record.num_builds) == (1, 20, 3, 2)


def test_experiment_placement_success_weighs_runs_equally_and_needs_every_run_to_succeed(tmp_path):
    """Scored Runs weigh the same whatever their sample size, Runs without diagnostics are listed but not scored, and
    a failed Run leaves the Experiment without rates."""
    runs = {"pick_apple": "completed", "pick_banana": "completed", "no_relations": "completed"}
    diagnostics_per_run = {
        # 45 of 50 environments valid over two rebuilds, at 300 of 500 valid candidates.
        "pick_apple": {0: _diagnostics(25, 150, 250, 3, 2), 1: _diagnostics(25, 150, 250, 2, 1)},
        # Rebuild 0 replays recorded layouts and writes no diagnostics; rebuild 1 solves them live.
        "pick_banana": {1: _diagnostics(25, 250, 250, 0, 1)},
    }
    _write_experiment(tmp_path / "completed", runs, diagnostics_per_run)
    _write_experiment(tmp_path / "with_failure", {**runs, "place_lime": "failed"}, diagnostics_per_run)

    records = read_run_placement_records(tmp_path / "completed")

    assert [(record.run_name, record.num_builds) for record in records] == [
        ("pick_apple", 2),
        ("pick_banana", 1),
        ("no_relations", 0),
    ]
    summary = summarize_placement_success(records)
    assert (summary.num_scored, summary.num_failed) == (2, 0)
    assert summary.valid_candidate_rate_first_batch == pytest.approx((0.6 + 1.0) / 2)
    assert summary.valid_env_rate_after_retries == pytest.approx((0.9 + 1.0) / 2)
    assert summary.batches_per_build == pytest.approx((1.5 + 1.0) / 2)
    table = format_placement_success(records)
    assert "| pick_apple | 60.0% | 90.0% | 1.50 |" in table
    assert "| no_relations | no diagnostics | no diagnostics | no diagnostics |" in table

    # A failed Run could have lowered the rates, so they are left out.
    records = read_run_placement_records(tmp_path / "with_failure")
    summary = summarize_placement_success(records)
    assert (summary.num_scored, summary.num_failed, summary.valid_candidate_rate_first_batch) == (2, 1, None)
    table = format_placement_success(records)
    assert "| place_lime | failed | failed | failed |" in table
    assert table.endswith("| mean of 2 scored, 1 failed | n/a | n/a | n/a |")


def _write_experiment(
    experiment_output_directory: Path,
    runs: dict[str, str],
    diagnostics_per_run: dict[str, dict[int, PlacementDiagnostics]],
) -> None:
    """Write the parts of an Experiment output directory that placement success reads: Run statuses and diagnostics,
    keyed by rebuild index."""
    experiment_output_directory.mkdir()
    result = {"runs": {run_name: {"status": status} for run_name, status in runs.items()}}
    (experiment_output_directory / ARENA_EXPERIMENT_RESULT_FILENAME).write_text(json.dumps(result))
    for run_name, diagnostics_per_rebuild in diagnostics_per_run.items():
        (experiment_output_directory / run_name).mkdir()
        for rebuild_index, diagnostics in diagnostics_per_rebuild.items():
            filename = PLACEMENT_DIAGNOSTICS_FILENAME.format(rebuild_index=rebuild_index)
            diagnostics.save(experiment_output_directory / run_name / filename)


def _diagnostics(
    num_envs: int, num_valid: int, num_candidates: int, num_fallback_envs: int, num_batches: int
) -> PlacementDiagnostics:
    """Return diagnostics whose first batch has the given counts, followed by copies of it for later batches."""
    batch = BatchValidationDiagnostics(num_candidates, num_valid, {})
    return PlacementDiagnostics(num_envs, [batch] * num_batches, list(range(num_fallback_envs)))

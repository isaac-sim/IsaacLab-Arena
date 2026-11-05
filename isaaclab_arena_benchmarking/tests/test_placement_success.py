# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the placement experiments and for reading placement success from an Experiment output directory."""

import json
import yaml
from pathlib import Path

import pytest

from isaaclab_arena.environment_spec.arena_env_graph_yaml_loader import load_env_graph_spec_dict
from isaaclab_arena.evaluation.arena_experiment_result import (
    ARENA_EXPERIMENT_RESULT_FILENAME,
    ARENA_SINGLE_RUN_PLACEMENT_DIAGNOSTICS_FILENAME,
)
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_diagnostics import (
    AggregatedPlacementDiagnostics,
    BatchValidationDiagnostics,
    PlacementDiagnostics,
)
from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app
from isaaclab_arena_benchmarking.object_placement.compute_placement_kpis_from_experiment_output import (
    RunPlacementCounts,
    format_placement_success,
    read_run_placement_counts,
    summarize_placement_success,
)

REPO_ROOT = Path(TestConstants.repo_root)
EXPERIMENT_CONFIGS_DIR = REPO_ROOT / "isaaclab_arena_benchmarking" / "experiment_configs"


def _test_placement_experiment_run_builds_with_placement_diagnostics(simulation_app) -> bool:
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

    # One Run of the RoboLab placement experiment, built from its spec with its placement seed and two environments.
    # Building the spec directly keeps the test independent of the CLI arguments and environment registrations
    # that other tests leave in a shared pytest process.
    experiment = yaml.safe_load((EXPERIMENT_CONFIGS_DIR / "robolab_placement_experiment.yaml").read_text())
    run = experiment["runs"]["banana_on_plate"]
    arena_env = ArenaEnvGraphSpec.from_yaml(REPO_ROOT / run["environment"]["type"]).to_arena_env()
    builder_cfg = ArenaEnvBuilderCfg(num_envs=2, placement_seed=run["environment_builder"]["placement_seed"])
    builder = ArenaEnvBuilder(arena_env, builder_cfg)
    builder.build_env_cfg()

    placement_diagnostics = builder.placement_diagnostics
    assert placement_diagnostics is not None and placement_diagnostics.num_envs == 2
    assert 0 < placement_diagnostics.batches[0].num_valid <= placement_diagnostics.batches[0].num_candidates
    return True


def test_placement_experiment_run_builds_with_placement_diagnostics():
    """A placement experiment Run's spec builds and records the placement diagnostics the Run writes."""
    assert run_function_with_persistent_simulation_app(_test_placement_experiment_run_builds_with_placement_diagnostics)


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
    # The bound below uses the default placer parameters, so no spec may override them.
    for run in experiment["runs"].values():
        spec_placer_params = load_env_graph_spec_dict(REPO_ROOT / run["environment"]["type"]).get("placer_params")
        overridden = {"min_unique_layouts_per_env", "max_placement_attempts"} & set(spec_placer_params or {})
        assert not overridden, f"{run['environment']['type']} sets {sorted(overridden)}; widen the seed spacing"
    # A build solves at most max_placement_attempts batches, each of at most
    # num_envs * min_unique_layouts_per_env * max_placement_attempts candidates, seeded consecutively.
    params = ObjectPlacerParams()
    num_envs = experiment["shared"]["environment_builder"]["num_envs"]
    seeds_per_build = num_envs * params.min_unique_layouts_per_env * params.max_placement_attempts**2
    seeds = sorted(run["environment_builder"]["placement_seed"] for run in experiment["runs"].values())
    assert all(next_seed - seed >= seeds_per_build for seed, next_seed in zip(seeds, seeds[1:]))


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


def test_experiment_placement_success_weights_runs_equally_and_needs_every_run_to_succeed(tmp_path):
    """Scored Runs are weighted equally whatever their sample size, Runs without diagnostics are listed but not
    scored, and a failed Run leaves the Experiment without rates."""
    runs = {"pick_apple": "completed", "pick_banana": "completed", "no_relations": "completed"}
    diagnostics_per_run = {
        # 45 of 50 environments valid over two rebuilds, at 300 of 500 valid candidates.
        "pick_apple": [_diagnostics(25, 150, 250, 3, 2), _diagnostics(25, 150, 250, 2, 1)],
        # 25 of 25 environments valid over one rebuild, at 250 of 250 valid candidates.
        "pick_banana": [_diagnostics(25, 250, 250, 0, 1)],
    }
    _write_experiment(tmp_path / "completed", runs, diagnostics_per_run)
    # A failed Run writes the diagnostics of the fills it made, but they are not scored.
    _write_experiment(
        tmp_path / "with_failure",
        {**runs, "place_lime": "failed"},
        {**diagnostics_per_run, "place_lime": [_diagnostics(25, 0, 250, 0, 2)]},
    )

    counts_per_run = read_run_placement_counts(tmp_path / "completed")

    assert [(run_counts.run_name, run_counts.is_scored) for run_counts in counts_per_run] == [
        ("pick_apple", True),
        ("pick_banana", True),
        ("no_relations", False),
    ]
    summary = summarize_placement_success(counts_per_run)
    assert (summary.num_scored, summary.num_failed) == (2, 0)
    assert summary.valid_candidate_rate_first_batch == pytest.approx((0.6 + 1.0) / 2)
    assert summary.valid_env_rate_after_retries == pytest.approx((0.9 + 1.0) / 2)
    assert summary.batches_per_build == pytest.approx((1.5 + 1.0) / 2)
    table = format_placement_success(counts_per_run)
    assert "| pick_apple | 60.0% | 90.0% | 1.50 |" in table
    assert "| no_relations | no diagnostics | no diagnostics | no diagnostics |" in table

    # A failed Run could have lowered the rates, so they are left out.
    counts_per_run = read_run_placement_counts(tmp_path / "with_failure")
    summary = summarize_placement_success(counts_per_run)
    assert (summary.num_scored, summary.num_failed, summary.valid_candidate_rate_first_batch) == (2, 1, None)
    table = format_placement_success(counts_per_run)
    assert "| place_lime | failed | failed | failed |" in table
    assert table.endswith("| mean of 2 scored, 1 failed | n/a | n/a | n/a |")


def _write_experiment(
    experiment_output_directory: Path,
    runs: dict[str, str],
    diagnostics_per_run: dict[str, list[PlacementDiagnostics]],
) -> None:
    """Write the parts of an Experiment output directory that placement success reads: Run statuses and the
    diagnostics of each Run's builds, as the experiment runner writes them."""
    experiment_output_directory.mkdir()
    result = {"runs": {run_name: {"status": status} for run_name, status in runs.items()}}
    (experiment_output_directory / ARENA_EXPERIMENT_RESULT_FILENAME).write_text(json.dumps(result))
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

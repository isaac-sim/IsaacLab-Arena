# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the placement Experiments."""

import yaml
from pathlib import Path

import pytest

from isaaclab_arena.environment_spec.arena_env_graph_yaml_loader import load_env_graph_spec_dict
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

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
    "experiment_config",
    [
        "robolab_placement_experiment.yaml",
        "kitchen_bench_placement_experiment.yaml",
        "placement_subset_experiment.yaml",
    ],
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


def test_placement_subset_experiment_runs_are_runs_of_the_suites():
    """Each Run of the placement subset Experiment is identical to the Run of the same name in a suite Experiment."""
    suite_runs = {}
    for suite_config in ("robolab_placement_experiment.yaml", "kitchen_bench_placement_experiment.yaml"):
        suite_runs.update(yaml.safe_load((EXPERIMENT_CONFIGS_DIR / suite_config).read_text())["runs"])
    subset_runs = yaml.safe_load((EXPERIMENT_CONFIGS_DIR / "placement_subset_experiment.yaml").read_text())["runs"]
    assert len(subset_runs) == 10
    for run_name, run in subset_runs.items():
        assert run_name in suite_runs, f"{run_name} is not a Run of a placement suite"
        assert run == suite_runs[run_name]


@pytest.mark.parametrize(
    "experiment_config",
    [
        "robolab_placement_experiment.yaml",
        "kitchen_bench_placement_experiment.yaml",
        "placement_subset_experiment.yaml",
    ],
)
def test_placement_experiment_selects_the_placement_kpi_exporters(experiment_config):
    """A placement Experiment selects the placement success and timing KPI exporters."""
    experiment = yaml.safe_load((EXPERIMENT_CONFIGS_DIR / experiment_config).read_text())
    assert experiment["kpi_exporters"] == ["placement_success", "timing"]

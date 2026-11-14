# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify that the VLM policy benchmark Experiments differ only by their policy or their number of environments."""

from dataclasses import replace
from pathlib import Path

import pytest

from isaaclab_arena.assets import registries
from isaaclab_arena.evaluation import arena_experiment_config_loader as loader
from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena_vlm_agent.vlm_agent_policy import VLMChunkPolicyCfg, VLMGoalPolicyCfg

EXPERIMENT_CONFIGS_DIR = Path(TestConstants.repo_root) / "isaaclab_arena_benchmarking" / "experiment_configs"


@pytest.fixture
def load_experiment(monkeypatch):
    """Return a function loading a benchmark Experiment by file name, without registering assets."""
    from isaaclab_arena_environments.pick_and_place_maple_table_environment import PickAndPlaceMapleTableEnvironmentCfg

    # Registering assets needs Isaac Sim.
    monkeypatch.setattr(registries, "ensure_assets_registered", lambda: None)
    monkeypatch.setattr(
        loader,
        "_registered_environment_cfg_types",
        lambda: {"pick_and_place_maple_table": PickAndPlaceMapleTableEnvironmentCfg},
    )
    return lambda filename: loader.load_arena_experiment_from_config_file(
        EXPERIMENT_CONFIGS_DIR / filename, device="cpu"
    )


def test_vlm_chunk_and_goal_benchmarks_differ_only_by_policy(load_experiment):
    chunk_experiment = load_experiment("vlm_chunk_policy_benchmark_experiment.yaml")
    goal_experiment = load_experiment("vlm_goal_policy_benchmark_experiment.yaml")

    assert chunk_experiment.name == "vlm_chunk_policy_benchmark_experiment"
    assert goal_experiment.name == "vlm_goal_policy_benchmark_experiment"
    assert chunk_experiment.kpi_exporters == goal_experiment.kpi_exporters == ["success_rate", "timing"]
    (chunk_run,) = chunk_experiment.runs.values()
    (goal_run,) = goal_experiment.runs.values()
    assert isinstance(chunk_run.policy, VLMChunkPolicyCfg)
    assert isinstance(goal_run.policy, VLMGoalPolicyCfg)
    # One environment, so every VLM call is timed without concurrent requests.
    assert chunk_run.environment_builder.num_envs == 1
    assert chunk_run.policy.endpoint == goal_run.policy.endpoint
    assert chunk_run.policy.model == goal_run.policy.model
    assert chunk_run.policy.observation_modality == goal_run.policy.observation_modality
    assert replace(chunk_run, name=goal_run.name, policy=goal_run.policy) == goal_run


@pytest.mark.parametrize("policy", ["chunk", "goal"])
def test_parallel_envs_benchmarks_differ_only_by_number_of_environments(load_experiment, policy):
    experiment = load_experiment(f"vlm_{policy}_policy_benchmark_experiment.yaml")
    parallel_experiment = load_experiment(f"vlm_{policy}_policy_parallel_envs_benchmark_experiment.yaml")

    assert parallel_experiment.name == f"vlm_{policy}_policy_parallel_envs_benchmark_experiment"
    assert parallel_experiment.kpi_exporters == experiment.kpi_exporters
    (run,) = experiment.runs.values()
    (parallel_run,) = parallel_experiment.runs.values()
    # Every environment's request is sent at once.
    assert 1 < parallel_run.environment_builder.num_envs <= parallel_run.policy.max_concurrent_requests
    expected_parallel_run = replace(
        run,
        name=parallel_run.name,
        environment_builder=replace(run.environment_builder, num_envs=parallel_run.environment_builder.num_envs),
        rollout_limit=parallel_run.rollout_limit,
    )
    assert expected_parallel_run == parallel_run

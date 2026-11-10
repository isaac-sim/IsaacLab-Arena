# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify that the VLM chunk and goal policy benchmark Experiments differ only by their policy."""

from dataclasses import replace
from pathlib import Path

from isaaclab_arena.assets import registries
from isaaclab_arena.evaluation import arena_experiment_config_loader as loader
from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena_vlm_agent.vlm_agent_policy import VLMChunkPolicyCfg, VLMGoalPolicyCfg

EXPERIMENT_CONFIGS_DIR = Path(TestConstants.repo_root) / "isaaclab_arena_benchmarking" / "experiment_configs"


def test_vlm_chunk_and_goal_benchmarks_differ_only_by_policy(monkeypatch):
    from isaaclab_arena_environments.pick_and_place_maple_table_environment import PickAndPlaceMapleTableEnvironmentCfg

    # Load without registering assets, which needs Isaac Sim.
    monkeypatch.setattr(registries, "ensure_assets_registered", lambda: None)
    monkeypatch.setattr(
        loader,
        "_registered_environment_cfg_types",
        lambda: {"pick_and_place_maple_table": PickAndPlaceMapleTableEnvironmentCfg},
    )
    chunk_experiment = loader.load_arena_experiment_from_config_file(
        EXPERIMENT_CONFIGS_DIR / "vlm_chunk_policy_benchmark_experiment.yaml", device="cpu"
    )
    goal_experiment = loader.load_arena_experiment_from_config_file(
        EXPERIMENT_CONFIGS_DIR / "vlm_goal_policy_benchmark_experiment.yaml", device="cpu"
    )

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

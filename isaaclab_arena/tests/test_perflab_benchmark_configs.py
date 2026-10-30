# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify the checked-in PerfLab workload contract."""

from dataclasses import replace
from pathlib import Path

import pytest

from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.environments.arena_env_builder_cfg import CameraResolutionMode
from isaaclab_arena.evaluation.arena_experiment_config_loader import load_arena_experiment_from_config_file
from isaaclab_arena.hydra.typed_experiment_loader import load_experiment_run_definitions_from_yaml
from isaaclab_arena.tests.utils.constants import TestConstants

_PERFLAB_CONFIG_DIRECTORY = Path(TestConstants.arena_environments_dir) / "experiment_configs" / "perflab"

_RUN_NAME_BY_CONFIG_NAME = {
    "camera_free_benchmark_experiment.yaml": "camera_free_baseline",
    "production_camera_benchmark_experiment.yaml": "production_camera_baseline",
    "pi05_benchmark_experiment.yaml": "pi05_evaluation",
    "cosmos_benchmark_experiment.yaml": "cosmos_evaluation",
    "gr00t_benchmark_experiment.yaml": "gr00t_evaluation",
    "same_object_benchmark_experiment.yaml": "same_object_control",
    "mixed_object_benchmark_experiment.yaml": "mixed_object_workload",
}

_ENVIRONMENT_NAME_BY_OBJECT_CONFIG_NAME = {
    "same_object_benchmark_experiment.yaml": "same_object_environment.yaml",
    "mixed_object_benchmark_experiment.yaml": "mixed_object_environment.yaml",
}


def _load_single_run(config_name: str) -> dict:
    run_definitions = load_experiment_run_definitions_from_yaml(_PERFLAB_CONFIG_DIRECTORY / config_name)
    expected_run_name = _RUN_NAME_BY_CONFIG_NAME[config_name]
    assert list(run_definitions) == [expected_run_name]
    return run_definitions[expected_run_name]


def test_perflab_experiments_are_isolated_fixed_step_runs():
    """Keep every sweep point in one fresh process with the same rollout length."""
    for config_name in _RUN_NAME_BY_CONFIG_NAME:
        run_definition = _load_single_run(config_name)
        assert run_definition["environment_builder"]["num_envs"] == 1
        assert run_definition["environment_builder"]["placement_seed"] == 42
        assert run_definition["rollout_limit"] == {"num_steps": 300}
        assert run_definition["num_rebuilds"] == 1


@pytest.mark.parametrize("config_name", _RUN_NAME_BY_CONFIG_NAME)
def test_perflab_experiments_compose_through_the_typed_loader(config_name):
    """Check every shipped workload against its environment and policy schema."""
    experiment_cfg = load_arena_experiment_from_config_file(
        _PERFLAB_CONFIG_DIRECTORY / config_name,
        device="cuda:0",
    )

    assert list(experiment_cfg.runs) == [_RUN_NAME_BY_CONFIG_NAME[config_name]]


def test_camera_workloads_render_at_the_embodiment_camera_resolution():
    """Keep the camera workloads on the dimensions declared by the maintained DROID camera rig."""
    camera_configs = [
        "production_camera_benchmark_experiment.yaml",
        "pi05_benchmark_experiment.yaml",
        "cosmos_benchmark_experiment.yaml",
        "gr00t_benchmark_experiment.yaml",
    ]
    for config_name in camera_configs:
        experiment_cfg = load_arena_experiment_from_config_file(
            _PERFLAB_CONFIG_DIRECTORY / config_name, device="cuda:0"
        )
        run_cfg = experiment_cfg.runs[_RUN_NAME_BY_CONFIG_NAME[config_name]]
        assert run_cfg.environment.enable_cameras
        assert run_cfg.environment_builder.camera_resolution_mode is CameraResolutionMode.EMBODIMENT_DEFAULT


def test_zero_action_workloads_use_stationary_relative_joint_actions():
    """A zero action must leave the robot near its initial joint configuration."""
    registered_environment_configs = [
        "camera_free_benchmark_experiment.yaml",
        "production_camera_benchmark_experiment.yaml",
    ]
    for config_name in registered_environment_configs:
        run_definition = _load_single_run(config_name)
        assert run_definition["environment"]["embodiment"] == "droid_rel_joint_pos"
        assert run_definition["policy"]["type"] == "zero_action"

    for config_name, environment_name in _ENVIRONMENT_NAME_BY_OBJECT_CONFIG_NAME.items():
        run_definition = _load_single_run(config_name)
        environment_spec_path = Path(TestConstants.repo_root) / run_definition["environment"]["type"]
        assert environment_spec_path == _PERFLAB_CONFIG_DIRECTORY / environment_name
        assert ArenaEnvGraphSpec.from_yaml(environment_spec_path).embodiment.registry_name == "droid_rel_joint_pos"
        assert run_definition["policy"]["type"] == "zero_action"


def test_same_and_mixed_object_workloads_differ_only_by_object_set_members():
    """Isolate the scene-build cost of one asset type versus ten asset types."""
    same_object_spec = ArenaEnvGraphSpec.from_yaml(_PERFLAB_CONFIG_DIRECTORY / "same_object_environment.yaml")
    mixed_object_spec = ArenaEnvGraphSpec.from_yaml(_PERFLAB_CONFIG_DIRECTORY / "mixed_object_environment.yaml")
    (same_object_set,) = same_object_spec.object_sets
    (mixed_object_set,) = mixed_object_spec.object_sets
    assert same_object_set.members == ["apple_01_objaverse_robolab"]
    assert len(set(mixed_object_set.members)) == 10
    # Members are assigned in list order, so every fruit appears once there are at least ten environments.
    assert not mixed_object_set.random_choice

    normalized_same_object_spec = same_object_spec.model_copy(
        update={
            "env_name": mixed_object_spec.env_name,
            "object_sets": [same_object_set.model_copy(update={"members": mixed_object_set.members})],
        }
    )
    assert normalized_same_object_spec == mixed_object_spec

    same_object_run = load_arena_experiment_from_config_file(
        _PERFLAB_CONFIG_DIRECTORY / "same_object_benchmark_experiment.yaml",
        device="cuda:0",
    ).runs["same_object_control"]
    mixed_object_run = load_arena_experiment_from_config_file(
        _PERFLAB_CONFIG_DIRECTORY / "mixed_object_benchmark_experiment.yaml",
        device="cuda:0",
    ).runs["mixed_object_workload"]

    normalized_environment = replace(
        same_object_run.environment, env_spec_path=mixed_object_run.environment.env_spec_path
    )
    normalized_same_object_run = replace(
        same_object_run,
        name=mixed_object_run.name,
        environment=normalized_environment,
    )
    assert normalized_same_object_run == mixed_object_run

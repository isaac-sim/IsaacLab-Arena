# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify the checked-in PerfLab workload contract."""

from dataclasses import replace
from pathlib import Path

import pytest

from isaaclab_arena.evaluation.arena_experiment_config_loader import load_arena_experiment_from_config_file
from isaaclab_arena.hydra.typed_experiment_loader import load_experiment_run_definitions_from_yaml
from isaaclab_arena.tests.utils.constants import TestConstants

_PERFLAB_CONFIG_DIRECTORY = Path(TestConstants.arena_environments_dir) / "experiment_configs" / "perflab"

_RUN_NAME_BY_CONFIG_NAME = {
    "camera_free_benchmark_experiment.yaml": "camera_free_baseline",
    "production_camera_benchmark_experiment.yaml": "production_camera_baseline",
    "reduced_camera_benchmark_experiment.yaml": "reduced_camera_control",
    "pi05_benchmark_experiment.yaml": "pi05_evaluation",
    "cosmos_benchmark_experiment.yaml": "cosmos_evaluation",
    "gr00t_benchmark_experiment.yaml": "gr00t_evaluation",
    "same_object_benchmark_experiment.yaml": "same_object_control",
    "mixed_object_benchmark_experiment.yaml": "mixed_object_workload",
}


def _load_single_run(config_name: str) -> dict:
    run_definitions = load_experiment_run_definitions_from_yaml(_PERFLAB_CONFIG_DIRECTORY / config_name)
    expected_run_name = _RUN_NAME_BY_CONFIG_NAME[config_name]
    assert list(run_definitions) == [expected_run_name]
    return run_definitions[expected_run_name]


def test_perflab_experiments_are_isolated_fixed_step_runs():
    """Keep every sweep point in one fresh process with the same provisional smoke length."""
    for config_name in _RUN_NAME_BY_CONFIG_NAME:
        run_definition = _load_single_run(config_name)
        assert run_definition["environment_builder"]["num_envs"] == 1
        assert run_definition["environment_builder"]["placement_seed"] == 42
        assert run_definition["rollout_limit"] == {"num_steps": 10}
        assert run_definition["num_rebuilds"] == 1


@pytest.mark.parametrize("config_name", _RUN_NAME_BY_CONFIG_NAME)
def test_perflab_experiments_compose_through_the_typed_loader(config_name):
    """Check every shipped workload against its registered environment and policy schema."""
    experiment_cfg = load_arena_experiment_from_config_file(
        _PERFLAB_CONFIG_DIRECTORY / config_name,
        device="cuda:0",
    )

    assert list(experiment_cfg.runs) == [_RUN_NAME_BY_CONFIG_NAME[config_name]]


def test_reduced_camera_control_accepts_only_paired_shared_resolution_overrides():
    """Expose an explicit reduced-camera override without accepting a partial image size."""
    config_path = _PERFLAB_CONFIG_DIRECTORY / "reduced_camera_benchmark_experiment.yaml"
    experiment_cfg = load_arena_experiment_from_config_file(
        config_path,
        device="cuda:0",
        overrides=[
            "shared.environment_builder.camera_height=360",
            "shared.environment_builder.camera_width=640",
        ],
    )
    environment_builder = experiment_cfg.runs["reduced_camera_control"].environment_builder

    assert (environment_builder.camera_height, environment_builder.camera_width) == (360, 640)
    with pytest.raises(AssertionError, match="camera_height and camera_width must be set together"):
        load_arena_experiment_from_config_file(
            config_path,
            device="cuda:0",
            overrides=["shared.environment_builder.camera_height=360"],
        )


def test_only_reduced_camera_control_declares_shared_resolution_overrides():
    """Keep the standard shared command from reducing production-policy camera dimensions."""
    reduced_camera_builder = _load_single_run("reduced_camera_benchmark_experiment.yaml")["environment_builder"]
    assert reduced_camera_builder["camera_height"] is None
    assert reduced_camera_builder["camera_width"] is None

    production_camera_configs = [
        "production_camera_benchmark_experiment.yaml",
        "pi05_benchmark_experiment.yaml",
        "cosmos_benchmark_experiment.yaml",
        "gr00t_benchmark_experiment.yaml",
    ]
    for config_name in production_camera_configs:
        environment_builder = _load_single_run(config_name)["environment_builder"]
        assert "camera_height" not in environment_builder
        assert "camera_width" not in environment_builder


def test_zero_action_workloads_use_stationary_relative_joint_actions():
    """A zero action must leave the robot near its initial joint configuration."""
    registered_zero_action_configs = [
        "camera_free_benchmark_experiment.yaml",
        "production_camera_benchmark_experiment.yaml",
        "reduced_camera_benchmark_experiment.yaml",
        "same_object_benchmark_experiment.yaml",
        "mixed_object_benchmark_experiment.yaml",
    ]
    for config_name in registered_zero_action_configs:
        run_definition = _load_single_run(config_name)
        assert run_definition["environment"]["embodiment"] == "droid_rel_joint_pos"
        assert run_definition["policy"]["type"] == "zero_action"


def test_same_and_mixed_object_workloads_differ_only_by_object_set_members():
    """Isolate the scene-build cost of one asset type versus ten asset types."""
    same_object_experiment = load_arena_experiment_from_config_file(
        _PERFLAB_CONFIG_DIRECTORY / "same_object_benchmark_experiment.yaml",
        device="cuda:0",
    )
    mixed_object_experiment = load_arena_experiment_from_config_file(
        _PERFLAB_CONFIG_DIRECTORY / "mixed_object_benchmark_experiment.yaml",
        device="cuda:0",
    )
    same_object_run = same_object_experiment.runs["same_object_control"]
    mixed_object_run = mixed_object_experiment.runs["mixed_object_workload"]

    same_object_set = same_object_run.environment.object_set
    mixed_object_set = mixed_object_run.environment.object_set
    assert same_object_set == ["apple_01_objaverse_robolab"]
    assert mixed_object_set is not None
    assert len(mixed_object_set) == 10

    normalized_environment = replace(same_object_run.environment, object_set=mixed_object_set)
    normalized_same_object_run = replace(
        same_object_run,
        name=mixed_object_run.name,
        environment=normalized_environment,
    )
    assert normalized_same_object_run == mixed_object_run

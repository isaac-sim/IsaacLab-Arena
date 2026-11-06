# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify the checked-in PerfLab workload contract."""

from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.evaluation.arena_experiment_config_loader import load_arena_experiment_from_config_file
from isaaclab_arena.evaluation.legacy_graph_environment_cli import LegacyGraphEnvironmentCfg
from isaaclab_arena.tests.utils.constants import TestConstants

_PERFLAB_DIRECTORY = Path(TestConstants.arena_environments_dir) / "performance"
_PERFLAB_CONFIG_DIRECTORY = _PERFLAB_DIRECTORY / "experiment_configs"
_PERFLAB_TASK_DIRECTORY = _PERFLAB_DIRECTORY / "tasks"


@dataclass(frozen=True)
class _PerfLabWorkload:
    """What one PerfLab Experiment must declare."""

    run_name: str
    """The single Run the Experiment declares."""

    enable_cameras: bool
    """Whether the workload renders the DROID camera rig."""

    embodiment: str
    """Registered embodiment name. Zero-action workloads need relative joint actions to keep the robot still."""

    uses_remote_policy: bool = False
    """Whether the guide's command also overrides the policy server endpoint."""


_WORKLOAD_BY_CONFIG_NAME = {
    "camera_free_benchmark_experiment.yaml": _PerfLabWorkload(
        "camera_free_baseline", enable_cameras=False, embodiment="droid_rel_joint_pos"
    ),
    "production_camera_benchmark_experiment.yaml": _PerfLabWorkload(
        "production_camera_baseline", enable_cameras=True, embodiment="droid_rel_joint_pos"
    ),
    "pi05_benchmark_experiment.yaml": _PerfLabWorkload(
        "pi05_evaluation", enable_cameras=True, embodiment="droid_abs_joint_pos", uses_remote_policy=True
    ),
    "cosmos_benchmark_experiment.yaml": _PerfLabWorkload(
        "cosmos_evaluation", enable_cameras=True, embodiment="droid_abs_joint_pos", uses_remote_policy=True
    ),
    "gr00t_benchmark_experiment.yaml": _PerfLabWorkload(
        "gr00t_evaluation", enable_cameras=True, embodiment="droid_abs_joint_pos", uses_remote_policy=True
    ),
    "same_object_benchmark_experiment.yaml": _PerfLabWorkload(
        "same_object_control", enable_cameras=False, embodiment="droid_rel_joint_pos"
    ),
    "mixed_object_benchmark_experiment.yaml": _PerfLabWorkload(
        "mixed_object_workload", enable_cameras=False, embodiment="droid_rel_joint_pos"
    ),
}

# Values that differ from the YAML defaults, so the assertions show the overrides were applied.
_NUM_ENVS = 64
_NUM_STEPS = 150
_REMOTE_HOST = "10.0.0.1"
_REMOTE_PORT = 8001


@pytest.mark.parametrize("config_name", _WORKLOAD_BY_CONFIG_NAME)
def test_perflab_experiments_load_with_the_guide_overrides(config_name):
    """Load each Experiment with the overrides from the PerfLab guide's commands and check its workload."""
    workload = _WORKLOAD_BY_CONFIG_NAME[config_name]
    overrides = [
        f"shared.environment_builder.num_envs={_NUM_ENVS}",
        f"shared.rollout_limit.num_steps={_NUM_STEPS}",
    ]
    if workload.uses_remote_policy:
        overrides += [f"shared.policy.remote_host={_REMOTE_HOST}", f"shared.policy.remote_port={_REMOTE_PORT}"]

    experiment_cfg = load_arena_experiment_from_config_file(
        _PERFLAB_CONFIG_DIRECTORY / config_name,
        device="cuda:0",
        overrides=overrides,
    )

    assert list(experiment_cfg.runs) == [workload.run_name]
    run_cfg = experiment_cfg.runs[workload.run_name]
    assert run_cfg.environment_builder.num_envs == _NUM_ENVS
    assert run_cfg.rollout_limit.num_steps == _NUM_STEPS
    if workload.uses_remote_policy:
        assert run_cfg.policy.remote_host == _REMOTE_HOST
        assert run_cfg.policy.remote_port == _REMOTE_PORT
    assert run_cfg.environment.enable_cameras == workload.enable_cameras
    if isinstance(run_cfg.environment, LegacyGraphEnvironmentCfg):
        environment_spec = ArenaEnvGraphSpec.from_yaml(
            Path(TestConstants.repo_root) / run_cfg.environment.env_spec_path
        )
        assert environment_spec.embodiment.registry_name == workload.embodiment
    else:
        assert run_cfg.environment.embodiment == workload.embodiment


def test_same_and_mixed_object_workloads_differ_only_by_object_set_members():
    """Isolate the scene-build cost of one asset type versus ten asset types."""
    same_object_spec = ArenaEnvGraphSpec.from_yaml(_PERFLAB_TASK_DIRECTORY / "same_object.yaml")
    mixed_object_spec = ArenaEnvGraphSpec.from_yaml(_PERFLAB_TASK_DIRECTORY / "mixed_object.yaml")
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

# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate the workbench cleanup scene and tasks."""

from pathlib import Path

import pytest

from isaaclab_arena.agentic_tools.object_control.object_control_recipe import (
    CheckCommand,
    MarkCommand,
    load_object_control_recipe,
)
from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec

_ENVIRONMENT_DIR = Path(__file__).resolve().parents[2] / "isaaclab_arena_environments/workbench_cleanup"
_TASK_PATHS = sorted(_ENVIRONMENT_DIR.glob("task_*.yaml"))


def _test_workbench_task_definitions_build(simulation_app) -> bool:
    from isaaclab_arena.environment_spec.arena_env_graph_conversion_utils import build_arena_env_from_graph_spec

    for path in sorted(_ENVIRONMENT_DIR.glob("task_*.yaml")):
        environment = build_arena_env_from_graph_spec(ArenaEnvGraphSpec.from_yaml(path))
        assert environment.name
    return True


# TODO(qianl, 2026.11.11): Re-enable after the workbench assets are published to public S3.
@pytest.mark.skip(reason="Workbench staging assets require ovstorage access unavailable in CI")
def test_workbench_task_definitions_build():
    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    assert run_function_with_persistent_simulation_app(_test_workbench_task_definitions_build)


def _test_workbench_vlm_experiment_configs(simulation_app):
    from isaaclab_arena.evaluation.arena_experiment_config_loader import load_arena_experiment_from_config_file
    from isaaclab_arena_vlm_agent.vlm_agent_policy import ObservationModality, VLMGoalPolicyCfg

    paths = sorted((_ENVIRONMENT_DIR / "experiment_config").glob("*_vlm_goal_experiment.yaml"))
    assert len(paths) == 4
    covered_tasks = set()
    for path in paths:
        experiment = load_arena_experiment_from_config_file(path, device="cpu")
        assert len(experiment.runs) == 1
        run = next(iter(experiment.runs.values()))
        assert isinstance(run.policy, VLMGoalPolicyCfg)
        assert run.policy.vlm_embodiment_adapter == "droid"
        assert run.policy.observation_modality == ObservationModality.image_state
        assert run.environment.enable_cameras
        assert run.environment_builder.num_envs == 1
        assert run.rollout_limit.num_episodes == 5
        spec = ArenaEnvGraphSpec.from_yaml(run.environment.env_spec_path)
        assert spec.task.composition.value == "atomic"
        assert spec.embodiment.registry_name == "droid_abs_ik"
        assert spec.embodiment.params["concatenate_observation_terms"] is False
        scene_keys = {
            spec.background.id,
            *[obj.id for obj in spec.objects],
            *[ref.id for ref in spec.object_references],
        }
        assert run.policy.object_poses and set(run.policy.object_poses) <= scene_keys
        assert run.policy.system_prompt_files
        covered_tasks.add(Path(run.environment.env_spec_path).name)
    assert covered_tasks == {
        "task_bin_shelf.yaml",
        "task_close_drawer.yaml",
        "task_close_toolcase.yaml",
        "task_turn_off_scope.yaml",
    }
    return True


def test_workbench_vlm_experiment_configs():
    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    assert run_function_with_persistent_simulation_app(_test_workbench_vlm_experiment_configs)


def _test_workbench_vlm_pose_tracking(simulation_app):
    import math
    import torch
    from unittest.mock import patch

    from isaaclab.utils.math import quat_apply, subtract_frame_transforms

    from isaaclab_arena_environments.workbench_cleanup.vlm_policy import (
        WorkbenchVLMGoalPolicy,
        WorkbenchVLMGoalPolicyCfg,
        bin_side_wall_gripper_poses,
        gripper_pose_for_held_object,
        limit_pose_increment,
    )

    T_R_start = torch.tensor([0.0, 0, 0, 0, 0, 0, 1])
    T_R_goal = torch.tensor([1.0, 0, 0, 0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4)])
    expected = torch.tensor([0.05, 0, 0, 0, 0, math.sin(0.05), math.cos(0.05)])
    torch.testing.assert_close(limit_pose_increment(T_R_start, T_R_goal, 0.05, 0.1), expected)
    torch.testing.assert_close(limit_pose_increment(T_R_start, expected, 0.5, 0.5), expected)
    T_R_same = T_R_start.clone()
    T_R_same[3:] *= -1
    result = limit_pose_increment(T_R_start, T_R_same, 0.05, 0.1)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result[3:].abs(), T_R_start[3:].abs())

    T_R_bin = torch.tensor([[0.5, 0.2, 0.3, 0, 0, 0, 1]])
    T_R_gripper = torch.tensor([[0.6, 0.2, 0.5, 0, 0, 0, 1]])
    T_R_destination = torch.tensor([[0.2, -0.1, 0.8, 0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4)]])
    result = gripper_pose_for_held_object(T_R_bin, T_R_gripper, T_R_destination)
    torch.testing.assert_close(result[:, :3], torch.tensor([[0.2, 0.0, 1.0]]))
    torch.testing.assert_close(result[:, 3:], T_R_destination[:, 3:])

    minimum_B = torch.tensor([[-0.15, -0.11, 0.0]])
    maximum_B = torch.tensor([[0.15, 0.11, 0.126]])
    candidates = bin_side_wall_gripper_poses(T_R_destination, minimum_B, maximum_B)
    for name, T_R_contact in candidates.items():
        t_B_G, q_B_G = subtract_frame_transforms(
            T_R_destination[:, :3], T_R_destination[:, 3:], T_R_contact[:, :3], T_R_contact[:, 3:]
        )
        t_B_pad = t_B_G + quat_apply(q_B_G, torch.tensor([[0.1446, 0.0, 0.0]]))
        assert 0.06 < t_B_pad[0, 0] < 0.08
        assert 0.09 < t_B_pad[0, 1].abs() < 0.11
        assert (t_B_pad[0, 1] > 0) == (name == "positive_y")
        torch.testing.assert_close(t_B_pad[:, 2], torch.tensor([0.101]))

    config = WorkbenchVLMGoalPolicyCfg(max_steps_per_goal=90, object_poses=["socket_set"])
    with patch("isaaclab_arena_vlm_agent.vlm_agent_policy.InferenceBackend"):
        policy = WorkbenchVLMGoalPolicy(config)
    stores = (
        policy._goal_history,
        policy._pose_targets,
        policy._pose_goals,
        policy._last_actions,
        policy.observation_adapter._previous_grasps,
    )
    for values in stores:
        values.update({0: T_R_start.clone(), 1: T_R_goal.clone()})
    policy.reset(torch.tensor([0]))
    for values in stores:
        assert set(values) == {1}
    policy.reset()
    assert all(not values for values in stores)
    policy.close()
    return True


def test_workbench_vlm_pose_tracking():
    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    assert run_function_with_persistent_simulation_app(_test_workbench_vlm_pose_tracking)


@pytest.mark.parametrize("task_path", _TASK_PATHS, ids=[path.stem for path in _TASK_PATHS])
def test_workbench_task_and_recipe_load(task_path):
    spec = ArenaEnvGraphSpec.from_yaml(task_path)
    recipe_name = task_path.stem.removeprefix("task_")
    recipe = load_object_control_recipe(_ENVIRONMENT_DIR / "object_control" / f"{recipe_name}.jsonl")

    subtask_indices = {command.index for command in recipe.commands if isinstance(command, (MarkCommand, CheckCommand))}
    assert subtask_indices == set(range(len(spec.task.subtasks)))

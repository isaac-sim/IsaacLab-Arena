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
    from isaaclab_arena_vlm_agent.grasp_point_adapter import GraspPointCfg, RevolutePointMotionCfg
    from isaaclab_arena_vlm_agent.vlm_agent_policy import ObservationModality, VLMGoalPolicyCfg

    paths = sorted((_ENVIRONMENT_DIR / "experiment_configs").glob("*_vlm_goal_experiment.yaml"))
    assert len(paths) == 5
    contact_point_tasks = {"task_close_toolcase.yaml", "task_full_cleanup.yaml"}
    covered_tasks = set()
    for path in paths:
        experiment = load_arena_experiment_from_config_file(path, device="cpu")
        assert len(experiment.runs) == 1
        run = next(iter(experiment.runs.values()))
        assert isinstance(run.policy, VLMGoalPolicyCfg)
        task_name = Path(run.environment.env_spec_path).name
        assert type(run.policy) is VLMGoalPolicyCfg
        if task_name in contact_point_tasks:
            point = run.policy.grasp_points["lid_outer_edge"]
            assert isinstance(point, GraspPointCfg)
            assert isinstance(point.revolute_motion, RevolutePointMotionCfg)
        else:
            assert not run.policy.grasp_points
        assert run.policy.vlm_embodiment_adapter == "droid"
        assert run.policy.observation_modality == ObservationModality.image_state
        assert run.environment.enable_cameras
        assert run.environment_builder.num_envs == 1
        assert run.rollout_limit.num_episodes == 5
        spec = ArenaEnvGraphSpec.from_yaml(run.environment.env_spec_path)
        if task_name == "task_full_cleanup.yaml":
            assert spec.task.composition.value == "sequential"
            assert len(spec.task.subtasks) == 4
        else:
            assert spec.task.composition.value == "atomic"
        assert spec.embodiment.registry_name == "droid_abs_ik"
        assert spec.embodiment.params["concatenate_observation_terms"] is False
        scene_keys = {
            spec.background.id,
            *[obj.id for obj in spec.objects],
            *[ref.id for ref in spec.object_references],
        }
        assert run.policy.object_poses and set(run.policy.object_poses) <= scene_keys
        assert all(point.scene_key in scene_keys for point in run.policy.grasp_points.values())
        assert run.policy.system_prompt_files
        covered_tasks.add(task_name)
    assert covered_tasks == {
        "task_bin_shelf.yaml",
        "task_close_drawer.yaml",
        "task_close_toolcase.yaml",
        "task_turn_off_scope.yaml",
        "task_full_cleanup.yaml",
    }
    return True


def test_workbench_vlm_experiment_configs():
    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    assert run_function_with_persistent_simulation_app(_test_workbench_vlm_experiment_configs)


def _test_workbench_vlm_contact_observations(simulation_app):
    import torch
    from types import SimpleNamespace
    from unittest.mock import Mock

    from isaaclab_arena.evaluation.arena_experiment_config_loader import load_arena_experiment_from_config_file
    from isaaclab_arena_vlm_agent.grasp_point_adapter import GraspPointObservationAdapter
    from isaaclab_arena_vlm_agent.observation_adapter import VLMObservationAdapter

    half_sqrt_two = 2**-0.5
    q_W_F = [0.0, 0.0, half_sqrt_two, half_sqrt_two]
    poses = {
        "robot": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [10.0, 0.0, 0.0, *q_W_F]]),
        "socket_set_lid": torch.tensor([[20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 2.0, 3.0, *q_W_F]]),
    }
    world = SimpleNamespace(
        get_pose_w=Mock(side_effect=lambda name: poses[name]),
        get_body_pose_w=Mock(
            return_value=torch.tensor([[20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 2.0, 3.0, *q_W_F]])
        ),
        get_aabb_in_local_frame=Mock(
            return_value=SimpleNamespace(
                center=torch.zeros(2, 3),
                min_point=torch.tensor([[0.0, -1.0, 0.0], [0.0, -0.1, 0.0]]),
                max_point=torch.tensor([[0.0, 1.0, 2.0], [0.0, 0.1, 0.2]]),
            )
        ),
    )
    experiment = load_arena_experiment_from_config_file(
        _ENVIRONMENT_DIR / "experiment_configs/close_toolcase_vlm_goal_experiment.yaml", device="cpu"
    )
    inner = Mock(spec=VLMObservationAdapter)
    inner.extract_vlm_input_from_observation.return_value = ({}, {})
    adapter = GraspPointObservationAdapter(inner, experiment.runs["close_toolcase"].policy.grasp_points)
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            arena_world=world,
            action_manager=SimpleNamespace(
                active_terms=["arm"], get_term=lambda _: SimpleNamespace(cfg=SimpleNamespace(asset_name="robot"))
            ),
        )
    )

    _, values = adapter.extract_vlm_input_from_observation(env, {}, env_id=1)
    geometry = values["simulator_ground_truth_assistance"]["grasp_points"]["lid_outer_edge"]["points_in_robot_roots"][
        "robot"
    ]

    assert geometry["position_root_m"] == pytest.approx([2.0, 8.912, 3.204])
    assert geometry["outward_normal_root"] == pytest.approx([0.0, 0.0, 1.0])
    assert geometry["motion_tangent_root"] == pytest.approx([0.0, -0.731894, -0.681419])
    world.get_body_pose_w.assert_called_once_with("socket_set", "Case_base")
    world.get_aabb_in_local_frame.assert_called_once_with("socket_set_lid")
    return True


def test_workbench_vlm_contact_observations():
    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    assert run_function_with_persistent_simulation_app(_test_workbench_vlm_contact_observations)


@pytest.mark.parametrize("task_path", _TASK_PATHS, ids=[path.stem for path in _TASK_PATHS])
def test_workbench_task_and_recipe_load(task_path):
    spec = ArenaEnvGraphSpec.from_yaml(task_path)
    recipe_name = task_path.stem.removeprefix("task_")
    recipe = load_object_control_recipe(_ENVIRONMENT_DIR / "object_control" / f"{recipe_name}.jsonl")

    subtask_indices = {command.index for command in recipe.commands if isinstance(command, (MarkCommand, CheckCommand))}
    assert subtask_indices == set(range(len(spec.task.subtasks)))

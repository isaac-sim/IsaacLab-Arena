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


def _test_workbench_vlm_contact_observations(simulation_app):
    import torch
    from types import SimpleNamespace
    from unittest.mock import patch

    from isaaclab_arena_environments.workbench_cleanup.vlm_policy import WorkbenchObservationAdapter
    from isaaclab_arena_vlm_agent.object_pose_adapter import ObjectPoseObservationAdapter

    button = SimpleNamespace(
        joint_names=["joint_button_01_joint_00"],
        data=SimpleNamespace(joint_pos=SimpleNamespace(torch=torch.tensor([[0.0], [-0.00188]]))),
    )

    class Scene(dict):
        articulations = {"oscilloscope": button}

    scene = Scene(oscilloscope=button)
    env = SimpleNamespace(unwrapped=SimpleNamespace(scene=scene))
    adapter = WorkbenchObservationAdapter(["oscilloscope", "oscilloscope_power_button", "tabletop"])
    with patch.object(ObjectPoseObservationAdapter, "extract_vlm_input_from_observation", return_value=({}, {})):
        _, state = adapter.extract_vlm_input_from_observation(env, {}, 1)
    assistance = state["simulator_ground_truth_assistance"]
    assert set(assistance) == {"joint_positions_m_or_rad"}
    assert set(assistance["joint_positions_m_or_rad"]) == {"oscilloscope"}
    travel = assistance["joint_positions_m_or_rad"]["oscilloscope"]["joint_button_01_joint_00"]
    assert travel == pytest.approx(-0.00188)
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

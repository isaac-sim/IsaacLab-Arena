# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate the workbench cleanup scene and tasks."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec

_ENVIRONMENT_DIR = Path(__file__).resolve().parents[2] / "isaaclab_arena_environments/workbench_cleanup"


def _test_workbench_task_definitions_build(simulation_app) -> bool:
    from isaaclab_arena.environment_spec.arena_env_graph_conversion_utils import build_arena_env_from_graph_spec

    for path in sorted(_ENVIRONMENT_DIR.glob("task_*.yaml")):
        environment = build_arena_env_from_graph_spec(ArenaEnvGraphSpec.from_yaml(path))
        assert environment.name
    return True


def test_workbench_task_definitions_build():
    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    assert run_function_with_persistent_simulation_app(_test_workbench_task_definitions_build)


def _test_workbench_processed_articulations(simulation_app):
    from pxr import Usd, UsdPhysics

    from isaaclab_arena_environments.workbench_cleanup.assets import (
        WorkbenchCleanupOscilloscope,
        WorkbenchCleanupSocketSet,
        WorkbenchCleanupWorkbench,
    )

    stage = Usd.Stage.Open(WorkbenchCleanupWorkbench.processor.process())
    cabinet = stage.GetPrimAtPath("/Asset/Workbench")
    articulation_roots = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
    assert articulation_roots == [cabinet]
    movable_joints = {
        prim.GetName()
        for prim in stage.Traverse()
        if prim.IsA(UsdPhysics.Joint) and not prim.IsA(UsdPhysics.FixedJoint)
    }
    assert movable_joints == {
        "joint_drawersA_01_joint_00",
        "joint_drawersB_01_joint_00",
        "joint_drawersC_01_joint_00",
        "joint_drawersD_01_joint_00",
        "joint_door_01_joint_00",
    }
    for path in (
        "/Asset/Workbench/Geometry/table_01_obj_00/table_01_mesh_00/Tabletop",
        "/Asset/Workbench/Geometry/table_01_obj_00/table_01_mesh_00/Structure/OverheadShelf",
    ):
        assert stage.GetPrimAtPath(path).HasAPI(UsdPhysics.RigidBodyAPI)

    for asset_type, moving_joint in (
        (WorkbenchCleanupSocketSet, "Physics/Joints/case_lid"),
        (WorkbenchCleanupOscilloscope, "Joints/joint_button_01_joint_00"),
    ):
        stage = Usd.Stage.Open(asset_type.processor.process())
        root = stage.GetDefaultPrim()
        articulation_roots = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
        assert articulation_roots == [root]
        movable_joints = [
            prim.GetPath().MakeRelativePath(root.GetPath()).pathString
            for prim in stage.Traverse()
            if prim.IsA(UsdPhysics.Joint) and not prim.IsA(UsdPhysics.FixedJoint)
        ]
        assert movable_joints == [moving_joint]
        if asset_type is WorkbenchCleanupSocketSet:
            body_names = {prim.GetName() for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)}
            assert body_names == {"Case_base", "Case_lid"}
            lid = stage.GetPrimAtPath(root.GetPath().AppendPath(moving_joint))
            drive = UsdPhysics.DriveAPI(lid, "angular")
            assert drive.GetStiffnessAttr().Get() == 0, "The lid must not be driven back open during robot contact"
            assert drive.GetMaxForceAttr().Get() == pytest.approx(0.2)
    return True


def test_workbench_processed_articulations():
    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    assert run_function_with_persistent_simulation_app(_test_workbench_processed_articulations)


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


def _test_joint_triggered_material_toggle(simulation_app):
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.managers import EventTermCfg
    from pxr import Usd, UsdGeom, UsdShade

    from isaaclab_arena.terms.events import JointTriggeredMaterialToggle

    stage = Usd.Stage.CreateInMemory()
    roots = [f"/World/envs/env_{index}/device" for index in range(2)]
    for root in roots:
        UsdGeom.Subset.Define(stage, f"{root}/Visuals/Indicator")
        UsdShade.Material.Define(stage, f"{root}/Materials/Default")
        UsdShade.Material.Define(stage, f"{root}/Materials/Alternate")

    class Scene(dict):
        env_prim_paths = [root.rsplit("/", 1)[0] for root in roots]
        env_regex_ns = "/World/envs/env_[^/]+"

    q = torch.zeros(2, 1)
    asset = SimpleNamespace(
        cfg=SimpleNamespace(prim_path="/World/envs/env_[^/]+/device"),
        data=SimpleNamespace(
            joint_names=["power"],
            joint_pos=SimpleNamespace(torch=q),
            joint_pos_limits=SimpleNamespace(torch=torch.tensor([[[-0.00235, 0]]] * 2)),
        ),
    )
    env = SimpleNamespace(
        scene=Scene(device=asset),
        sim=SimpleNamespace(is_rendering=False),
        num_envs=2,
        device="cpu",
    )
    params = {
        "asset_name": "device",
        "joint_name": "power",
        "target_prim_path": "Visuals/Indicator",
        "default_material_path": "Materials/Default",
        "alternate_material_path": "Materials/Alternate",
    }
    with sim_utils.use_stage(stage):
        event = JointTriggeredMaterialToggle(
            EventTermCfg(func=JointTriggeredMaterialToggle, mode="interval", params=params), env
        )
    q[0] = -0.00235
    event(env, None, **params)
    assert event.alternate_active.tolist() == [True, False]
    event(env, None, **params)
    assert event.alternate_active.tolist() == [True, False]
    q[0] = 0
    event(env, None, **params)
    q[:] = -0.00235
    event(env, None, **params)
    assert event.alternate_active.tolist() == [False, True]
    event.reset(torch.tensor([0]))
    assert event.alternate_active.tolist() == [False, True]
    event.reset(slice(None))
    assert event.alternate_active.tolist() == [False, False]
    return True


def test_joint_triggered_material_toggle():
    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    assert run_function_with_persistent_simulation_app(_test_joint_triggered_material_toggle)


@pytest.mark.parametrize(
    "filename,task_kind",
    [
        ("task_bin_shelf.yaml", "PickAndPlaceTask"),
        ("task_turn_off_scope.yaml", "PressButtonTask"),
        ("task_close_toolcase.yaml", "CloseDoorTask"),
        ("task_close_drawer.yaml", "CloseDoorTask"),
    ],
)
def test_workbench_task_uses_shared_scene(filename, task_kind):
    spec = ArenaEnvGraphSpec.from_yaml(_ENVIRONMENT_DIR / filename)

    assert spec.background.registry_name == "workbench_cleanup_workbench"
    assert "usd_path" not in spec.background.params
    assert spec.background.params["collision_mode"] == "mesh"
    assert spec.task.subtasks[0].kind == task_kind
    socket_set = next(obj for obj in spec.objects if obj.id == "socket_set")
    assert socket_set.registry_name == "workbench_cleanup_socket_set"
    for obj in spec.objects:
        if obj.id != "light":
            assert obj.registry_name == f"workbench_cleanup_{obj.id}"
            assert "usd_path" not in obj.params
    object_ids = {obj.id for obj in spec.objects}
    assert "slotted_screwdriver" in object_ids
    assert "small_wrench" in object_ids
    small_wrench = next(obj for obj in spec.objects if obj.id == "small_wrench")
    assert small_wrench.params["scale"] == [0.75, 0.75, 0.75]
    clutter_relations = {
        relation.subject: relation.reference for relation in spec.relations if relation.kind == "clutter_on"
    }
    assert clutter_relations == {
        "slotted_screwdriver": "top_drawer_body",
        "small_wrench": "top_drawer_body",
    }
    references = {reference.id: reference for reference in spec.object_references}
    assert references["top_drawer_body"].parent_id == "top_drawer"
    assert references["top_drawer"].params["initial_joint_positions"] == {"joint_drawersA_01_joint_00": 0.251816005}


def test_workbench_full_cleanup_is_sequential():
    spec = ArenaEnvGraphSpec.from_yaml(_ENVIRONMENT_DIR / "task_full_cleanup.yaml")

    assert spec.task.composition.value == "sequential"
    assert [subtask.kind for subtask in spec.task.subtasks] == [
        "CloseDoorTask",
        "CloseDoorTask",
        "PickAndPlaceTask",
        "PressButtonTask",
    ]

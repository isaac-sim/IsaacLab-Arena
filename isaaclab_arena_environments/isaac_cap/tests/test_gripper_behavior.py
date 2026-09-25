# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for CAP gear-mesh gripper behavior."""

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

pytestmark = pytest.mark.isaac_cap


def _make_world():
    import math
    import torch
    from types import SimpleNamespace

    from isaaclab_arena.environments.arena_world import ArenaWorld

    half_sqrt_two = math.sqrt(0.5)
    articulation_data = SimpleNamespace(
        joint_names=["panda_finger_joint1", "panda_finger_joint2", "left_driver_joint"],
        joint_pos=SimpleNamespace(torch=torch.tensor([[0.01, 0.02, 0.0], [0.04, 0.04, 0.8]])),
        body_names=["robotiq_base"],
        body_link_pose_w=SimpleNamespace(
            torch=torch.tensor([
                [[0.2, 0.1, 0.3, 0.0, 0.0, half_sqrt_two, half_sqrt_two]],
                [[0.4, 0.2, 0.5, 0.0, 0.0, half_sqrt_two, half_sqrt_two]],
            ])
        ),
    )
    sensor_data = SimpleNamespace(
        target_frame_names=["end_effector", "tool_leftfinger", "tool_rightfinger"],
        target_pos_w=SimpleNamespace(
            torch=torch.tensor([
                [[0.1, 0.0, 0.0], [0.1, 0.02, 0.0], [0.1, -0.02, 0.0]],
                [[0.3, 0.0, 0.0], [0.3, 0.05, 0.0], [0.3, -0.05, 0.0]],
            ])
        ),
    )
    object_poses = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    scene = SimpleNamespace(
        num_envs=2,
        articulations={"robot": SimpleNamespace(data=articulation_data)},
        rigid_objects={
            "object": SimpleNamespace(data=SimpleNamespace(root_pose_w=SimpleNamespace(torch=object_poses)))
        },
        sensors={"ee_frame": SimpleNamespace(data=sensor_data)},
        extras={},
    )
    return ArenaWorld(scene)


def _test_gear_success_requires_jaw_release_and_samples_position_once(_simulation_app) -> bool:
    import torch
    from types import SimpleNamespace

    from isaaclab_arena_environments.isaac_cap.gear_insertion_v2.task.terminations import gear_mesh_success

    class TestGripper:
        def __init__(self):
            self.opening_width_m = torch.tensor([0.05])
            self.position_calls = 0

        def get_opening_width_m(self, _world):
            return self.opening_width_m

        def get_position_w(self, _world):
            self.position_calls += 1
            return torch.tensor([[0.1, 0.0, 0.0]])

    board = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=torch.tensor([[0.0, -0.01]]),
            root_pos_w=torch.zeros((1, 3)),
            root_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        ),
        set_joint_velocity_target=lambda _target, joint_ids: None,
    )
    gear_data = SimpleNamespace(
        root_pos_w=torch.zeros((1, 3)),
        root_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        root_com_vel_w=torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
    )
    success = object.__new__(gear_mesh_success)
    success.board = board
    success.gears = (SimpleNamespace(data=gear_data), SimpleNamespace(data=gear_data))
    success.pinion_joint = 0
    success.button_joint = 1
    success.latched = torch.tensor([False])
    success.seated_seen = torch.tensor([False])
    success.started_after_seating = torch.tensor([False])
    success.success_steps = torch.zeros(1, dtype=torch.int32)
    success.required_steps = 1
    success.spin_window_steps = 1
    success.spin_history = torch.zeros((1, 1, 2))
    success.spin_samples_seen = torch.zeros(1, dtype=torch.int32)
    success.spin_history_index = 0
    env = SimpleNamespace(num_envs=1, device="cpu", arena_world=SimpleNamespace())
    gripper = TestGripper()
    params = {
        "board_asset_cfg": SimpleNamespace(),
        "gear_asset_cfgs": (SimpleNamespace(), SimpleNamespace()),
        "gripper": gripper,
        "grasp_width_m": 0.055,
        "release_clearance_m": 0.005,
        "target_offsets_xyz": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        "gear_teeth": (20, 20),
    }

    assert not success(env, **params).item()
    assert gripper.position_calls == 1

    gripper.opening_width_m[:] = 0.061
    assert success(env, **params).item()
    assert gripper.position_calls == 2
    return True


def test_gear_success_requires_jaw_release_and_samples_position_once() -> None:
    assert run_function_with_persistent_simulation_app(
        _test_gear_success_requires_jaw_release_and_samples_position_once
    )


def _test_gear_mesh_reset_event_clears_legacy_state(_simulation_app) -> bool:
    import torch
    from types import SimpleNamespace

    from isaaclab.managers import SceneEntityCfg, TerminationTermCfg

    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena_environments.isaac_cap.gear_insertion_v2.task.terminations import (
        gear_mesh_success,
        reset_gear_mesh_state,
    )

    for objective_name in ("gear_mesh", "subtask_0/gear_mesh"):
        env = SimpleNamespace(
            num_envs=2,
            device="cpu",
            step_dt=0.1,
            scene={
                "board": SimpleNamespace(data=SimpleNamespace(joint_names=["pinion_joint", "button_joint"])),
                "gear": SimpleNamespace(),
            },
        )
        success_cfg = TerminationTermCfg(
            func=gear_mesh_success,
            params={
                "board_asset_cfg": SceneEntityCfg("board"),
                "gear_asset_cfgs": [SceneEntityCfg("gear")],
                "gripper": object(),
                "hold_time_s": 1.0,
                "spin_window_s": 0.5,
            },
        )
        objective = ProgressObjective(name=objective_name, predicate_sequence=[success_cfg])
        tracker = ProgressTracker([objective], num_envs=env.num_envs, device=env.device, env=env)
        env.progress_tracker = tracker
        success = tracker.get_predicate(objective_name)
        success.latched[:] = True
        success.seated_seen[:] = True
        success.started_after_seating[:] = True
        success.success_steps[:] = 3
        success.spin_history[:] = 4.0
        success.spin_samples_seen[:] = success.spin_window_steps
        success.spin_history_index = 2

        # The task-local event clears legacy state; tracker.reset() only clears progress.
        reset_gear_mesh_state(env, env_ids=torch.tensor([0]))
        tracker.reset([0])
        assert success.latched.tolist() == [False, True]
        assert success.seated_seen.tolist() == [False, True]
        assert success.started_after_seating.tolist() == [False, True]
        assert success.success_steps.tolist() == [0, 3]
        assert not success.spin_history[:, 0].any()
        assert (success.spin_history[:, 1] == 4.0).all()
        assert success.spin_samples_seen.tolist() == [0, success.spin_window_steps]
        assert success.spin_history_index == 2

        reset_gear_mesh_state(env)
        assert not success.latched.any()
        assert not success.seated_seen.any()
        assert not success.started_after_seating.any()
        assert not success.success_steps.any()
        assert not success.spin_history.any()
        assert not success.spin_samples_seen.any()
        assert success.spin_history_index == 0
    return True


def test_gear_mesh_reset_event_clears_legacy_state() -> None:
    assert run_function_with_persistent_simulation_app(_test_gear_mesh_reset_event_clears_legacy_state)


def _test_gear_task_configures_release_checks_for_the_embodiment_gripper(_simulation_app) -> bool:
    from types import SimpleNamespace

    from isaaclab_arena.tasks.predicates.gripper import gripper_released
    from isaaclab_arena_environments.isaac_cap.gear_insertion_v2.embodiment import IndustrialFr3Robotiq2f85Embodiment
    from isaaclab_arena_environments.isaac_cap.gear_insertion_v2.task.task import GearMeshTaskV2
    from isaaclab_arena_environments.isaac_cap.gear_insertion_v2.task.terminations import reset_gear_mesh_state

    task = GearMeshTaskV2(
        board=SimpleNamespace(name="board"),
        gear=SimpleNamespace(name="gear"),
        grasp_width_m=0.035,
        release_clearance_m=0.004,
    )
    params = task.get_termination_cfg().success[0].predicate_sequence[0].params
    assert "gripper" not in params
    reset_event = task.get_events_cfg().reset_gear_mesh_state
    assert reset_event.mode == "reset"
    assert reset_event.func is reset_gear_mesh_state

    embodiment = IndustrialFr3Robotiq2f85Embodiment()
    task.configure_for_embodiment(embodiment)

    assert params["gripper"] is embodiment.gripper
    assert "robot_asset_cfg" not in params
    assert "tcp_body_name" not in params
    assert "tcp_offset_xyz" not in params
    assert params["grasp_width_m"] == pytest.approx(0.035)
    assert params["release_clearance_m"] == pytest.approx(0.004)
    released = gripper_released(
        SimpleNamespace(arena_world=_make_world()),
        gripper=params["gripper"],
        grasp_width_m=params["grasp_width_m"],
        release_clearance_m=params["release_clearance_m"],
    )
    assert released.tolist() == [True, False]
    return True


def test_gear_task_configures_release_checks_for_the_embodiment_gripper() -> None:
    assert run_function_with_persistent_simulation_app(
        _test_gear_task_configures_release_checks_for_the_embodiment_gripper
    )


def _test_gear_task_derives_default_grasp_width_from_teeth(_simulation_app) -> bool:
    from types import SimpleNamespace

    from isaaclab_arena_environments.isaac_cap.gear_insertion_v2.task.task import GearMeshTaskV2

    task = GearMeshTaskV2(board=SimpleNamespace(name="board"), gear=SimpleNamespace(name="gear"), gear_teeth=20)
    params = task.get_termination_cfg().success[0].predicate_sequence[0].params
    assert params["grasp_width_m"] == pytest.approx(0.055)

    task.set_gear_teeth(24)
    assert params["grasp_width_m"] == pytest.approx(0.065)

    override_task = GearMeshTaskV2(
        board=SimpleNamespace(name="board"),
        gear=SimpleNamespace(name="gear"),
        gear_teeth=20,
        grasp_width_m=0.04,
    )
    override_params = override_task.get_termination_cfg().success[0].predicate_sequence[0].params
    override_task.set_gear_teeth(24)
    assert override_params["grasp_width_m"] == pytest.approx(0.04)
    return True


def test_gear_task_derives_default_grasp_width_from_teeth() -> None:
    assert run_function_with_persistent_simulation_app(_test_gear_task_derives_default_grasp_width_from_teeth)

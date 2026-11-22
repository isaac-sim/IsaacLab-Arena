# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

if TYPE_CHECKING:
    import torch

SETTLE_STEPS = 30
ARM_HOLD_TOLERANCE_RAD = 0.02
FINGER_TOLERANCE_M = 0.002
ROBOT_POSITION = (0.2, -0.1, 0.05)
"""Env-local robot base position, away from the origin so placement is observable."""
ROOT_POSITION_TOLERANCE_M = 0.001
ROOT_WRITE_SHIFT_X_M = 0.1


def _test_yam_embodiment_backends(simulation_app) -> bool:
    """Check both backends spawn the AUTOLab USD, only PhysX disables gravity, and the cameras are exposed."""
    import math

    from isaaclab_arena.assets.registries import AssetRegistry, ensure_assets_registered
    from isaaclab_arena.embodiments.yam.yam import YAM_USD_PATH, YamAbsoluteJointPositionEmbodiment
    from isaaclab_arena.utils.physics_backend import PhysicsBackend

    ensure_assets_registered()
    assert AssetRegistry().get_component_by_name("yam_abs_joint_pos") is YamAbsoluteJointPositionEmbodiment
    assert YAM_USD_PATH.endswith("/Arena/assets/robot_library/yam/i2rt_yam_default.usda")

    camera_rig = YamAbsoluteJointPositionEmbodiment(enable_cameras=True).get_camera_cfg()
    assert camera_rig.camera_names() == ["wrist_camera", "top_camera"]
    assert camera_rig.wrist_camera.prim_path.endswith(
        "/Robot/Geometry/arm/link_1/link_2/link_3/link_4/link_5/link_6/wrist_camera"
    )
    assert camera_rig.top_camera.prim_path.endswith("/Robot/Geometry/arm/top_camera")
    for camera in (camera_rig.wrist_camera, camera_rig.top_camera):
        lens = camera.spawn
        assert lens.vertical_aperture / lens.horizontal_aperture == pytest.approx(camera.height / camera.width)
        vertical_fov_deg = math.degrees(2.0 * math.atan(lens.vertical_aperture / (2.0 * lens.focal_length)))
        assert vertical_fov_deg == pytest.approx(58.0)

    newton_embodiment = YamAbsoluteJointPositionEmbodiment()
    newton_embodiment.configure_physics_backend(PhysicsBackend.NEWTON)
    newton_spawn = newton_embodiment.get_scene_cfg().robot.spawn
    assert newton_spawn.usd_path == YAM_USD_PATH
    assert newton_spawn.rigid_props is None
    assert newton_embodiment.get_command_body_name() == "link_6"

    physx_embodiment = YamAbsoluteJointPositionEmbodiment()
    physx_embodiment.configure_physics_backend(PhysicsBackend.PHYSX)
    physx_robot = physx_embodiment.get_scene_cfg().robot
    assert physx_robot.spawn.usd_path == YAM_USD_PATH
    assert physx_robot.spawn.rigid_props.disable_gravity is True
    shoulder = physx_robot.actuators["arm_joints_1_3"]
    assert (shoulder.stiffness, shoulder.damping, shoulder.joint_effort_limit) == (40.0, 2.5, 28.0)
    assert physx_robot.actuators["gripper"].joint_effort_limit == 40.0
    return True


def test_yam_embodiment_backends():
    assert run_function_with_persistent_simulation_app(_test_yam_embodiment_backends)


@contextmanager
def _yam_env(env_name: str, backend_name: str):
    """Yield a single-env scene holding only the YAM at ``ROBOT_POSITION``, simulated on the named physics backend."""
    import gymnasium as gym

    from isaaclab_arena.embodiments.yam.yam import YamAbsoluteJointPositionEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.physics_backend import PhysicsBackend
    from isaaclab_arena.utils.pose import Pose

    embodiment = YamAbsoluteJointPositionEmbodiment(initial_pose=Pose(position_xyz=ROBOT_POSITION))
    arena_env = IsaacLabArenaEnvironment(name=env_name, embodiment=embodiment, scene=Scene())
    if env_name in gym.registry:
        del gym.registry[env_name]
    builder_cfg = ArenaEnvBuilderCfg(num_envs=1, presets=PhysicsBackend(backend_name))
    env = ArenaEnvBuilder(arena_env, builder_cfg).make_registered()
    try:
        env.reset()
        yield env
    finally:
        env.close()
        if env_name in gym.registry:
            del gym.registry[env_name]


def _hold_default_arm_pose(env, gripper_command: float) -> torch.Tensor:
    """Command the default arm pose and a gripper closure in [0, 1], step until settled, and return joint positions."""
    import torch

    import warp as wp

    from isaaclab_arena.embodiments.yam.yam import ARM_JOINT_NAMES

    robot = env.unwrapped.scene["robot"]
    arm_joint_ids, _ = robot.find_joints(ARM_JOINT_NAMES, preserve_order=True)
    action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    action[:, :6] = wp.to_torch(robot.data.default_joint_pos)[:, arm_joint_ids]
    action[:, 6] = gripper_command
    for _ in range(SETTLE_STEPS):
        env.step(action)
    return wp.to_torch(robot.data.joint_pos)[0].clone()


def _test_yam_holds_arm_and_mirrors_fingers(simulation_app, backend_name: str) -> bool:
    """The robot should stay at its placed pose, hold the default arm pose and mirror every gripper command."""
    import torch

    import warp as wp

    from isaaclab_arena.embodiments.yam.yam import (
        ARM_JOINT_NAMES,
        GRIPPER_CLOSED_POSITION,
        GRIPPER_JOINT_NAME,
        GRIPPER_OPEN_POSITION,
        PASSIVE_GRIPPER_JOINT_NAME,
    )

    with _yam_env(f"yam_{backend_name}_hold_test", backend_name) as env:
        robot = env.unwrapped.scene["robot"]
        arm_joint_ids, _ = robot.find_joints(ARM_JOINT_NAMES, preserve_order=True)
        (driven_id,), _ = robot.find_joints([GRIPPER_JOINT_NAME])
        (passive_id,), _ = robot.find_joints([PASSIVE_GRIPPER_JOINT_NAME])
        default_arm_joint_pos = wp.to_torch(robot.data.default_joint_pos)[0, arm_joint_ids]
        with torch.inference_mode():
            for gripper_command in (0.5, 1.0, 0.0):
                joint_pos = _hold_default_arm_pose(env, gripper_command)
                arm_error = torch.max(torch.abs(joint_pos[arm_joint_ids] - default_arm_joint_pos)).item()
                expected_finger_pos = GRIPPER_OPEN_POSITION + gripper_command * (
                    GRIPPER_CLOSED_POSITION - GRIPPER_OPEN_POSITION
                )
                driven_pos = joint_pos[driven_id].item()
                passive_pos = joint_pos[passive_id].item()
                print(
                    f"[{backend_name}] gripper={gripper_command}: arm error {arm_error:.4f} rad, driven finger"
                    f" {driven_pos:.4f} m, passive finger {passive_pos:.4f} m, expected {expected_finger_pos:.4f} m"
                )
                assert arm_error < ARM_HOLD_TOLERANCE_RAD, f"Arm drifted {arm_error:.4f} rad from its default pose."
                assert abs(driven_pos - expected_finger_pos) < FINGER_TOLERANCE_M
                assert (
                    abs(passive_pos + driven_pos) < FINGER_TOLERANCE_M
                ), "The passive finger must mirror the driven one."
                t_E_R = wp.to_torch(robot.data.root_pos_w)[0] - env.unwrapped.scene.env_origins[0]
                root_error = torch.norm(t_E_R - torch.tensor(ROBOT_POSITION, device=t_E_R.device)).item()
                print(f"[{backend_name}] root position {t_E_R.tolist()}, error {root_error:.4f} m")
                assert root_error < ROOT_POSITION_TOLERANCE_M, f"Robot base is {root_error:.4f} m from its placed pose."

            # Placement resets write the root pose; the fixed base must follow it.
            T_W_R = wp.to_torch(robot.data.root_pose_w).clone()
            T_W_R[:, 0] += ROOT_WRITE_SHIFT_X_M
            robot.write_root_pose_to_sim_index(root_pose=T_W_R)
            _hold_default_arm_pose(env, 0.0)
            shift_x = (wp.to_torch(robot.data.root_pos_w)[0, 0] - T_W_R[0, 0] + ROOT_WRITE_SHIFT_X_M).item()
            print(f"[{backend_name}] root moved {shift_x:.4f} m after a {ROOT_WRITE_SHIFT_X_M} m root write")
            assert (
                abs(shift_x - ROOT_WRITE_SHIFT_X_M) < ROOT_POSITION_TOLERANCE_M
            ), "The base ignored a root pose write."
    return True


@pytest.mark.parametrize("backend_name", ["physx", "newton"])
def test_yam_holds_arm_and_mirrors_fingers(backend_name: str):
    assert run_function_with_persistent_simulation_app(
        _test_yam_holds_arm_and_mirrors_fingers, backend_name=backend_name
    )

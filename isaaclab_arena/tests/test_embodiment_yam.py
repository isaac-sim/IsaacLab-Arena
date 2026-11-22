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
RIG_POSITION = (0.2, -0.1, 0.05)
"""Env-local rig position, away from the origin so placement is observable."""
ROOT_POSITION_TOLERANCE_M = 0.001
ROOT_WRITE_SHIFT_X_M = 0.1
RESET_TEST_JOINT1_OFFSET_RAD = 0.3


def _test_yam_bimanual_embodiment_backends(simulation_app) -> bool:
    """Check both arms spawn the AUTOLab USD, only PhysX overrides them, and the three cameras are exposed."""
    import math

    from isaaclab_arena.assets.registries import AssetRegistry, ensure_assets_registered
    from isaaclab_arena.embodiments.common.arm_mode import ArmMode
    from isaaclab_arena.embodiments.yam.yam import (
        LEFT_ROBOT_KEY,
        RIGHT_ROBOT_KEY,
        YAM_USD_PATH,
        YamBimanualAbsoluteJointPositionEmbodiment,
    )
    from isaaclab_arena.utils.physics_backend import PhysicsBackend

    ensure_assets_registered()
    registered = AssetRegistry().get_component_by_name("yam_bimanual_abs_joint_pos")
    assert registered is YamBimanualAbsoluteJointPositionEmbodiment
    assert YAM_USD_PATH.endswith("/Arena/assets/robot_library/yam/i2rt_yam_default.usda")

    camera_rig = YamBimanualAbsoluteJointPositionEmbodiment(enable_cameras=True).get_camera_cfg()
    assert camera_rig.camera_names() == ["left_wrist_camera", "right_wrist_camera", "top_camera"]
    link_six = "Geometry/arm/link_1/link_2/link_3/link_4/link_5/link_6"
    assert camera_rig.left_wrist_camera.prim_path.endswith(f"/LeftRobot/{link_six}/left_wrist_camera")
    assert camera_rig.right_wrist_camera.prim_path.endswith(f"/RightRobot/{link_six}/right_wrist_camera")
    assert camera_rig.top_camera.prim_path.endswith("/LeftRobot/Geometry/arm/top_camera")
    # The top camera rides on the left base, 0.31 m left of the rig midpoint it is centred over.
    assert camera_rig.top_camera.offset.pos == pytest.approx((0.335, -0.31, 0.93732053))
    for name in camera_rig.camera_names():
        camera = getattr(camera_rig, name)
        lens = camera.spawn
        assert lens.vertical_aperture / lens.horizontal_aperture == pytest.approx(camera.height / camera.width)
        vertical_fov_deg = math.degrees(2.0 * math.atan(lens.vertical_aperture / (2.0 * lens.focal_length)))
        assert vertical_fov_deg == pytest.approx(58.0)

    newton_embodiment = YamBimanualAbsoluteJointPositionEmbodiment()
    newton_embodiment.configure_physics_backend(PhysicsBackend.NEWTON)
    newton_scene = newton_embodiment.get_scene_cfg()
    physx_embodiment = YamBimanualAbsoluteJointPositionEmbodiment()
    physx_embodiment.configure_physics_backend(PhysicsBackend.PHYSX)
    physx_scene = physx_embodiment.get_scene_cfg()
    for name in (LEFT_ROBOT_KEY, RIGHT_ROBOT_KEY):
        newton_robot = getattr(newton_scene, name)
        assert newton_robot.spawn.usd_path == YAM_USD_PATH
        assert newton_robot.spawn.rigid_props is None
        physx_robot = getattr(physx_scene, name)
        assert physx_robot.spawn.usd_path == YAM_USD_PATH
        assert physx_robot.spawn.rigid_props.disable_gravity is True
        shoulder = physx_robot.actuators["arm_joints_1_3"]
        assert (shoulder.stiffness, shoulder.damping, shoulder.joint_effort_limit) == (40.0, 2.5, 28.0)
        assert physx_robot.actuators["gripper"].joint_effort_limit == 40.0
    assert newton_embodiment.get_scene_root_keys() == (LEFT_ROBOT_KEY, RIGHT_ROBOT_KEY)
    # Placement bounds cover both arms, whose bases sit 0.31 m either side of the rig midpoint.
    bbox = newton_embodiment.get_bounding_box()
    assert bbox.min_point[0, 1].item() < -0.31 and bbox.max_point[0, 1].item() > 0.31

    # Each arm has its own end-effector frame sensor and joint reset; LEFT or RIGHT selects one arm.
    assert newton_embodiment.get_ee_frame_transformer_names() == ["left_ee_frame", "right_ee_frame"]
    assert newton_scene.left_ee_frame is not None and newton_scene.right_ee_frame is not None
    assert newton_embodiment.get_ee_frame_name(ArmMode.LEFT) == "left_ee_frame"
    assert newton_embodiment.get_ee_frame_name(ArmMode.RIGHT) == "right_ee_frame"
    right_arm_embodiment = YamBimanualAbsoluteJointPositionEmbodiment(arm_mode=ArmMode.RIGHT)
    assert right_arm_embodiment.get_scene_key() == RIGHT_ROBOT_KEY
    assert newton_embodiment.get_scene_key() == LEFT_ROBOT_KEY
    events = newton_embodiment.get_events_cfg()
    assert events.reset_left_robot_joints.params["asset_cfg"].name == LEFT_ROBOT_KEY
    assert events.reset_right_robot_joints.params["asset_cfg"].name == RIGHT_ROBOT_KEY
    return True


def test_yam_bimanual_embodiment_backends():
    assert run_function_with_persistent_simulation_app(_test_yam_bimanual_embodiment_backends)


@contextmanager
def _yam_bimanual_env(env_name: str, backend_name: str):
    """Yield a single-env scene holding only the YAM pair at ``RIG_POSITION``, simulated on the named backend."""
    import gymnasium as gym

    from isaaclab_arena.embodiments.yam.yam import YamBimanualAbsoluteJointPositionEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.physics_backend import PhysicsBackend
    from isaaclab_arena.utils.pose import Pose

    embodiment = YamBimanualAbsoluteJointPositionEmbodiment()
    # set_initial_pose also adds the reset event that writes the rig pose back at every episode start.
    embodiment.set_initial_pose(Pose(position_xyz=RIG_POSITION))
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


def _hold_default_arm_poses(
    env, left_gripper_command: float, right_gripper_command: float, joint1_offset: float = 0.0
) -> dict[str, torch.Tensor]:
    """Command both arms' default poses, shifted by ``joint1_offset`` on joint1, and each gripper's closure in [0, 1].

    Steps until settled and returns each arm's joint positions.
    """
    import torch

    import warp as wp

    from isaaclab_arena.embodiments.yam.yam import ARM_JOINT_NAMES, LEFT_ROBOT_KEY, RIGHT_ROBOT_KEY

    action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    # Policy order: left arm (6), left gripper (1), right arm (6), right gripper (1).
    for offset, name, gripper_command in (
        (0, LEFT_ROBOT_KEY, left_gripper_command),
        (7, RIGHT_ROBOT_KEY, right_gripper_command),
    ):
        robot = env.unwrapped.scene[name]
        arm_joint_ids, _ = robot.find_joints(ARM_JOINT_NAMES, preserve_order=True)
        action[:, offset : offset + 6] = wp.to_torch(robot.data.default_joint_pos)[:, arm_joint_ids]
        action[:, offset] += joint1_offset
        action[:, offset + 6] = gripper_command
    for _ in range(SETTLE_STEPS):
        env.step(action)
    return {
        name: wp.to_torch(env.unwrapped.scene[name].data.joint_pos)[0].clone()
        for name in (LEFT_ROBOT_KEY, RIGHT_ROBOT_KEY)
    }


def _test_yam_bimanual_holds_arms_and_mirrors_fingers(simulation_app, backend_name: str) -> bool:
    """Both arms should stay at their mounts, hold their default poses, follow their own gripper commands and reset."""
    import math
    import torch

    import warp as wp

    from isaaclab_arena.embodiments.yam.yam import (
        ARM_JOINT_NAMES,
        END_EFFECTOR_BODY_NAME,
        GRASP_SITE_OFFSET_FROM_LINK_SIX,
        GRIPPER_CLOSED_POSITION,
        GRIPPER_JOINT_NAME,
        GRIPPER_OPEN_POSITION,
        LEFT_ROBOT_KEY,
        PASSIVE_GRIPPER_JOINT_NAME,
        RIGHT_ROBOT_KEY,
        t_M_B_BY_ARM,
    )

    with _yam_bimanual_env(f"yam_bimanual_{backend_name}_hold_test", backend_name) as env:
        scene = env.unwrapped.scene
        with torch.inference_mode():
            # Different commands per arm check that each gripper action drives its own arm.
            for left_command, right_command in ((0.5, 1.0), (1.0, 0.0), (0.0, 0.5)):
                joint_pos_by_arm = _hold_default_arm_poses(env, left_command, right_command)
                for name, command in ((LEFT_ROBOT_KEY, left_command), (RIGHT_ROBOT_KEY, right_command)):
                    robot = scene[name]
                    joint_pos = joint_pos_by_arm[name]
                    arm_joint_ids, _ = robot.find_joints(ARM_JOINT_NAMES, preserve_order=True)
                    (driven_id,), _ = robot.find_joints([GRIPPER_JOINT_NAME])
                    (passive_id,), _ = robot.find_joints([PASSIVE_GRIPPER_JOINT_NAME])
                    default_arm_joint_pos = wp.to_torch(robot.data.default_joint_pos)[0, arm_joint_ids]
                    arm_error = torch.max(torch.abs(joint_pos[arm_joint_ids] - default_arm_joint_pos)).item()
                    expected_finger_pos = GRIPPER_OPEN_POSITION + command * (
                        GRIPPER_CLOSED_POSITION - GRIPPER_OPEN_POSITION
                    )
                    driven_pos = joint_pos[driven_id].item()
                    passive_pos = joint_pos[passive_id].item()
                    t_E_B = wp.to_torch(robot.data.root_pos_w)[0] - scene.env_origins[0]
                    expected_t_E_B = torch.tensor(RIG_POSITION, device=t_E_B.device) + torch.tensor(
                        t_M_B_BY_ARM[name], device=t_E_B.device
                    )
                    base_error = torch.norm(t_E_B - expected_t_E_B).item()
                    print(
                        f"[{backend_name}] {name} gripper={command}: arm error {arm_error:.4f} rad, driven finger"
                        f" {driven_pos:.4f} m, passive finger {passive_pos:.4f} m, expected {expected_finger_pos:.4f}"
                        f" m, base error {base_error:.4f} m"
                    )
                    assert arm_error < ARM_HOLD_TOLERANCE_RAD, f"{name} drifted {arm_error:.4f} rad."
                    assert abs(driven_pos - expected_finger_pos) < FINGER_TOLERANCE_M
                    assert abs(passive_pos + driven_pos) < FINGER_TOLERANCE_M, "The passive finger must mirror."
                    assert base_error < ROOT_POSITION_TOLERANCE_M, f"{name} base is {base_error:.4f} m off its mount."

            # Placement resets write the root poses; each fixed base must follow its write.
            written_x = {}
            for name in (LEFT_ROBOT_KEY, RIGHT_ROBOT_KEY):
                robot = scene[name]
                T_W_B = wp.to_torch(robot.data.root_pose_w).clone()
                T_W_B[:, 0] += ROOT_WRITE_SHIFT_X_M
                robot.write_root_pose_to_sim_index(root_pose=T_W_B)
                written_x[name] = T_W_B[0, 0].item()
            _hold_default_arm_poses(env, 0.0, 0.0)
            for name, x in written_x.items():
                x_error = abs(wp.to_torch(scene[name].data.root_pos_w)[0, 0].item() - x)
                print(f"[{backend_name}] {name} base is {x_error:.4f} m from its written x")
                assert x_error < ROOT_POSITION_TOLERANCE_M, f"The {name} base ignored a root pose write."

            # Each end-effector frame tracks its arm's grasp site, a fixed distance from link_6.
            for name, ee_frame_name in ((LEFT_ROBOT_KEY, "left_ee_frame"), (RIGHT_ROBOT_KEY, "right_ee_frame")):
                robot = scene[name]
                (link_six_id,), _ = robot.find_bodies([END_EFFECTOR_BODY_NAME])
                t_W_link6 = wp.to_torch(robot.data.body_pos_w)[0, link_six_id]
                t_W_grasp = wp.to_torch(scene[ee_frame_name].data.target_pos_w)[0, 0]
                distance = torch.norm(t_W_grasp - t_W_link6).item()
                expected = math.dist((0.0, 0.0, 0.0), GRASP_SITE_OFFSET_FROM_LINK_SIX)
                print(f"[{backend_name}] {ee_frame_name} is {distance:.4f} m from link_6, expected {expected:.4f} m")
                assert abs(distance - expected) < ROOT_POSITION_TOLERANCE_M

            # A reset must restore the initial joint state and mounts after the arms move and the grippers close.
            _hold_default_arm_poses(env, 1.0, 1.0, joint1_offset=RESET_TEST_JOINT1_OFFSET_RAD)
            for name in (LEFT_ROBOT_KEY, RIGHT_ROBOT_KEY):
                joint_pos = wp.to_torch(scene[name].data.joint_pos)[0]
                default_joint_pos = wp.to_torch(scene[name].data.default_joint_pos)[0]
                moved = torch.max(torch.abs(joint_pos - default_joint_pos)).item()
                assert moved > RESET_TEST_JOINT1_OFFSET_RAD / 2, f"{name} did not move before the reset."
            env.reset()
            for name in (LEFT_ROBOT_KEY, RIGHT_ROBOT_KEY):
                robot = scene[name]
                joint_error = torch.abs(
                    wp.to_torch(robot.data.joint_pos)[0] - wp.to_torch(robot.data.default_joint_pos)[0]
                )
                arm_joint_ids, _ = robot.find_joints(ARM_JOINT_NAMES, preserve_order=True)
                finger_joint_ids, _ = robot.find_joints([GRIPPER_JOINT_NAME, PASSIVE_GRIPPER_JOINT_NAME])
                arm_error = torch.max(joint_error[arm_joint_ids]).item()
                # The open finger starts at its hard limit; the reset clamps it to the 0.95 soft limit, 1 mm short.
                finger_error = torch.max(joint_error[finger_joint_ids]).item()
                t_E_B = wp.to_torch(robot.data.root_pos_w)[0] - scene.env_origins[0]
                expected_t_E_B = torch.tensor(RIG_POSITION, device=t_E_B.device) + torch.tensor(
                    t_M_B_BY_ARM[name], device=t_E_B.device
                )
                base_error = torch.norm(t_E_B - expected_t_E_B).item()
                print(
                    f"[{backend_name}] {name} after reset: arm error {arm_error:.5f} rad, finger error"
                    f" {finger_error:.4f} m, base error {base_error:.4f} m"
                )
                assert arm_error < 1e-4, f"{name} kept {arm_error:.4f} rad of its previous arm pose."
                assert finger_error < FINGER_TOLERANCE_M, f"{name} kept {finger_error:.4f} m of its gripper closure."
                assert base_error < ROOT_POSITION_TOLERANCE_M, f"{name} base is {base_error:.4f} m off its mount."
    return True


@pytest.mark.parametrize("backend_name", ["physx", "newton"])
def test_yam_bimanual_holds_arms_and_mirrors_fingers(backend_name: str):
    assert run_function_with_persistent_simulation_app(
        _test_yam_bimanual_holds_arms_and_mirrors_fingers, backend_name=backend_name
    )

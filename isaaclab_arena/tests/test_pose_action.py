# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Generic relative IK conversion and DROID integration checks."""

import numpy as np
import torch
from types import SimpleNamespace

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_pose_action(simulation_app, offset, vector_scale):
    from isaaclab.utils.math import apply_delta_pose, combine_frame_transforms, matrix_from_quat

    from isaaclab_arena.policy.pose_action import pose_to_relative_ik_action

    current = torch.tensor([[0.2, 0.1, 0.3, 0, 0, 0, 1], [0.4, 0.2, 0.1, 0, 0, 0, 1]])
    cfg = SimpleNamespace(
        clip=None,
        scale=0.5,
        body_offset=None,
        controller=SimpleNamespace(command_type="pose", use_relative_mode=True),
    )
    if vector_scale:
        cfg.scale = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    if offset:
        cfg.body_offset = SimpleNamespace(pos=(0.131, 0, 0), rot=(0, 0, 2**-0.5, 2**-0.5))
    target = current.clone()
    target[:, :3] += torch.tensor([0.01, -0.02, 0.03])
    target[:, 3:] = torch.tensor([0.1, 0, 0, (1 - 0.1**2) ** 0.5])
    target[1, 3:] *= -1  # q and -q must produce equivalent rotations.
    arm_actions = pose_to_relative_ik_action(current, target, cfg)
    position, rotation = current[:, :3], current[:, 3:]
    expected_position, expected_rotation = target[:, :3], target[:, 3:]
    if offset:
        translation = current.new_tensor(cfg.body_offset.pos).expand(2, -1)
        quaternion = current.new_tensor(cfg.body_offset.rot).expand(2, -1)
        position, rotation = combine_frame_transforms(position, rotation, translation, quaternion)
        expected_position, expected_rotation = combine_frame_transforms(
            expected_position, expected_rotation, translation, quaternion
        )
    actual_position, actual_rotation = apply_delta_pose(position, rotation, arm_actions * torch.tensor(cfg.scale))
    torch.testing.assert_close(actual_position, expected_position)
    torch.testing.assert_close(matrix_from_quat(actual_rotation), matrix_from_quat(expected_rotation))
    with pytest.raises(AssertionError, match="unit XYZW"):
        pose_to_relative_ik_action(current, target * 2, cfg)
    cfg.scale = 0
    with pytest.raises(AssertionError, match="nonzero"):
        pose_to_relative_ik_action(current, target, cfg)
    cfg.controller.use_relative_mode = False
    with pytest.raises(AssertionError, match="relative pose IK"):
        pose_to_relative_ik_action(current, target, cfg)
    return True


@pytest.mark.parametrize("offset", [False, True])
@pytest.mark.parametrize("vector_scale", [False, True])
def test_pose_action(offset, vector_scale):
    assert run_function_with_persistent_simulation_app(_test_pose_action, offset=offset, vector_scale=vector_scale)


def _test_droid_runtime(simulation_app, cameras):
    import gymnasium as gym

    from isaaclab.utils.math import matrix_from_quat

    from isaaclab_arena.embodiments.droid.droid import DroidDifferentialIKEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.policy.pose_action import pose_to_relative_ik_action
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.camera_inputs import enable_camera_pose_updates, extract_camera_inputs
    from isaaclab_arena.utils.pose import Pose

    embodiment = DroidDifferentialIKEmbodiment(enable_cameras=cameras)
    embodiment.set_initial_pose(Pose(position_xyz=(0, 0, 1), rotation_xyzw=(0, 0, 2**-0.5, 2**-0.5)))
    camera_keys = ("external_camera_rgb", "wrist_camera_rgb")
    if cameras:
        enable_camera_pose_updates(embodiment.camera_config, camera_keys)
        for name in ("external_camera", "external_camera_2", "wrist_camera"):
            camera_cfg = getattr(embodiment.camera_config, name)
            camera_cfg.width, camera_cfg.height = 96, 64
    name = "droid_policy_utils_test"
    arena = IsaacLabArenaEnvironment(name=name, embodiment=embodiment, scene=Scene(assets=[]))
    env = ArenaEnvBuilder(arena, ArenaEnvBuilderCfg(num_envs=2)).make_registered()
    try:
        with torch.inference_mode():
            observation, _ = env.reset()
            world = env.unwrapped.arena_world
            measured = world.get_body_pose_in_root("robot", "base_link")
            target = measured.clone()
            target[:, 2] += 0.01
            closure = target.new_tensor([[0.0], [1.0]])
            arm = env.unwrapped.action_manager.get_term("arm_action")
            arm_action = pose_to_relative_ik_action(measured, target, arm.cfg)
            action = torch.cat((arm_action, closure), dim=-1)
            arm.process_actions(action[:, :6])
            torch.testing.assert_close(arm._ik_controller.ee_pos_des, target[:, :3])
            torch.testing.assert_close(
                matrix_from_quat(arm._ik_controller.ee_quat_des), matrix_from_quat(target[:, 3:])
            )
            gripper = env.unwrapped.action_manager.get_term("gripper_action")
            gripper.process_actions(action[:, 6:])
            torch.testing.assert_close(gripper.processed_actions[0], gripper._open_command)
            torch.testing.assert_close(gripper.processed_actions[1], gripper._close_command)
            observation, *_ = env.step(action)
            assert torch.isfinite(world.get_body_pose_in_root("robot", "base_link")).all()
            if cameras:
                inputs = extract_camera_inputs(
                    env, observation, 1, camera_keys, T_W_R=world.get_pose_w("robot"), image_max_edge=48
                )
                for key, camera in inputs.items():
                    assert camera.rgb.shape == (32, 48, 3)
                    assert camera.rgb.dtype == np.uint8
                    assert np.isfinite(camera.T_R_C).all()
                    assert np.linalg.norm(camera.T_R_C[3:]) == pytest.approx(1, abs=1e-5)
                    original_K = (
                        env.unwrapped.scene[key.removesuffix("_rgb")].data.intrinsic_matrices.torch[1].cpu().numpy()
                    )
                    np.testing.assert_allclose(camera.intrinsics[:2], original_K[:2] * 0.5)
    finally:
        env.close()
        del gym.registry[name]
    return True


def test_droid_runtime():
    assert run_function_with_persistent_simulation_app(_test_droid_runtime, cameras=False)


@pytest.mark.with_cameras
def test_droid_camera_runtime():
    assert run_function_with_persistent_simulation_app(_test_droid_runtime, enable_cameras=True, cameras=True)

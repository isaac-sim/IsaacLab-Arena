# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""DROID policy frame, controller, and camera input contracts."""

import numpy as np
import torch
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _stub_env():
    from isaaclab.utils.math import combine_frame_transforms

    from isaaclab_arena.embodiments.droid.actions import BinaryJointPositionZeroToOneAction

    # B is translated and rotated 90 degrees about world Z in the second environment.
    T_W_B = torch.tensor([[0, 0, 1, 0, 0, 0, 1], [2, 3, 4, 0, 0, 2**-0.5, 2**-0.5]])
    T_B_G = torch.tensor([[0.2, 0.1, 0.3, 0, 0, 0, 1], [0.4, 0.2, 0.1, 0, 0, 0, 1]])
    position, rotation = combine_frame_transforms(T_W_B[:, :3], T_W_B[:, 3:], T_B_G[:, :3], T_B_G[:, 3:])
    T_W_G = torch.cat((position, rotation), dim=-1)
    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_names=["panda_link0", "base_link"],
            root_pose_w=SimpleNamespace(torch=T_W_B),
            body_link_pose_w=SimpleNamespace(torch=torch.stack((T_W_B, T_W_G), dim=1)),
        )
    )
    cfg = SimpleNamespace(
        asset_name="robot",
        body_name="base_link",
        clip=None,
        scale=0.5,
        body_offset=None,
        controller=SimpleNamespace(command_type="pose", use_relative_mode=True),
    )
    arm = SimpleNamespace(cfg=cfg, action_dim=6)
    gripper = Mock(spec=BinaryJointPositionZeroToOneAction)
    gripper.action_dim = 1
    gripper.cfg = SimpleNamespace(clip=None)
    terms = {"arm_action": arm, "gripper_action": gripper}
    manager = SimpleNamespace(active_terms=list(terms), get_term=terms.__getitem__)
    base = SimpleNamespace(scene={"robot": robot}, num_envs=2, action_manager=manager)
    env = SimpleNamespace(unwrapped=base)
    observation = {
        "policy": {"joint_pos": torch.arange(14).reshape(2, 7).float(), "gripper_pos": torch.tensor([[0.1], [0.9]])}
    }
    return env, observation, T_B_G


def _test_state_snapshot(simulation_app):
    from isaaclab_arena.embodiments.droid.policy_utils import extract_droid_state

    env, observation, expected = _stub_env()
    state = extract_droid_state(env, observation)
    torch.testing.assert_close(state.T_B_G, expected)
    torch.testing.assert_close(state.gripper_closed, torch.tensor([[0.1], [0.9]]))
    observation["policy"]["joint_pos"].zero_()
    observation["policy"]["gripper_pos"].zero_()
    env.unwrapped.scene["robot"].data.root_pose_w.torch.zero_()
    env.unwrapped.scene["robot"].data.body_link_pose_w.torch.zero_()
    torch.testing.assert_close(state.T_B_G, expected)
    assert state.joint_position[1, 6] == 13
    assert state.gripper_closed[1, 0] == 0.9
    assert state.T_W_B[1, 2] == 4
    return True


def test_state_snapshot():
    assert run_function_with_persistent_simulation_app(_test_state_snapshot)


def _test_pose_action(simulation_app, offset, reverse):
    from isaaclab.utils.math import apply_delta_pose, combine_frame_transforms, matrix_from_quat

    from isaaclab_arena.embodiments.droid.policy_utils import droid_pose_to_action

    env, _, current = _stub_env()
    manager = env.unwrapped.action_manager
    cfg = manager.get_term("arm_action").cfg
    if reverse:
        manager.active_terms.reverse()
        cfg.scale = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    if offset:
        cfg.body_offset = SimpleNamespace(pos=(0.131, 0, 0), rot=(0, 0, 2**-0.5, 2**-0.5))
    target = current.clone()
    target[:, :3] += torch.tensor([0.01, -0.02, 0.03])
    target[:, 3:] = torch.tensor([0.1, 0, 0, (1 - 0.1**2) ** 0.5])
    target[1, 3:] *= -1  # q and -q must produce equivalent rotations.
    closure = torch.tensor([[0.0], [1.0]])
    actions = droid_pose_to_action(env, target, closure)
    if reverse:
        arm_actions, gripper_actions = actions[:, 1:], actions[:, :1]
    else:
        arm_actions, gripper_actions = actions[:, :6], actions[:, 6:]
    torch.testing.assert_close(gripper_actions, closure)
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
        droid_pose_to_action(env, target * 2, closure)
    with pytest.raises(AssertionError, match="closure"):
        droid_pose_to_action(env, target, closure - 1)
    cfg.scale = 0
    with pytest.raises(AssertionError, match="nonzero"):
        droid_pose_to_action(env, target, closure)
    cfg.controller.use_relative_mode = False
    with pytest.raises(AssertionError, match="relative pose IK"):
        droid_pose_to_action(env, target, closure)
    return True


@pytest.mark.parametrize("offset", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_pose_action(offset, reverse):
    assert run_function_with_persistent_simulation_app(_test_pose_action, offset=offset, reverse=reverse)


def _test_camera_inputs(simulation_app):
    from isaaclab.utils.math import combine_frame_transforms

    from isaaclab_arena.embodiments.droid.camera_utils import (
        enable_droid_camera_pose_updates,
        extract_droid_camera_inputs,
    )

    env, observation, _ = _stub_env()
    key = "wrist_camera_rgb"
    pixels = torch.zeros((2, 5, 9, 4), dtype=torch.uint8)
    pixels[1, ..., :3] = torch.tensor([25, 80, 160], dtype=torch.uint8)
    pixels[..., 3] = 255
    observation["camera_obs"] = {key: pixels}
    K = torch.tensor([[100.0, 2.0, 4.5], [0.0, 120.0, 2.5], [0.0, 0.0, 1.0]]).repeat(2, 1, 1)
    root = env.unwrapped.scene["robot"].data.root_pose_w.torch
    position, rotation = combine_frame_transforms(
        root[:, :3],
        root[:, 3:],
        torch.tensor([[0.1, 0.2, 0.3]]).repeat(2, 1),
        torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1),
    )
    cfg = SimpleNamespace(update_latest_camera_pose=False, update_period=0.1)
    env.unwrapped.scene["wrist_camera"] = SimpleNamespace(
        cfg=cfg,
        data=SimpleNamespace(
            image_shape=(5, 9),
            intrinsic_matrices=SimpleNamespace(torch=K),
            pos_w=SimpleNamespace(torch=position),
            quat_w_ros=SimpleNamespace(torch=rotation),
        ),
    )
    with pytest.raises(AssertionError, match="enable_droid_camera_pose_updates"):
        extract_droid_camera_inputs(env, observation, 1, (key,))
    uncalibrated = extract_droid_camera_inputs(env, observation, 1, (key,), include_calibration=False)[key]
    assert uncalibrated.rgb.shape == (5, 9, 3)  # No upscaling.
    assert uncalibrated.intrinsics is None and uncalibrated.T_B_C is None
    enable_droid_camera_pose_updates(SimpleNamespace(wrist_camera=cfg), (key,))
    camera = extract_droid_camera_inputs(env, observation, 1, (key,), image_max_edge=4)[key]
    assert camera.rgb.shape == (2, 4, 3)
    np.testing.assert_array_equal(camera.rgb, np.broadcast_to([25, 80, 160], (2, 4, 3)))
    np.testing.assert_allclose(camera.intrinsics, [[100 * 4 / 9, 2 * 4 / 9, 2], [0, 48, 1], [0, 0, 1]])
    np.testing.assert_allclose(camera.T_B_C, [0.1, 0.2, 0.3, 0, 0, 0, 1], atol=1e-6)
    pixels.zero_()
    K.zero_()
    position.zero_()
    np.testing.assert_array_equal(uncalibrated.rgb[0, 0], [25, 80, 160])
    assert camera.intrinsics[2, 2] == 1
    assert camera.T_B_C[0] == pytest.approx(0.1, abs=1e-6)
    return True


def test_camera_inputs():
    assert run_function_with_persistent_simulation_app(_test_camera_inputs)


def _test_droid_runtime(simulation_app, cameras):
    import gymnasium as gym

    from isaaclab.utils.math import matrix_from_quat

    from isaaclab_arena.embodiments.droid.camera_utils import (
        enable_droid_camera_pose_updates,
        extract_droid_camera_inputs,
    )
    from isaaclab_arena.embodiments.droid.droid import DroidDifferentialIKEmbodiment
    from isaaclab_arena.embodiments.droid.policy_utils import droid_pose_to_action, extract_droid_state
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    embodiment = DroidDifferentialIKEmbodiment(enable_cameras=cameras)
    embodiment.set_initial_pose(Pose(position_xyz=(0, 0, 1), rotation_xyzw=(0, 0, 2**-0.5, 2**-0.5)))
    if cameras:
        enable_droid_camera_pose_updates(embodiment.camera_config)
        for name in ("external_camera", "external_camera_2", "wrist_camera"):
            camera_cfg = getattr(embodiment.camera_config, name)
            camera_cfg.width, camera_cfg.height = 96, 64
    name = "droid_policy_utils_test"
    arena = IsaacLabArenaEnvironment(name=name, embodiment=embodiment, scene=Scene(assets=[]))
    env = ArenaEnvBuilder(arena, ArenaEnvBuilderCfg(num_envs=2)).make_registered()
    try:
        with torch.inference_mode():
            observation, _ = env.reset()
            state = extract_droid_state(env, observation)
            target = state.T_B_G.clone()
            target[:, 2] += 0.01
            closure = target.new_tensor([[0.0], [1.0]])
            action = droid_pose_to_action(env, target, closure)
            arm = env.unwrapped.action_manager.get_term("arm_action")
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
            assert torch.isfinite(extract_droid_state(env, observation).T_B_G).all()
            if cameras:
                inputs = extract_droid_camera_inputs(env, observation, 1, image_max_edge=48)
                for key, camera in inputs.items():
                    assert camera.rgb.shape == (32, 48, 3)
                    assert camera.rgb.dtype == np.uint8
                    assert np.isfinite(camera.T_B_C).all()
                    assert np.linalg.norm(camera.T_B_C[3:]) == pytest.approx(1, abs=1e-5)
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

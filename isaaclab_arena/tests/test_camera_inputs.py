# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Camera preparation with caller-provided frames and no robot scene entity."""

import numpy as np
import torch
from types import SimpleNamespace

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_camera_inputs(simulation_app):
    from isaaclab.utils.math import combine_frame_transforms

    from isaaclab_arena.utils.camera_inputs import enable_camera_pose_updates, extract_camera_inputs

    env = SimpleNamespace(unwrapped=SimpleNamespace(num_envs=2, scene={}))
    observation = {}
    key = "overhead_rgb"
    pixels = torch.zeros((2, 5, 9, 4), dtype=torch.uint8)
    pixels[1, ..., :3] = torch.tensor([25, 80, 160], dtype=torch.uint8)
    pixels[..., 3] = 255
    observation["camera_obs"] = {key: pixels}
    K = torch.tensor([[100.0, 2.0, 4.5], [0.0, 120.0, 2.5], [0.0, 0.0, 1.0]]).repeat(2, 1, 1)
    root = torch.tensor([[0, 0, 1, 0, 0, 0, 1], [2, 3, 4, 0, 0, 2**-0.5, 2**-0.5]])
    position, rotation = combine_frame_transforms(
        root[:, :3],
        root[:, 3:],
        torch.tensor([[0.1, 0.2, 0.3]]).repeat(2, 1),
        torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1),
    )
    cfg = SimpleNamespace(update_latest_camera_pose=False, update_period=0.1)
    env.unwrapped.scene["overhead"] = SimpleNamespace(
        cfg=cfg,
        data=SimpleNamespace(
            image_shape=(5, 9),
            intrinsic_matrices=SimpleNamespace(torch=K),
            pos_w=SimpleNamespace(torch=position),
            quat_w_ros=SimpleNamespace(torch=rotation),
        ),
    )
    with pytest.raises(AssertionError, match="enable_camera_pose_updates"):
        extract_camera_inputs(env, observation, 1, (key,), T_W_R=root)
    uncalibrated = extract_camera_inputs(env, observation, 1, (key,), include_calibration=False)[key]
    assert uncalibrated.rgb.shape == (5, 9, 3)  # No upscaling.
    assert uncalibrated.intrinsics is None and uncalibrated.T_R_C is None
    enable_camera_pose_updates(SimpleNamespace(overhead=cfg), (key,))
    camera = extract_camera_inputs(env, observation, 1, (key,), T_W_R=root, image_max_edge=4)[key]
    assert camera.rgb.shape == (2, 4, 3)
    np.testing.assert_array_equal(camera.rgb, np.broadcast_to([25, 80, 160], (2, 4, 3)))
    np.testing.assert_allclose(camera.intrinsics, [[100 * 4 / 9, 2 * 4 / 9, 2], [0, 48, 1], [0, 0, 1]])
    np.testing.assert_allclose(camera.T_R_C, [0.1, 0.2, 0.3, 0, 0, 0, 1], atol=1e-6)
    identity = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)
    world_camera = extract_camera_inputs(env, observation, 1, (key,), T_W_R=identity)[key]
    np.testing.assert_allclose(world_camera.T_R_C[:3], position[1].numpy(), atol=1e-6)
    np.testing.assert_allclose(world_camera.T_R_C[3:], rotation[1].numpy(), atol=1e-6)
    with pytest.raises(AssertionError, match="T_W_R is required"):
        extract_camera_inputs(env, observation, 1, (key,))
    with pytest.raises(AssertionError, match="one reference pose"):
        extract_camera_inputs(env, observation, 1, (key,), T_W_R=root[0])
    with pytest.raises(AssertionError, match="unit XYZW"):
        extract_camera_inputs(env, observation, 1, (key,), T_W_R=root * 2)
    pixels.zero_()
    K.zero_()
    position.zero_()
    np.testing.assert_array_equal(uncalibrated.rgb[0, 0], [25, 80, 160])
    assert camera.intrinsics[2, 2] == 1
    assert camera.T_R_C[0] == pytest.approx(0.1, abs=1e-6)
    return True


def test_camera_inputs():
    assert run_function_with_persistent_simulation_app(_test_camera_inputs)

# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Regression test for Isaac Lab ``XformPrimView.get_local_poses`` quaternion layout.

``XformPrimView`` docstrings claim local orientations are ``(w, x, y, z)``, but the
implementation uses ``(x, y, z, w)``. This test checks that ``get_local_poses()``
returns the same quaternion as :class:`~isaaclab.sensors.camera.camera_cfg.CameraCfg.OffsetCfg`.
"""

import torch

import pytest

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function
from isaaclab_arena.utils.pose import Pose

HEADLESS = True
ENABLE_CAMERAS = True
CAMERA_NAME = "wrist_cam"

TEST_CAMERA_OFFSET = Pose(
    position_xyz=(0.123, -0.045, 0.067),
    rotation_xyzw=(0.3826834323650898, 0.0, 0.0, 0.9238795325112867),
)


def _test_get_local_poses_matches_camera_offset_cfg(simulation_app) -> bool:
    from isaaclab.sensors import CameraCfg

    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.franka.franka import FrankaIKEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    arena_env = IsaacLabArenaEnvironment(
        name="test_isaac_lab_bug_local_poses",
        embodiment=FrankaIKEmbodiment(enable_cameras=True, camera_offset=TEST_CAMERA_OFFSET),
        scene=Scene(),
    )
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1", "--enable_cameras"])
    builder = ArenaEnvBuilder(arena_env, args_cli)
    env_cfg, env_kwargs = builder.compose_manager_cfg()

    env_cfg.scene.wrist_cam.offset = CameraCfg.OffsetCfg(
        pos=TEST_CAMERA_OFFSET.position_xyz,
        rot=TEST_CAMERA_OFFSET.rotation_xyzw,
        convention="opengl",
    )

    env, _ = builder.make_registered_and_return_cfg(env_cfg, env_kwargs)
    env.reset()

    camera = env.unwrapped.scene[CAMERA_NAME]
    offset_cfg = camera.cfg.offset
    view = camera._view
    assert view is not None, "Camera XformPrimView was not initialized."

    _, q_parent_C = view.get_local_poses()
    expected_rot_xyzw = torch.tensor(offset_cfg.rot, device=q_parent_C.device, dtype=q_parent_C.dtype)
    torch.testing.assert_close(q_parent_C[0], expected_rot_xyzw, atol=1e-5, rtol=1e-5)

    env.close()
    return True


@pytest.mark.with_cameras
def test_get_local_poses_matches_camera_offset_cfg():
    assert run_simulation_app_function(
        _test_get_local_poses_matches_camera_offset_cfg,
        headless=HEADLESS,
        enable_cameras=ENABLE_CAMERAS,
    )

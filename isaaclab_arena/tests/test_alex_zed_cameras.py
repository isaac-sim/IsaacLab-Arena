# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import tqdm

import pytest

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

NUM_STEPS = 2
HEADLESS = True
ENABLE_CAMERAS = True


def _test_alex_zed_cameras(simulation_app) -> bool:
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    args_parser = get_isaaclab_arena_cli_parser()
    args_cli = args_parser.parse_args(["--enable_cameras", "--num_envs", "1"])

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("kitchen")()
    microwave = asset_registry.get_asset_by_name("microwave")()
    microwave.set_initial_pose(
        Pose(
            position_xyz=(0.4, -0.00586, 0.22773),
            rotation_xyzw=(0, 0, -0.7071068, 0.7071068),
        )
    )

    embodiment = asset_registry.get_asset_by_name("alex_ability_hands")(enable_cameras=True)
    embodiment.set_initial_pose(Pose(position_xyz=(-0.55, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    scene = Scene(assets=[background, microwave])
    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="alex_zed_camera_test",
        embodiment=embodiment,
        scene=scene,
    )

    builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
    env = builder.make_registered()
    env.reset()
    for _ in tqdm.tqdm(range(NUM_STEPS)):
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            obs, _, _, _, _ = env.step(actions)
            camera_obs = obs["camera_obs"]
            assert "zed_left_cam_rgb" in camera_obs, f"Missing zed_left_cam_rgb in {list(camera_obs.keys())}"
            assert "zed_right_cam_rgb" in camera_obs, f"Missing zed_right_cam_rgb in {list(camera_obs.keys())}"
            left_rgb = camera_obs["zed_left_cam_rgb"]
            right_rgb = camera_obs["zed_right_cam_rgb"]
            assert left_rgb.shape[-1] == 3, left_rgb.shape
            assert right_rgb.shape[-1] == 3, right_rgb.shape

    env.close()
    return True


@pytest.mark.with_cameras
def test_alex_zed_cameras():
    result = run_simulation_app_function(
        _test_alex_zed_cameras,
        headless=HEADLESS,
        enable_cameras=ENABLE_CAMERAS,
    )
    assert result, "Test failed"

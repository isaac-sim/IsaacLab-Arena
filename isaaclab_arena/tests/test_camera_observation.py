# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import tqdm

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

NUM_STEPS = 2
HEADLESS = True
ENABLE_CAMERAS = True


def _test_camera_observation(simulation_app) -> bool:

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.gr1t2.gr1t2 import GR1T2PinkEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    args_parser = get_isaaclab_arena_cli_parser()
    args_cli = args_parser.parse_args(["--enable_cameras"])

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("packing_table")()
    cracker_box = asset_registry.get_asset_by_name("cracker_box")()

    cracker_box.set_initial_pose(
        Pose(
            position_xyz=(0.0758066475391388, -0.5088448524475098, 0.0),
            rotation_xyzw=(0, 0, 0, 1),
        )
    )

    scene = Scene(assets=[background, cracker_box])

    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="camera_observation_test",
        embodiment=GR1T2PinkEmbodiment(enable_cameras=True),
        scene=scene,
    )

    # Compile an IsaacLab compatible arena environment configuration
    builder = ArenaEnvBuilder(isaaclab_arena_environment, arena_env_builder_cfg_from_argparse(args_cli))
    env = builder.make_registered()
    env.reset()
    for _ in tqdm.tqdm(range(NUM_STEPS)):
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            obs, _, _, _, _ = env.step(actions)
            # Get the camera observation
            camera_observation = obs["camera_obs"]["robot_pov_cam_rgb"]
            # Assert that the camera rgb observation has three channels
            assert camera_observation.shape[3] == 3, "Camera rgb observation does not have three channels"
            # Make sure the camera observation contains values other than 0
            assert camera_observation.any() != 0, "Camera observation contains only 0s"

    _check_camera_frames_render_only_when_requested(env)

    env.close()

    return True


def _check_camera_frames_render_only_when_requested(env) -> None:
    """Step without a frame request, then with one, and compare the camera render count and images."""
    from isaaclab_arena.utils.timer import get_timer_stats

    def render_count() -> int:
        return sum(stats.count for name, stats in get_timer_stats().items() if name.endswith("camera_render"))

    def step(requested: bool):
        env.unwrapped.request_camera_frames(requested)
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        with torch.inference_mode():
            obs, _, _, _, _ = env.step(actions)
        return obs["camera_obs"]["robot_pov_cam_rgb"].clone()

    rendered_image = step(True)
    renders_before = render_count()
    held_image = step(False)
    assert render_count() == renders_before, "Camera rendered although no frame was requested"
    assert torch.equal(held_image, rendered_image), "Held camera observation differs from the last rendered frame"
    assert env.unwrapped.camera_frames_requested, "Frame request flag must reset to True after a step"

    step(True)
    assert render_count() == renders_before + 1, "Camera did not render when a frame was requested"


@pytest.mark.with_cameras
def test_camera_observation():

    result = run_function_with_persistent_simulation_app(
        _test_camera_observation,
        headless=HEADLESS,
        enable_cameras=ENABLE_CAMERAS,
    )
    assert result, "Test failed"


if __name__ == "__main__":
    test_camera_observation()

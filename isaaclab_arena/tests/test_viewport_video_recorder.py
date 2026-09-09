# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the report video recorder's viewer-frame handling and headless capture."""

import numpy as np
from pathlib import Path
from types import SimpleNamespace

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

NUM_ENVS = 2
NUM_STEPS = 8
HEADLESS = True
ENABLE_CAMERAS = True

EXPECTED_VIDEO_FILENAME = "viewport-env0-viewport-episode-0.mp4"


def test_viewer_origin_resolves_to_the_selected_environment():
    """Environment-relative viewpoints resolve against the origin of the selected clone."""
    import torch

    from isaaclab_arena.video.viewport_video_recorder import ArenaViewportVideoRecorder

    resolve = ArenaViewportVideoRecorder._resolve_viewer_origin
    scene = SimpleNamespace(env_origins=torch.tensor([[15.0, -15.0, 0.0], [-15.0, 15.0, 0.0]]))

    assert np.allclose(resolve(scene, "world", 0), (0.0, 0.0, 0.0))
    assert np.allclose(resolve(scene, "env", 0), (15.0, -15.0, 0.0))
    assert np.allclose(resolve(scene, "env", 1), (-15.0, 15.0, 0.0))


def test_viewer_origin_rejects_frames_it_cannot_resolve():
    """Unsupported viewer frames fail loudly rather than silently aiming the camera at world zero."""
    import torch

    from isaaclab_arena.video.viewport_video_recorder import ArenaViewportVideoRecorder

    resolve = ArenaViewportVideoRecorder._resolve_viewer_origin
    scene = SimpleNamespace(env_origins=torch.tensor([[15.0, -15.0, 0.0], [-15.0, 15.0, 0.0]]))

    with pytest.raises(AssertionError, match="asset_root"):
        resolve(scene, "asset_root", 0)
    with pytest.raises(AssertionError, match="outside the available range"):
        resolve(scene, "env", NUM_ENVS)


def test_report_recorder_does_not_follow_an_interactive_visualizer():
    """An active visualizer must not replace the resolved viewpoint with its own camera.

    Isaac Lab's ``"visualizer"`` backend source overwrites the recorder's eye and target from the
    visualizer config whenever one is running, which would discard the environment offset.
    """
    from isaaclab_arena.video.viewport_video_recorder import ArenaViewportVideoRecorderCfg

    assert ArenaViewportVideoRecorderCfg().backend_source == "renderer"


def _build_multi_env_goal_pose_env(render_mode: str):
    """Build a two-environment goal-pose environment whose viewer frame is environment-relative."""
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.franka.franka import FrankaIKEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.goal_pose_task import GoalPoseTask
    from isaaclab_arena.utils.pose import Pose

    args_parser = get_isaaclab_arena_cli_parser()
    args_cli = args_parser.parse_args(["--num_envs", str(NUM_ENVS), "--enable_cameras"])

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("table")()
    light = asset_registry.get_asset_by_name("light")()
    dex_cube = asset_registry.get_asset_by_name("dex_cube")()
    dex_cube.set_initial_pose(Pose(position_xyz=(0.1, 0.0, 0.05), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    embodiment = FrankaIKEmbodiment()
    embodiment.set_initial_pose(Pose(position_xyz=(-0.4, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    arena_environment = IsaacLabArenaEnvironment(
        name="test_viewport_video_recorder",
        embodiment=embodiment,
        scene=Scene(assets=[background, light, dex_cube]),
        task=GoalPoseTask(dex_cube),
    )
    builder = ArenaEnvBuilder(arena_environment, arena_env_builder_cfg_from_argparse(args_cli))
    return builder.make_registered(render_mode=render_mode)


def _test_viewport_video_recording(simulation_app, video_dir: str) -> bool:
    import torch

    from isaaclab_arena.video.video_recording import VideoRecordingCfg, wrap_env_for_video

    env = _build_multi_env_goal_pose_env(render_mode="rgb_array")

    # The recorder must aim at the selected environment, not at the same coordinates in world space.
    viewer_cfg = env.unwrapped.cfg.viewer
    assert viewer_cfg.origin_type == "env", f"Expected an environment-relative viewer, got '{viewer_cfg.origin_type}'."
    env_origin = env.unwrapped.scene.env_origins[viewer_cfg.env_index].detach().cpu().numpy()
    assert (
        np.linalg.norm(env_origin) > 0.0
    ), "Environment origin is at the world origin; the test cannot detect the bug."

    recorder_cfg = env.unwrapped.video_recorder.cfg
    assert np.allclose(recorder_cfg.eye, env_origin + np.asarray(viewer_cfg.eye)), (
        f"Report camera eye {recorder_cfg.eye} does not match the viewer eye {viewer_cfg.eye} "
        f"placed at environment origin {env_origin}."
    )
    assert np.allclose(recorder_cfg.lookat, env_origin + np.asarray(viewer_cfg.lookat)), (
        f"Report camera target {recorder_cfg.lookat} does not match the viewer target {viewer_cfg.lookat} "
        f"placed at environment origin {env_origin}."
    )

    # Recording must produce a real mp4 with visible content while headless.
    video_cfg = VideoRecordingCfg(record_viewport_video=True, video_base_dir=video_dir)
    env = wrap_env_for_video(env, video_cfg, num_steps=NUM_STEPS, num_episodes=None)
    env.reset()
    for _ in range(NUM_STEPS):
        with torch.inference_mode():
            env.step(torch.zeros(env.action_space.shape, device=env.unwrapped.device))

    frame = env.unwrapped.render()
    assert frame is not None, "Headless render returned no frame."
    assert frame.any(), "Headless render returned an all-black frame."
    env.close()

    videos = sorted(Path(video_dir).glob("*.mp4"))
    assert [video.name for video in videos] == [EXPECTED_VIDEO_FILENAME], f"Unexpected videos written: {videos}."
    assert videos[0].stat().st_size > 0, "Viewport video is empty."

    return True


@pytest.mark.with_cameras
def test_viewport_video_recording(tmp_path):
    result = run_function_with_persistent_simulation_app(
        _test_viewport_video_recording,
        headless=HEADLESS,
        enable_cameras=ENABLE_CAMERAS,
        video_dir=str(tmp_path),
    )
    assert result, "Test failed"


if __name__ == "__main__":
    test_viewport_video_recording(Path("/tmp/test_viewport_video_recorder"))

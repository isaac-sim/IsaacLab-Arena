# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Tests poke private helpers and dynamic Isaac Lab env attributes by design.
# pyright: reportPrivateUsage=false, reportAttributeAccessIssue=false

"""Tests for SynchronizedVisualizer.

Phase 1 (no sim): pure-logic helpers — config validation, tiling, the rgb
guard, dependency check, and frame conversion. Phase 2 (``with_cameras``): an
end-to-end smoke test that builds a tiny env, captures frames, and writes video.
"""

import numpy as np
import os
import torch
from types import SimpleNamespace

import pytest

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function
from isaaclab_arena.utils.synchronized_visualizer import EnvView, GlobalView, SynchronizedVisualizer, _CaptureBuffers


# --------------------------------------------------------------------------- #
# Phase 1: pure-logic unit tests (no SimulationApp)                           #
# --------------------------------------------------------------------------- #
def test_envview_rejects_equal_eye_and_lookat():
    with pytest.raises(AssertionError):
        EnvView(eye_offset=(1.0, 1.0, 1.0), lookat_offset=(1.0, 1.0, 1.0))


def test_envview_rejects_nonpositive_dims():
    with pytest.raises(AssertionError):
        EnvView(width=0)


def test_globalview_accepts_default():
    GlobalView()  # should not raise


@pytest.mark.parametrize("elevation", [90.0, -90.0, 120.0])
def test_globalview_rejects_degenerate_elevation(elevation):
    with pytest.raises(AssertionError):
        GlobalView(elevation_deg=elevation)


def test_globalview_rejects_nonpositive_dims():
    with pytest.raises(AssertionError):
        GlobalView(height=-10)


def test_tile_empty_raises():
    with pytest.raises(AssertionError):
        SynchronizedVisualizer._tile([], num_cols=2)


def test_tile_rejects_mismatched_shapes():
    a = np.zeros((2, 3, 3), dtype=np.uint8)
    b = np.zeros((4, 3, 3), dtype=np.uint8)
    with pytest.raises(AssertionError):
        SynchronizedVisualizer._tile([a, b], num_cols=2)


def test_tile_layout_places_images_in_grid():
    red = np.full((2, 3, 3), 0, dtype=np.uint8)
    red[..., 0] = 255
    green = np.full((2, 3, 3), 0, dtype=np.uint8)
    green[..., 1] = 255

    grid = SynchronizedVisualizer._tile([red, green], num_cols=2)

    assert grid.shape == (2, 6, 3)  # 1 row x 2 cols of 2x3 tiles
    np.testing.assert_array_equal(grid[:, 0:3], red)
    np.testing.assert_array_equal(grid[:, 3:6], green)


def test_read_rgb_raises_on_empty_output():
    camera = SimpleNamespace(data=SimpleNamespace(output={}))
    with pytest.raises(RuntimeError, match="rgb"):
        SynchronizedVisualizer._read_rgb(camera, "per-env")


def test_read_rgb_returns_present_output():
    sentinel = object()
    camera = SimpleNamespace(data=SimpleNamespace(output={"rgb": sentinel}))
    assert SynchronizedVisualizer._read_rgb(camera, "global") is sentinel


def test_check_optional_deps_passes_when_installed():
    # moviepy and Pillow ship in the container; this should be a no-op.
    SynchronizedVisualizer._check_optional_deps()


def test_to_uint8_clamps_and_drops_alpha():
    frame = torch.tensor([[[300.0, -5.0, 128.0, 10.0]]])  # 1x1, RGBA, out-of-range
    out = SynchronizedVisualizer._to_uint8(frame)
    assert out.dtype == np.uint8
    assert out.shape == (1, 1, 3)
    np.testing.assert_array_equal(out[0, 0], np.array([255, 0, 128], dtype=np.uint8))


def test_downscale_to_width_keeps_aspect_and_even_dims():
    img = np.zeros((20, 40, 3), dtype=np.uint8)
    out = SynchronizedVisualizer._downscale_to_width(img, max_width=10)
    assert out.shape[1] == 10  # target width
    assert out.shape[0] % 2 == 0 and out.shape[1] % 2 == 0  # h264-friendly
    # aspect-preserved height 5 is decremented to even 4 for h264 compatibility
    assert out.shape[0] == 4


def test_tile_rejects_width_mismatch():
    a = np.zeros((2, 3, 3), dtype=np.uint8)
    b = np.zeros((2, 5, 3), dtype=np.uint8)
    with pytest.raises(AssertionError):
        SynchronizedVisualizer._tile([a, b], num_cols=2)


def test_tile_pads_incomplete_last_row():
    img = np.full((2, 3, 3), 7, dtype=np.uint8)
    grid = SynchronizedVisualizer._tile([img, img, img], num_cols=2)
    assert grid.shape == (4, 6, 3)  # 3 images, 2 cols -> 2 rows
    # The unfilled 4th cell (row 1, col 1) stays zero-padded.
    np.testing.assert_array_equal(grid[2:4, 3:6], np.zeros((2, 3, 3), dtype=np.uint8))


def test_to_uint8_passthrough_for_uint8_input():
    frame = torch.zeros((2, 2, 3), dtype=torch.uint8)
    frame[0, 0, 0] = 200
    out = SynchronizedVisualizer._to_uint8(frame)
    assert out.dtype == np.uint8
    assert out.shape == (2, 2, 3)
    assert out[0, 0, 0] == 200


def test_resolve_global_pose_uses_explicit_eye_and_lookat():
    # Bare instance: _resolve_global_pose only needs global_view + device, no sim.
    viz = SynchronizedVisualizer.__new__(SynchronizedVisualizer)
    viz.device = "cpu"
    viz.global_view = GlobalView(eye=(1.0, 2.0, 3.0), lookat=(4.0, 5.0, 6.0))

    eye, target = viz._resolve_global_pose(torch.zeros((2, 3)))

    torch.testing.assert_close(eye, torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(target, torch.tensor([4.0, 5.0, 6.0]))


def test_resolve_global_pose_auto_frames_from_origins():
    viz = SynchronizedVisualizer.__new__(SynchronizedVisualizer)
    viz.device = "cpu"
    viz.global_view = GlobalView()  # eye=None -> auto-frame, elevation_deg=35

    eye, target = viz._resolve_global_pose(torch.zeros((2, 3)))  # centroid at origin

    torch.testing.assert_close(target, torch.zeros(3))
    assert eye[2] > 0.0  # elevation_deg > 0 -> camera above ground
    assert not torch.allclose(eye, target)  # camera has a viewing direction


def test_close_clears_state_and_initialized_flag():
    viz = SynchronizedVisualizer.__new__(SynchronizedVisualizer)
    viz._env_camera = object()
    viz._global_camera = object()
    viz._initialized = True

    viz.close()

    assert viz._env_camera is None
    assert viz._global_camera is None
    assert viz._initialized is False


def test_capture_before_initialize_raises():
    viz = SynchronizedVisualizer.__new__(SynchronizedVisualizer)
    viz._initialized = False
    viz._env_camera = None
    viz._global_camera = None
    with pytest.raises(AssertionError, match="initialize"):
        viz.capture()


def test_save_writes_gif_and_per_env_outputs(tmp_path):
    # Bare instance: save() only needs _buffers, no sim. Covers also_gif + save_per_env.
    viz = SynchronizedVisualizer.__new__(SynchronizedVisualizer)
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    buffers = _CaptureBuffers()
    buffers.global_frames = [frame, frame]
    buffers.grid_frames = [frame, frame]
    buffers.per_env_frames = {0: [frame, frame], 1: [frame, frame]}
    viz._buffers = buffers

    written = viz.save(str(tmp_path), fps=5, name_prefix="t", save_per_env=True, also_gif=True)

    for key in ("global", "grid", "global_gif", "grid_gif", "env000", "env001"):
        assert os.path.exists(written[key]), f"missing output: {key}"


# --------------------------------------------------------------------------- #
# Phase 2: sim smoke test (with_cameras)                                      #
# --------------------------------------------------------------------------- #
def _test_sync_viz_smoke(simulation_app) -> bool:
    import tempfile

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.gr1t2.gr1t2 import GR1T2PinkEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--enable_cameras", "--num_envs", "2"])

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("packing_table")()  # type: ignore[call-arg]
    cracker_box = asset_registry.get_asset_by_name("cracker_box")()  # type: ignore[call-arg]
    cracker_box.set_initial_pose(Pose(position_xyz=(0.0, -0.5, 0.0), rotation_xyzw=(0, 0, 0, 1)))
    scene = Scene(assets=[background, cracker_box])

    arena_env = IsaacLabArenaEnvironment(
        name="sync_viz_smoke_test",
        embodiment=GR1T2PinkEmbodiment(enable_cameras=True),
        scene=scene,
    )
    builder = ArenaEnvBuilder(arena_env, args_cli)

    # Small render targets keep the smoke test fast.
    env_view = EnvView(width=64, height=48)
    env_cfg = builder.compose_manager_cfg()
    SynchronizedVisualizer.add_env_camera_to_cfg(env_cfg, env_view)
    env = builder.make_registered(env_cfg=env_cfg, render_mode=None)

    viz = SynchronizedVisualizer(env, env_view=env_view, global_view=GlobalView(width=128, height=96))
    try:
        viz.initialize()
        assert viz._using_tiled, "Injected scene TiledCamera should be used, not the standalone fallback."

        # OrderEnforcing requires reset() before step(); reposition after reset
        # so the per-env camera offset is re-applied (mirrors the example script).
        env.reset()
        viz.reposition()

        sim = env.unwrapped.sim
        num_frames = 2
        for _ in range(num_frames):
            with torch.inference_mode():
                env.step(torch.zeros(env.action_space.shape, device=env.unwrapped.device))
            sim.render()
            viz.capture()

        assert len(viz._buffers.global_frames) == num_frames
        assert len(viz._buffers.grid_frames) == num_frames
        # 2 envs -> ceil(sqrt(2)) = 2 cols x 1 row of (height x width) tiles.
        # Asserting the composed size verifies tiling against real camera output
        # without depending on scene lighting (an ``any()`` check can flake).
        grid = viz._buffers.grid_frames[0]
        assert grid.dtype == np.uint8
        assert grid.shape == (env_view.height, 2 * env_view.width, 3)

        with tempfile.TemporaryDirectory() as out_dir:
            written = viz.save(out_dir, fps=5, name_prefix="smoke")
            assert os.path.exists(written["global"])
            assert os.path.exists(written["grid"])
    finally:
        viz.close()
        env.close()

    return True


@pytest.mark.with_cameras
def test_sync_viz_smoke():
    result = run_simulation_app_function(_test_sync_viz_smoke, headless=True, enable_cameras=True)
    assert result, "Test failed"


if __name__ == "__main__":
    test_sync_viz_smoke()

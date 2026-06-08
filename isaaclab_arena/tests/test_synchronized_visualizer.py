# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Tests poke private helpers and dynamic Isaac Lab env attributes by design.
# pyright: reportPrivateUsage=false, reportAttributeAccessIssue=false

"""Tests for SynchronizedVisualizer.

Most tests run without a simulator: config validation, tiling, the rgb guard,
dependency check, and frame conversion. The with_cameras smoke test builds a tiny
env, captures frames, and writes video end to end.
"""

import numpy as np
import os
import torch
from types import SimpleNamespace

import pytest

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function
from isaaclab_arena.utils import video_io
from isaaclab_arena.utils.synchronized_visualizer import EnvView, GlobalView, SynchronizedVisualizer, _CaptureBuffers


# --------------------------------------------------------------------------- #
# Pure-logic unit tests (no SimulationApp)                                     #
# --------------------------------------------------------------------------- #
def test_envview_rejects_equal_eye_and_lookat():
    with pytest.raises(AssertionError):
        EnvView(eye_offset=(1.0, 1.0, 1.0), lookat_offset=(1.0, 1.0, 1.0))


def test_envview_rejects_nonpositive_dims():
    with pytest.raises(AssertionError):
        EnvView(width=0)
    with pytest.raises(AssertionError):
        EnvView(height=0)


def test_envview_rejects_nonpositive_focal_length():
    with pytest.raises(AssertionError):
        EnvView(focal_length=0.0)


def test_globalview_accepts_default():
    GlobalView()


def test_globalview_rejects_nonpositive_focal_length():
    with pytest.raises(AssertionError):
        GlobalView(focal_length=0.0)


@pytest.mark.parametrize("elevation", [90.0, -90.0, 120.0])
def test_globalview_rejects_degenerate_elevation(elevation):
    with pytest.raises(AssertionError):
        GlobalView(elevation_deg=elevation)


def test_globalview_explicit_eye_allows_out_of_range_elevation():
    # With an explicit eye, elevation_deg/distance_scale are unused, so even
    # otherwise-degenerate values must be accepted (the guard is auto-frame only).
    GlobalView(eye=(1.0, 2.0, 3.0), elevation_deg=120.0, distance_scale=-1.0)


def test_globalview_rejects_nonpositive_distance_scale_when_auto_framing():
    with pytest.raises(AssertionError):
        GlobalView(distance_scale=0.0)  # eye=None -> auto-frame path


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
    out = video_io.to_uint8(frame)
    assert out.dtype == np.uint8
    assert out.shape == (1, 1, 3)
    np.testing.assert_array_equal(out[0, 0], np.array([255, 0, 128], dtype=np.uint8))


def test_to_uint8_rescales_normalized_float():
    # All values <= 1.0 -> normalize=True path: rescale [0, 1] to [0, 255].
    frame = torch.tensor([[[0.0, 0.5, 1.0]]])
    out = video_io.to_uint8(frame)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out[0, 0], np.array([0, 127, 255], dtype=np.uint8))


def test_to_uint8_clamps_integer_dtype():
    # int32/uint16 are neither uint8 nor float, so they hit the catch-all branch:
    # clip to [0, 255] with no rescale (the normalize heuristic is float-only).
    out = video_io.to_uint8(np.array([[[300, -5, 128]]], dtype=np.int32))
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out[0, 0], np.array([255, 0, 128], dtype=np.uint8))
    out16 = video_io.to_uint8(np.array([[[300, 0, 200]]], dtype=np.uint16))
    np.testing.assert_array_equal(out16[0, 0], np.array([255, 0, 200], dtype=np.uint8))


def test_to_uint8_rejects_empty_frame():
    with pytest.raises(AssertionError):
        video_io.to_uint8(np.zeros((0, 4, 3), dtype=np.float32))


def test_write_video_rejects_empty_frames(tmp_path):
    # The assert fires before moviepy is imported, so this stays sim/dep-free.
    with pytest.raises(AssertionError):
        video_io.write_video([], str(tmp_path / "x.mp4"), fps=30)


def test_write_gif_rejects_empty_frames(tmp_path):
    with pytest.raises(AssertionError):
        video_io.write_gif([], str(tmp_path / "x.gif"), fps=30)


def test_downscale_to_width_keeps_aspect_and_even_dims():
    img = np.zeros((20, 40, 3), dtype=np.uint8)
    out = SynchronizedVisualizer._downscale_to_width(img, max_width=10)
    assert out.shape[1] == 10
    assert out.shape[0] % 2 == 0 and out.shape[1] % 2 == 0  # h264 needs even dims
    assert out.shape[0] == 4  # aspect height 5 rounded down to even


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
    out = video_io.to_uint8(frame)
    assert out.dtype == np.uint8
    assert out.shape == (2, 2, 3)
    assert out[0, 0, 0] == 200


def test_to_uint8_does_not_alias_source_ndarray():
    # env.render() hands back a reused annotator buffer; retained frames must not alias it.
    src = np.zeros((2, 2, 3), dtype=np.uint8)
    out = video_io.to_uint8(src)
    src[0, 0, 0] = 123
    assert out[0, 0, 0] == 0
    assert not np.shares_memory(out, src)


def test_to_uint8_does_not_alias_source_cpu_tensor():
    # tensor.cpu().numpy() shares storage for a CPU tensor; the uint8 path must copy.
    src = torch.zeros((2, 2, 3), dtype=torch.uint8)
    out = video_io.to_uint8(src)
    src[0, 0, 0] = 123
    assert out[0, 0, 0] == 0
    assert not np.shares_memory(out, src.numpy())


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


def test_resolve_global_pose_explicit_eye_auto_target():
    # eye set, lookat None: explicit eye, target falls back to origins centroid.
    viz = SynchronizedVisualizer.__new__(SynchronizedVisualizer)
    viz.device = "cpu"
    viz.global_view = GlobalView(eye=(1.0, 2.0, 3.0), lookat=None)

    origins = torch.tensor([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]])  # centroid (1, 2, 3)
    eye, target = viz._resolve_global_pose(origins)

    torch.testing.assert_close(eye, torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(target, torch.tensor([1.0, 2.0, 3.0]))


def test_resolve_global_pose_explicit_target_auto_eye():
    # lookat set, eye None: auto-frame eye, target uses the explicit lookat.
    viz = SynchronizedVisualizer.__new__(SynchronizedVisualizer)
    viz.device = "cpu"
    viz.global_view = GlobalView(eye=None, lookat=(9.0, 8.0, 7.0))

    eye, target = viz._resolve_global_pose(torch.zeros((2, 3)))

    torch.testing.assert_close(target, torch.tensor([9.0, 8.0, 7.0]))
    assert eye[2] > 0.0  # auto-framed above ground
    assert not torch.allclose(eye, target)


def test_reposition_poses_global_camera_prim():
    # reposition() resolves the global pose and authors it onto the dedicated global
    # camera prim. Stub _pose_camera_prim (the only sim/USD-touching part) to capture args.
    viz = SynchronizedVisualizer.__new__(SynchronizedVisualizer)
    viz.device = "cpu"
    viz.global_view = GlobalView(eye=(1.0, 2.0, 3.0), lookat=(0.0, 0.0, 0.0))
    viz.unwrapped = SimpleNamespace(scene=SimpleNamespace(env_origins=torch.zeros((2, 3))))

    captured: dict[str, object] = {}

    def _pose(path, eye, target):
        captured["path"], captured["eye"], captured["target"] = path, list(eye), list(target)

    viz._pose_camera_prim = _pose

    viz.reposition()

    assert captured["path"] == SynchronizedVisualizer.GLOBAL_CAM_PRIM_PATH
    np.testing.assert_allclose(captured["eye"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(captured["target"], [0.0, 0.0, 0.0])


def test_close_clears_state_and_buffers():
    viz = SynchronizedVisualizer.__new__(SynchronizedVisualizer)
    viz._env_camera = object()
    viz._initialized = True
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    old_buffers = _CaptureBuffers()
    old_buffers.global_frames = [frame]
    old_buffers.grid_frames = [frame]
    viz._buffers = old_buffers

    viz.close()

    assert viz._env_camera is None
    assert viz._initialized is False
    assert viz._closed is True
    # close() must swap in a fresh, empty buffer object (not mutate in place) so
    # the old buffers are dropped and any retained reference is decoupled.
    assert viz._buffers is not old_buffers
    assert viz._buffers.global_frames == [] and viz._buffers.grid_frames == []
    assert old_buffers.global_frames == [frame]  # the freed object is untouched


def test_initialize_rejects_double_init():
    viz = SynchronizedVisualizer.__new__(SynchronizedVisualizer)
    viz._closed = False
    viz._initialized = True
    with pytest.raises(AssertionError, match="exactly once"):
        viz.initialize()


def test_initialize_after_close_raises():
    # close() clears _initialized, so without the _closed guard initialize() would
    # silently re-arm a dead object; it must fail fast instead.
    viz = SynchronizedVisualizer.__new__(SynchronizedVisualizer)
    viz._closed = True
    viz._initialized = False
    with pytest.raises(AssertionError, match="after close"):
        viz.initialize()


def test_double_close_is_safe():
    viz = SynchronizedVisualizer.__new__(SynchronizedVisualizer)
    viz._env_camera = object()
    viz._initialized = True
    viz._buffers = _CaptureBuffers()
    viz.close()
    viz.close()  # idempotent: a second close must not raise
    assert viz._closed is True
    assert viz._buffers.global_frames == []


def test_capture_before_initialize_raises():
    viz = SynchronizedVisualizer.__new__(SynchronizedVisualizer)
    viz._closed = False
    viz._initialized = False
    viz._env_camera = None
    with pytest.raises(AssertionError, match="initialize"):
        viz.capture()


def test_capture_after_close_raises():
    viz = SynchronizedVisualizer.__new__(SynchronizedVisualizer)
    viz._closed = True
    viz._initialized = False  # close() also clears this; the closed check must win
    with pytest.raises(AssertionError, match="after close"):
        viz.capture()


def _bare_viz_for_capture(capture_per_env):
    """A bare instance wired for capture() without a simulator: fake camera + render."""
    viz = SynchronizedVisualizer.__new__(SynchronizedVisualizer)
    viz._closed = False
    viz._initialized = True
    viz.num_envs = 2
    viz.grid_cols = 2
    viz.max_grid_width = None
    viz.capture_per_env = capture_per_env
    viz._buffers = _CaptureBuffers()
    viz._warned_global_none = False
    viz._warned_global_black = False
    rgb = torch.zeros((2, 4, 6, 3), dtype=torch.uint8)
    rgb[..., 0] = 10
    viz._env_camera = SimpleNamespace(data=SimpleNamespace(output={"rgb": rgb}))
    viz.env = SimpleNamespace(render=lambda: np.full((96, 128, 3), 5, dtype=np.uint8))
    return viz


def test_capture_skips_per_env_when_disabled():
    viz = _bare_viz_for_capture(capture_per_env=False)
    viz.capture()
    assert viz._buffers.per_env_frames == {}  # gated off
    assert len(viz._buffers.grid_frames) == 1  # grid is still built from per_env


def test_capture_retains_per_env_when_enabled():
    viz = _bare_viz_for_capture(capture_per_env=True)
    viz.capture()
    assert set(viz._buffers.per_env_frames) == {0, 1}
    assert all(len(frames) == 1 for frames in viz._buffers.per_env_frames.values())


def test_capture_warns_once_on_none_render(capsys):
    # env.render() returning None means no global render product: warn one time, keep
    # the grid going, and record no global frame (buffers stay co-indexed minus global).
    viz = _bare_viz_for_capture(capture_per_env=False)
    viz.env = SimpleNamespace(render=lambda: None)
    viz.capture()
    viz.capture()
    out = capsys.readouterr().out
    assert out.count("env.render() returned None") == 1
    assert viz._buffers.global_frames == []
    assert len(viz._buffers.grid_frames) == 2


def test_capture_warns_once_on_all_black_global(capsys):
    # An all-zeros global frame (annotator not warmed up) warns once but is still encoded.
    viz = _bare_viz_for_capture(capture_per_env=False)
    viz.env = SimpleNamespace(render=lambda: np.zeros((96, 128, 3), dtype=np.uint8))
    viz.capture()
    viz.capture()
    out = capsys.readouterr().out
    assert out.count("all-zeros global frame") == 1
    assert len(viz._buffers.global_frames) == 2


def _fake_env(render_mode, sensors, resolution=(1280, 720)):
    """A minimal stand-in for a built env, enough to drive SynchronizedVisualizer.__init__.

    resolution defaults to the GlobalView default so __init__'s viewer-resolution
    contract check passes; override it to exercise the mismatch guard.
    """
    scene = SimpleNamespace(num_envs=2, sensors=sensors)
    cfg = SimpleNamespace(viewer=SimpleNamespace(resolution=resolution))
    return SimpleNamespace(render_mode=render_mode, unwrapped=SimpleNamespace(device="cpu", scene=scene, cfg=cfg))


def test_init_requires_rgb_array_render_mode():
    env = _fake_env(render_mode="human", sensors={SynchronizedVisualizer.ENV_CAM_NAME: object()})
    with pytest.raises(AssertionError, match="rgb_array"):
        SynchronizedVisualizer(env)


def test_init_requires_registered_env_camera():
    # The most common user mistake under the new API: forgetting Scene.add_sensor().
    env = _fake_env(render_mode="rgb_array", sensors={})
    with pytest.raises(AssertionError, match="No scene sensor named"):
        SynchronizedVisualizer(env)


def test_init_rejects_nonpositive_grid_cols():
    env = _fake_env(render_mode="rgb_array", sensors={SynchronizedVisualizer.ENV_CAM_NAME: object()})
    with pytest.raises(AssertionError, match="grid_cols must be positive"):
        SynchronizedVisualizer(env, grid_cols=0)


def test_init_rejects_viewer_resolution_mismatch():
    # GlobalView size must mirror cfg.viewer.resolution; otherwise the global frame
    # silently comes out at the viewport size instead of GlobalView's.
    env = _fake_env(
        render_mode="rgb_array",
        sensors={SynchronizedVisualizer.ENV_CAM_NAME: object()},
        resolution=(640, 480),
    )
    with pytest.raises(AssertionError, match="must match GlobalView"):
        SynchronizedVisualizer(env, global_view=GlobalView(width=1280, height=720))


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

    expected = ("global", "grid", "global_gif", "grid_gif", "env000", "env001", "env000_gif", "env001_gif")
    for key in expected:
        assert os.path.exists(written[key]), f"missing output: {key}"


def test_save_returns_empty_and_warns_when_no_frames(tmp_path, capsys):
    # Nothing captured: save() must return {} and warn rather than write empty files.
    viz = SynchronizedVisualizer.__new__(SynchronizedVisualizer)
    viz._buffers = _CaptureBuffers()
    written = viz.save(str(tmp_path), fps=5, name_prefix="t")
    assert written == {}
    assert "wrote nothing" in capsys.readouterr().out


def test_save_rejects_name_prefix_with_separator(tmp_path):
    # name_prefix joins into a path; a separator would let it escape output_dir.
    viz = SynchronizedVisualizer.__new__(SynchronizedVisualizer)
    viz._buffers = _CaptureBuffers()
    with pytest.raises(AssertionError, match="path separators"):
        viz.save(str(tmp_path), name_prefix="evil/escape")


def test_save_warns_when_per_env_requested_but_not_captured(tmp_path, capsys):
    # save_per_env=True without capture_per_env=True: warn, write global/grid anyway.
    viz = SynchronizedVisualizer.__new__(SynchronizedVisualizer)
    buffers = _CaptureBuffers()
    buffers.grid_frames = [np.zeros((4, 6, 3), dtype=np.uint8)]
    viz._buffers = buffers

    written = viz.save(str(tmp_path), fps=5, name_prefix="t", save_per_env=True)

    assert "capture_per_env=True" in capsys.readouterr().out
    assert "grid" in written and not any(k.startswith("env") for k in written)


# --------------------------------------------------------------------------- #
# Simulation smoke test (requires a SimulationApp with cameras)                #
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
    global_view = GlobalView(width=128, height=96)
    # Register the per-env tiled camera on the scene before composing the cfg.
    cam_name = SynchronizedVisualizer.ENV_CAM_NAME
    cam_cfg = SynchronizedVisualizer.build_env_camera_cfg(env_view, cam_name)
    scene.add_sensor(cam_name, cam_cfg)
    assert cam_name in scene.sensors

    # Scene.add_sensor must reject name collisions (Scene needs a live SimulationApp
    # to import, so these asserts live in the sim smoke test).
    with pytest.raises(AssertionError):
        scene.add_sensor(cam_name, cam_cfg)  # duplicate sensor name
    with pytest.raises(AssertionError):
        scene.add_sensor(background.name, cam_cfg)  # collides with an asset name
    with pytest.raises(AssertionError):
        scene.add_sensor("bad_type", object())  # not a SensorBaseCfg  # type: ignore[arg-type]

    env_cfg = builder.compose_manager_cfg()
    env_cfg.viewer.resolution = (global_view.width, global_view.height)
    env = builder.make_registered(env_cfg=env_cfg, render_mode="rgb_array")
    # add_sensor() should flow through to a real scene sensor on the built env.
    assert cam_name in env.unwrapped.scene.sensors

    viz = SynchronizedVisualizer(env, env_view=env_view, global_view=global_view, capture_per_env=True)
    try:
        # OrderEnforcing requires reset() before step(); initialize poses the global viewport.
        env.reset()
        viz.initialize()

        # Warm up: the viewport annotator returns a zeros array for the first few
        # frames, so step a couple of times (rendering, not capturing) before counting.
        for _ in range(3):
            with torch.inference_mode():
                env.step(torch.zeros(env.action_space.shape, device=env.unwrapped.device))
            env.render()

        num_frames = 2
        for _ in range(num_frames):
            with torch.inference_mode():
                env.step(torch.zeros(env.action_space.shape, device=env.unwrapped.device))
            viz.capture()

        # grid is always appended; global is skipped if env.render() ever returns
        # None, so assert it produced at least one frame rather than an exact count.
        assert len(viz._buffers.grid_frames) == num_frames
        assert len(viz._buffers.global_frames) >= 1
        # capture_per_env=True: one retained stream per env, each num_frames long.
        assert set(viz._buffers.per_env_frames) == set(range(env.unwrapped.scene.num_envs))
        assert all(len(f) == num_frames for f in viz._buffers.per_env_frames.values())
        # 2 envs -> ceil(sqrt(2)) = 2 cols x 1 row of (height x width) tiles.
        # Asserting the composed size verifies tiling against real camera output
        # without depending on scene lighting (an any() check can flake).
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

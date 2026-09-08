# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Arena's frame-aware viewport video recorder."""

import numpy as np
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from isaaclab.envs.utils.video_recorder import VideoRecorder

from isaaclab_arena.video.viewport_video_recorder import (
    ArenaViewportVideoRecorder,
    ArenaViewportVideoRecorderCfg,
    _define_camera_with_matching_intrinsics,
    resolve_viewer_camera_pose,
)


def test_report_camera_copies_viewport_intrinsics_without_copying_its_transform():
    """The isolated report camera retains the viewport's field of view and optical settings."""
    source_attributes = {
        "focalLength": 18.14756,
        "horizontalAperture": 20.955,
        "verticalAperture": 11.784375,
        "clippingRange": (0.01, 1_000_000.0),
    }
    source_prim = MagicMock()
    source_prim.IsValid.return_value = True
    source_prim.GetAttribute.side_effect = lambda name: SimpleNamespace(
        IsValid=lambda: name in source_attributes,
        Get=lambda: source_attributes.get(name),
    )
    destination_attributes = {}
    destination_prim = MagicMock()
    destination_prim.GetAttribute.side_effect = lambda name: SimpleNamespace(
        Set=lambda value: destination_attributes.__setitem__(name, value)
    )
    stage = MagicMock()
    stage.GetPrimAtPath.return_value = source_prim
    stage.DefinePrim.return_value = destination_prim

    _define_camera_with_matching_intrinsics(stage, "/OmniverseKit_Persp", "/World/ArenaReportCamera")

    stage.DefinePrim.assert_called_once_with("/World/ArenaReportCamera", "Camera")
    assert destination_attributes == source_attributes


def test_env_relative_camera_uses_selected_parallel_environment_origin():
    """Environment-relative camera coordinates include the selected clone's world offset."""
    scene = SimpleNamespace(
        env_origins=np.array([
            [15.0, -15.0, 0.0],
            [15.0, 15.0, 0.0],
            [-15.0, -15.0, 0.0],
            [-15.0, 15.0, 0.0],
        ])
    )
    cfg = ArenaViewportVideoRecorderCfg(viewer_origin_type="env", viewer_env_index=0)

    eye, lookat = resolve_viewer_camera_pose(
        scene,
        cfg,
        eye=np.array([-0.9, -1.3, 1.6]),
        lookat=np.array([0.6, 0.2, 0.1]),
    )

    assert np.allclose(eye, (14.1, -16.3, 1.6))
    assert np.allclose(lookat, (15.6, -14.8, 0.1))


def test_world_relative_camera_does_not_apply_an_environment_origin():
    """World-relative camera coordinates remain unchanged for parallel scenes."""
    scene = SimpleNamespace(env_origins=np.array([[15.0, -15.0, 0.0]]))
    cfg = ArenaViewportVideoRecorderCfg(viewer_origin_type="world")

    eye, lookat = resolve_viewer_camera_pose(
        scene,
        cfg,
        eye=np.array([1.0, 2.0, 3.0]),
        lookat=np.array([4.0, 5.0, 6.0]),
    )

    assert eye == (1.0, 2.0, 3.0)
    assert lookat == (4.0, 5.0, 6.0)


def test_kit_recording_uses_dedicated_camera_without_moving_viewport(monkeypatch):
    """Kit report frames update ArenaReportCamera rather than OmniverseKit_Persp."""

    class _Capture:
        def __init__(self):
            self.cfg = SimpleNamespace(camera_prim_path="/OmniverseKit_Persp", eye=None, lookat=None)

        def render_rgb_array(self):
            return np.zeros((2, 2, 3), dtype=np.uint8)

    def _fake_video_recorder_init(self, cfg, scene):
        # Mirror the base recorder replacing the task camera from a visualizer cfg.
        cfg.eye = (4.0, -4.0, 3.0)
        cfg.lookat = (0.0, 0.0, 0.0)
        self.cfg = cfg
        self._scene = scene
        self._backend = "kit"
        self._capture = _Capture()
        self._matched_visualizer = "kit"

    monkeypatch.setattr(VideoRecorder, "__init__", _fake_video_recorder_init)
    scene = SimpleNamespace(env_origins=np.array([[10.0, 20.0, 0.0]]), stage=MagicMock())
    cfg = ArenaViewportVideoRecorderCfg(
        eye=(1.0, 2.0, 3.0),
        lookat=(0.0, 0.0, 1.0),
        viewer_origin_type="env",
        viewer_env_index=0,
    )
    recorder = ArenaViewportVideoRecorder(cfg, scene)

    with patch("isaaclab_physx.renderers.kit_viewport_utils.set_kit_renderer_camera_view") as set_camera_view:
        frame = recorder.render_rgb_array()

    scene.stage.DefinePrim.assert_called_once_with("/World/ArenaReportCamera", "Camera")
    set_camera_view.assert_called_once_with(
        eye=(11.0, 22.0, 3.0),
        target=(10.0, 20.0, 1.0),
        camera_prim_path="/World/ArenaReportCamera",
    )
    assert recorder._capture.cfg.camera_prim_path == "/World/ArenaReportCamera"
    assert frame.shape == (2, 2, 3)

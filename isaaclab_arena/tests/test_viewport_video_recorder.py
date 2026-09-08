# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from isaaclab.envs.utils.video_recorder import VideoRecorder

from isaaclab_arena.video.viewport_video_recorder import (
    ArenaViewportVideoRecorder,
    ArenaViewportVideoRecorderCfg,
    _define_report_camera,
    _set_report_camera_pose,
    resolve_viewer_camera_pose,
)


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


def test_report_camera_uses_selected_environment_scene_partition():
    """The dedicated camera sees geometry from the selected parallel environment."""
    from pxr import Sdf, Usd

    stage = Usd.Stage.CreateInMemory()
    env_prim = stage.DefinePrim("/World/envs/env_2")
    env_prim.CreateAttribute("primvars:omni:scenePartition", Sdf.ValueTypeNames.Token).Set("env_2")

    camera_prim = _define_report_camera(stage, "/World/ArenaReportCamera", env_index=2)

    assert camera_prim.GetAttribute("omni:scenePartition").Get() == "env_2"


def test_report_camera_pose_is_authored_without_a_viewport():
    """Headless recording authors the dedicated USD camera without an active viewport."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()

    _set_report_camera_pose(
        stage,
        "/World/ArenaReportCamera",
        eye=(1.0, 2.0, 3.0),
        lookat=(0.0, 0.0, 1.0),
    )

    camera_prim = stage.GetPrimAtPath("/World/ArenaReportCamera")
    transform = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    assert np.allclose(transform.ExtractTranslation(), (1.0, 2.0, 3.0))
    assert not stage.GetPrimAtPath("/OmniverseKit_Persp").IsValid()


def test_kit_recording_uses_dedicated_camera_without_moving_viewport(monkeypatch):
    """Kit report frames update ArenaReportCamera rather than OmniverseKit_Persp."""

    def _fake_video_recorder_init(self, cfg, env):
        self.cfg = cfg
        self._env = env

    monkeypatch.setattr(VideoRecorder, "__init__", _fake_video_recorder_init)
    scene = SimpleNamespace(env_origins=np.array([[10.0, 20.0, 0.0]]), stage=MagicMock())
    env = SimpleNamespace(scene=scene)
    cfg = ArenaViewportVideoRecorderCfg(
        eye=(1.0, 2.0, 3.0),
        lookat=(0.0, 0.0, 1.0),
        window_width=2,
        window_height=2,
        viewer_origin_type="env",
        viewer_env_index=0,
    )
    annotator = MagicMock()
    annotator.get_data.return_value = np.zeros((2, 2, 4), dtype=np.uint8)
    render_product = MagicMock()
    settings = MagicMock()
    settings.get.side_effect = lambda key, default=None: {
        "/isaaclab/has_gui": False,
        "/app/player/playSimulations": True,
    }.get(key, default)
    replicator = SimpleNamespace(
        create=SimpleNamespace(render_product=MagicMock(return_value=render_product)),
        AnnotatorRegistry=SimpleNamespace(get_annotator=MagicMock(return_value=annotator)),
    )

    with (
        patch("isaaclab_arena.video.viewport_video_recorder._define_report_camera") as define_report_camera,
        patch("isaaclab_arena.video.viewport_video_recorder._set_report_camera_pose") as set_report_camera_pose,
        patch.dict(
            sys.modules,
            {
                "omni.replicator": SimpleNamespace(core=replicator),
                "omni.replicator.core": replicator,
            },
        ),
        patch("omni.kit.app.get_app") as get_app,
        patch("isaaclab.app.settings_manager.get_settings_manager", return_value=settings),
    ):
        recorder = ArenaViewportVideoRecorder(cfg, env)
        frame = recorder.render_rgb_array()
        recorder.render_rgb_array()

    define_report_camera.assert_called_once_with(scene.stage, "/World/ArenaReportCamera", 0)
    assert set_report_camera_pose.call_args_list == [
        ((scene.stage, "/World/ArenaReportCamera", (11.0, 22.0, 3.0), (10.0, 20.0, 1.0)),),
        ((scene.stage, "/World/ArenaReportCamera", (11.0, 22.0, 3.0), (10.0, 20.0, 1.0)),),
    ]
    assert get_app.return_value.update.call_count == 2
    annotator.attach.assert_called_once_with([render_product])
    render_product.resume.assert_called_once_with()
    assert render_product.pause.call_count == 2
    assert frame.shape == (2, 2, 3)

# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from isaaclab_arena.video import video_recording, viewport_video_recorder


def test_viewport_recorder_writes_report_compatible_episode_filename(monkeypatch, tmp_path):
    captured = {}
    viewer = SimpleNamespace(
        eye=(1.0, 2.0, 3.0),
        lookat=(0.0, 0.0, 1.0),
        resolution=(640, 480),
        origin_type="env",
        env_index=2,
        asset_name=None,
        body_name=None,
    )
    base_env = SimpleNamespace(
        max_episode_length=10,
        cfg=SimpleNamespace(viewer=viewer),
        video_recorders=[],
    )
    env = SimpleNamespace(unwrapped=base_env)

    def recorder(recorder_cfg, recorder_env):
        captured["cfg"] = recorder_cfg
        captured["env"] = recorder_env
        return "viewport-recorder"

    monkeypatch.setattr(viewport_video_recorder, "ArenaViewportVideoRecorder", recorder)
    video_cfg = video_recording.VideoRecordingCfg(
        record_viewport_video=True,
        video_base_dir=str(tmp_path),
        viewport_name_prefix="viewport-rebuild2-env0-viewport",
    )

    assert video_recording.wrap_env_for_video(env, video_cfg, num_steps=20, num_episodes=None) is env
    assert captured["env"] is base_env
    assert captured["cfg"].output_filename_prefix == "viewport-rebuild2-env0-viewport"
    assert captured["cfg"].video_length == 20
    assert captured["cfg"].eye == (1.0, 2.0, 3.0)
    assert captured["cfg"].viewer_env_index == 2
    assert base_env.video_recorders == ["viewport-recorder"]

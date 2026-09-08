# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from isaaclab_arena.video import video_recording


def test_viewport_recorder_writes_report_compatible_episode_filename(monkeypatch, tmp_path):
    captured = {}
    env = SimpleNamespace(unwrapped=SimpleNamespace(max_episode_length=10))

    def record_video(wrapped_env, **kwargs):
        captured.update(kwargs)
        return wrapped_env

    monkeypatch.setattr(video_recording, "RecordVideo", record_video)
    video_cfg = video_recording.VideoRecordingCfg(
        record_viewport_video=True,
        video_base_dir=str(tmp_path),
        viewport_name_prefix="viewport-rebuild2-env0-viewport",
    )

    assert video_recording.wrap_env_for_video(env, video_cfg, num_steps=20, num_episodes=None) is env
    assert captured["name_prefix"] == "viewport-rebuild2-env0-viewport"
    assert captured["episode_trigger"](0)
    assert not captured["episode_trigger"](1)
    assert "step_trigger" not in captured
    assert captured["video_length"] == 20

# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from isaaclab_arena.video.video_recording import VideoRecordingCfg, configure_env_for_video


def _env_cfg():
    return SimpleNamespace(
        episode_length_s=10.0,
        decimation=2,
        sim=SimpleNamespace(dt=0.1),
        video_recorders=[],
    )


def test_configure_env_for_viewport_video_uses_native_recorder(tmp_path):
    env_cfg = _env_cfg()

    configure_env_for_video(
        env_cfg,
        VideoRecordingCfg(record_viewport_video=True, video_base_dir=str(tmp_path)),
        num_steps=25,
        num_episodes=None,
    )

    assert len(env_cfg.video_recorders) == 1
    recorder_cfg = env_cfg.video_recorders[0]
    assert recorder_cfg.source == "visualizer:kit"
    assert recorder_cfg.output_dir == str(tmp_path)
    assert recorder_cfg.output_filename_prefix == "viewport"
    assert recorder_cfg.video_length == 25


def test_configure_env_for_viewport_video_sizes_episode_rollout(tmp_path):
    env_cfg = _env_cfg()

    configure_env_for_video(
        env_cfg,
        VideoRecordingCfg(record_viewport_video=True, video_base_dir=str(tmp_path)),
        num_steps=None,
        num_episodes=3,
    )

    assert env_cfg.video_recorders[0].video_length == 150

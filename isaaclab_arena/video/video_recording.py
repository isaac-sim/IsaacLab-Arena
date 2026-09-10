# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import datetime
import math
import os


@dataclasses.dataclass
class VideoRecordingCfg:
    """Options describing which rollout video recorders to enable and where to write them."""

    record_viewport_video: bool = False
    """Record the kit viewport (third-person scene view) via ``env.render()``."""

    record_camera_video: bool = False
    """Record the embodiment-mounted cameras from ``obs['camera_obs']``."""

    video_base_dir: str = "videos"
    """Base directory the mp4s are written to (a reverse-dated run subdirectory is added per run)."""

    camera_name_prefix: str = "robot-cam"
    """Filename prefix for the per-camera mp4s written by ``CameraObsVideoRecorder``."""

    @property
    def enabled(self) -> bool:
        """Whether any recorder is requested."""
        return self.record_viewport_video or self.record_camera_video


def timestamped_run_dir(base_dir: str) -> str:
    """Append a reverse-dated subdirectory to ``base_dir``, e.g. ``base_dir/2026-06-16_14-42-54``.

    Mirrors Isaac Lab's log layout so repeated runs land in distinct folders. Call once per run and
    share the result across recorders (and, for the Experiment Runner, across jobs).
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(base_dir, timestamp)


def _resolve_video_length(env_cfg, num_steps: int | None, num_episodes: int | None) -> int:
    """Number of env steps to record from the rollout limit and environment config.

    ``episode_length_s / (sim.dt * decimation)`` is the configured maximum episode
    length in environment steps.
    """
    if num_steps is not None:
        return num_steps
    assert num_episodes is not None, "Cannot determine video length: both num_steps and num_episodes are None."
    max_episode_length = math.ceil(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))
    return num_episodes * max_episode_length


def configure_env_for_video(
    env_cfg,
    video_cfg: VideoRecordingCfg,
    num_steps: int | None,
    num_episodes: int | None,
) -> None:
    """Add requested native video recorders to an Isaac Lab environment config."""
    if not video_cfg.record_viewport_video:
        return

    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

    os.makedirs(video_cfg.video_base_dir, exist_ok=True)
    video_length = _resolve_video_length(env_cfg, num_steps, num_episodes)
    env_cfg.video_recorders.append(
        VideoRecorderCfg(
            source="visualizer:kit",
            output_dir=video_cfg.video_base_dir,
            output_filename_prefix="viewport",
            video_length=video_length,
        )
    )
    print(f"Recording {video_length}-step viewport video to: {video_cfg.video_base_dir}")


def wrap_env_for_video(
    env,
    video_cfg: VideoRecordingCfg,
    num_steps: int | None,
    num_episodes: int | None,
):
    """Wrap ``env`` with the camera-observation recorder when requested.

    Viewport recording is configured natively before environment construction by
    :func:`configure_env_for_video`.

    Args:
        env: The env to wrap.
        video_cfg: The video recording configuration struct.
        num_steps: Unused; retained for call-site compatibility.
        num_episodes: Unused; retained for call-site compatibility.
    """
    if not video_cfg.record_camera_video:
        return env

    os.makedirs(video_cfg.video_base_dir, exist_ok=True)

    # Record the embodiment-mounted cameras (from obs["camera_obs"]),
    # flushed at each episode reset rather than after a fixed number of steps.
    if video_cfg.record_camera_video:
        from isaaclab_arena.video.camera_observation_video_recorder import CameraObsVideoRecorder

        env = CameraObsVideoRecorder(
            env,
            video_folder=video_cfg.video_base_dir,
            name_prefix=video_cfg.camera_name_prefix,
        )
        print(f"Recording per-episode per-camera videos to: {video_cfg.video_base_dir}")

    return env

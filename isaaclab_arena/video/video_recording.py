# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import datetime
import os
from gymnasium.wrappers import RecordVideo

from isaaclab_arena.video.camera_observation_video_recorder import CameraObsVideoRecorder


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

    @property
    def render_mode(self) -> str | None:
        """The ``render_mode`` the env must be built with to capture the viewport video."""
        return "rgb_array" if self.record_viewport_video else None


def timestamped_run_dir(base_dir: str) -> str:
    """Append a reverse-dated subdirectory to ``base_dir``, e.g. ``base_dir/2026-06-16_14-42-54``.

    Mirrors Isaac Lab's log layout so repeated runs land in distinct folders. Call once per run and
    share the result across recorders (and, for the eval runner, across jobs).
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(base_dir, timestamp)


def _resolve_video_length(env, num_steps: int | None, num_episodes: int | None) -> int:
    """Number of env steps to record: the step budget, or one episode's worth per episode.

    ``max_episode_length`` is in environment steps, which matches the rollout cadence.
    """
    if num_steps is not None:
        return num_steps
    assert num_episodes is not None, "Cannot determine video length: both num_steps and num_episodes are None."
    return num_episodes * env.unwrapped.max_episode_length


def wrap_env_for_video(
    env,
    video_cfg: VideoRecordingCfg,
    num_steps: int | None,
    num_episodes: int | None,
):
    """Wrap ``env`` with the recorders enabled in ``video_cfg`` and return the wrapped env.

    Returns ``env`` unchanged when no recorder is requested. ``num_steps`` and ``num_episodes``
    are mutually exclusive and size the viewport video.

    Args:
        env: The env to wrap.
        video_cfg: The video recording configuration struct.
        num_steps: Step budget for the rollout, or ``None`` when episode-driven.
        num_episodes: Episode budget for the rollout, or ``None`` when step-driven.
    """
    if not video_cfg.enabled:
        return env

    os.makedirs(video_cfg.video_base_dir, exist_ok=True)

    # Record the kit viewport (via env.render()).
    if video_cfg.record_viewport_video:
        video_length = _resolve_video_length(env, num_steps, num_episodes)
        env = RecordVideo(
            env,
            video_folder=video_cfg.video_base_dir,
            step_trigger=lambda step: step == 0,
            video_length=video_length,
            disable_logger=True,
        )
        print(f"Recording {video_length}-step viewport video to: {video_cfg.video_base_dir}")

    # Record the embodiment-mounted cameras (from obs["camera_obs"]),
    # flushed at each episode reset rather than after a fixed number of steps.
    if video_cfg.record_camera_video:
        env = CameraObsVideoRecorder(
            env,
            video_folder=video_cfg.video_base_dir,
            name_prefix=video_cfg.camera_name_prefix,
        )
        print(f"Recording per-episode per-camera videos to: {video_cfg.video_base_dir}")

    return env

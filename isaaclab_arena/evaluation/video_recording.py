# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Central helpers for wrapping a rollout env with video recorders.

Two independent recordings are supported and may be active at the same time:

* ``video`` records the kit viewport (third-person scene view) via ``env.render()``
  using gymnasium's ``RecordVideo``.
* ``camera_video`` records the embodiment-mounted cameras from ``obs['camera_obs']``
  using ``CameraObsVideoRecorder``.

The runners (``policy_runner``, ``eval_runner``) build a ``VideoRecordingCfg`` from their
CLI/job options and apply it through ``wrap_env_for_video``, so the gym-wrapper plumbing lives
in one place instead of being duplicated inline.
"""

from __future__ import annotations

import dataclasses
import os
from gymnasium.wrappers import RecordVideo

from isaaclab_arena.evaluation.camera_video import CameraObsVideoRecorder


@dataclasses.dataclass
class VideoRecordingCfg:
    """Options describing which rollout video recorders to enable and where to write them."""

    video: bool = False
    """Record the kit viewport (third-person scene view) via ``env.render()``."""

    camera_video: bool = False
    """Record the embodiment-mounted cameras from ``obs['camera_obs']``."""

    video_dir: str = "videos"
    """Directory the mp4s are written to."""

    camera_name_prefix: str = "robot-cam"
    """Filename prefix for the per-camera mp4s written by ``CameraObsVideoRecorder``."""

    @property
    def enabled(self) -> bool:
        """Whether any recorder is requested."""
        return self.video or self.camera_video

    @property
    def render_mode(self) -> str | None:
        """The ``render_mode`` the env must be built with to capture the viewport video."""
        return "rgb_array" if self.video else None


def _resolve_video_length(env, num_steps: int | None, num_episodes: int | None) -> int:
    """Number of env steps to record: the step budget, or one episode's worth per episode.

    ``max_episode_length`` is in environment steps, which matches the rollout cadence.
    """
    if num_steps is not None:
        return num_steps
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
        env: The env to wrap (already built with ``video_cfg.render_mode`` when ``video`` is set).
        video_cfg: Which recorders to enable and where to write them.
        num_steps: Step budget for the rollout, or ``None`` when episode-driven.
        num_episodes: Episode budget for the rollout, or ``None`` when step-driven.
    """
    if not video_cfg.enabled:
        return env

    os.makedirs(video_cfg.video_dir, exist_ok=True)

    # --video records the kit viewport (via env.render()).
    if video_cfg.video:
        video_length = _resolve_video_length(env, num_steps, num_episodes)
        env = RecordVideo(
            env,
            video_folder=video_cfg.video_dir,
            step_trigger=lambda step: step == 0,
            video_length=video_length,
            disable_logger=True,
        )
        print(f"Recording {video_length}-step viewport video to: {video_cfg.video_dir}")

    # --camera_video records the embodiment-mounted cameras (from obs["camera_obs"]),
    # flushed at each episode reset rather than after a fixed number of steps.
    if video_cfg.camera_video:
        env = CameraObsVideoRecorder(
            env,
            video_folder=video_cfg.video_dir,
            name_prefix=video_cfg.camera_name_prefix,
        )
        print(f"Recording per-episode per-camera videos to: {video_cfg.video_dir}")

    return env

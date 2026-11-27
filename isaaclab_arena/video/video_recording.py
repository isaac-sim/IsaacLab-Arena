# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import datetime
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnvCfg


@dataclasses.dataclass
class VideoRecordingCfg:
    """Options describing which rollout video recorders to enable and where to write them."""

    record_viewport_video: bool = False
    """Record the Kit viewer (third-person scene view). Needs the Kit visualizer and camera support."""

    record_camera_video: bool = False
    """Record RGB embodiment-mounted cameras from ``obs['camera_obs']``."""

    video_base_dir: str = "videos"
    """Base directory the mp4s are written to (a reverse-dated run subdirectory is added per run)."""

    viewport_name_prefix: str = "viewport"
    """Filename prefix for the viewport mp4, which is written as ``<prefix>_0000.mp4``."""

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


def add_viewport_video_recorder(env_cfg: ManagerBasedEnvCfg, video_cfg: VideoRecordingCfg) -> None:
    """Add an Isaac Lab video recorder that writes the Kit viewer to one mp4 when the environment closes.

    The clip starts at the environment's first step and shows all of its parallel envs in the Kit viewer.
    Recording needs the Kit visualizer (``--viz kit``) and camera support (``--enable_cameras``).

    Args:
        env_cfg: Configuration of the environment to record, before the environment is made.
        video_cfg: The video recording configuration struct.
    """
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

    viewport_recorder_cfg = VideoRecorderCfg(
        source="visualizer:kit",
        output_dir=video_cfg.video_base_dir,
        output_filename_prefix=video_cfg.viewport_name_prefix,
        # The rollout length is known only once the policy exists, so the clip runs until the
        # environment closes, which writes it.
        video_length=sys.maxsize,
    )
    env_cfg.video_recorders = [*env_cfg.video_recorders, viewport_recorder_cfg]
    print(f"Recording the viewport video to: {video_cfg.video_base_dir}")


def wrap_env_for_video(env, video_cfg: VideoRecordingCfg):
    """Wrap ``env`` with a step timer and the requested camera recorder, and return the wrapped env.

    Returns ``env`` unchanged when no recorder is requested. The environment records the viewport video
    itself, through the recorder that ``add_viewport_video_recorder`` adds to its configuration.

    Args:
        env: The env to wrap.
        video_cfg: The video recording configuration struct.
    """
    if not video_cfg.enabled:
        return env

    os.makedirs(video_cfg.video_base_dir, exist_ok=True)

    # Wrap the env in a step timer before adding the camera recording, so the timer leaves out camera
    # recording. The viewport recorder runs inside env.step, so its frame captures are timed.
    from isaaclab_arena.utils.env_step_timer import EnvStepTimerWrapper

    env = EnvStepTimerWrapper(env, timer_name="sim_step")

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

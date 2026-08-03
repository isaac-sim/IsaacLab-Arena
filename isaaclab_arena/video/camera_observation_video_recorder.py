# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Gym wrapper that records one mp4 per (env, camera, episode) in ``obs['camera_obs']``.

Each frame is streamed straight to a per-(env, camera) ffmpeg encoder as it arrives, and
the file is finalised when that environment resets (terminated or truncated), so each
output file corresponds to exactly one complete episode. Partial episodes cut off by
``num_steps`` are deleted on ``close()``.

Output filename: ``<name_prefix>-env<N>-<camera_name>-episode-<E>.mp4``

policy_runner.py wraps the env with this alongside ``RecordVideo`` so
the kit viewport mp4 (third-person scene view) and the embodiment-
mounted camera mp4s (what the policy actually sees) are written
together when ``--record_viewport_video --record_camera_video`` is set.

Memory note: This class uses incremental ffmpeg encoding to avoid storing the raw frames in
memory, which uses substantial amounts of RAM.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import os
import re
import torch
from dataclasses import dataclass

from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

CAMERA_OBS_GROUP_KEY = "camera_obs"

# Regular expression to parse the filename of an episode video.
_EPISODE_VIDEO_FILENAME_PATTERN = re.compile(
    r"^(?P<prefix>.+?)(?:-rebuild(?P<rebuild>\d+))?-env(?P<env>\d+)-(?P<camera>.+)-episode-(?P<episode>\d+)\.mp4$"
)


@dataclass
class ParsedEpisodeVideoName:
    """The fields recovered from a recorder mp4 filename by ``parse_episode_video_filename``."""

    prefix: str
    env_index: int
    camera_name: str
    episode_index: int
    rebuild_index: int | None
    """The rebuild this video belongs to, or ``None`` when the prefix carried no ``-rebuild`` segment."""


def format_episode_video_filename(name_prefix: str, env_index: int, camera_name: str, episode_index: int) -> str:
    """Build the mp4 filename for one (env, camera, episode). Inverse of ``parse_episode_video_filename``."""
    return f"{name_prefix}-env{env_index}-{_sanitize_cam_key(camera_name)}-episode-{episode_index}.mp4"


def parse_episode_video_filename(filename: str) -> ParsedEpisodeVideoName | None:
    """Parse a recorder mp4 filename, or return ``None`` if it does not match the recorder's format."""
    match = _EPISODE_VIDEO_FILENAME_PATTERN.match(filename)
    if match is None:
        return None
    rebuild = match.group("rebuild")
    return ParsedEpisodeVideoName(
        prefix=match.group("prefix"),
        env_index=int(match.group("env")),
        camera_name=match.group("camera"),
        episode_index=int(match.group("episode")),
        rebuild_index=int(rebuild) if rebuild is not None else None,
    )


def _to_uint8(frame: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    if frame.dtype == np.uint8:
        return frame
    if frame.dtype.kind == "f":
        # mdp.image with normalize=True returns float in [0, 1]; rescale.
        scale = 255.0 if float(frame.max()) <= 1.0 else 1.0
        return np.clip(frame * scale, 0, 255).astype(np.uint8)
    return frame.astype(np.uint8)


def _sanitize_cam_key(camera_name: str) -> str:
    """Strip path separators so a camera name can't escape video_folder."""
    return camera_name.replace("/", "_").replace(os.sep, "_")


@dataclass
class EpisodeVideoWriter:
    """The open ffmpeg encoder for one (env, camera, episode) and the file it is writing."""

    writer: FFMPEG_VideoWriter
    """Encoder consuming raw frames; closing it finalises the mp4."""

    path: str
    """Destination the encoder is writing to, kept so a partial episode can be deleted."""


class CameraObsVideoRecorder(gym.Wrapper):
    """Record one mp4 per (env, camera, episode) in ``obs['camera_obs']``.

    Cameras are batched as ``[N_envs, H, W, C]``.  Each env is recorded
    independently; its encoder is finalised when that env resets (terminated
    or truncated), producing one file per completed episode:
    ``<name_prefix>-env<N>-<camera_name>-episode-<E>.mp4``.
    """

    def __init__(
        self,
        env: gym.Env,
        video_folder: str,
        name_prefix: str = "robot-cam",
        fps: int | None = None,
    ):
        super().__init__(env)
        os.makedirs(video_folder, exist_ok=True)
        self.video_folder = video_folder
        self.name_prefix = name_prefix
        self.fps = fps if fps is not None else int(env.metadata.get("render_fps", 30))

        # camera_name -> one entry per env, holding that env's open encoder for its current
        # episode, or None while no episode is in progress.
        self.writers: dict[str, list[EpisodeVideoWriter | None]] = {}

    def step(self, action):
        result = self.env.step(action)
        obs, _, terminated, truncated, _ = result
        cam_obs = obs.get(CAMERA_OBS_GROUP_KEY, {}) if isinstance(obs, dict) else {}

        if cam_obs:
            n_envs = next(iter(cam_obs.values())).shape[0]

            # Determine done envs before appending frames. Isaac Lab auto-resets on
            # termination, so the obs returned for a done env is the post-reset first
            # frame of the new episode — discard it so it doesn't contaminate the
            # current episode. This means each recorded episode is missing its first
            # frame, which is acceptable given episodes are typically hundreds of steps.
            done_envs = (terminated | truncated).nonzero().flatten().tolist()
            done_set = set(done_envs)

            for camera_name, frames in cam_obs.items():
                if camera_name not in self.writers:
                    self.writers[camera_name] = [None] * n_envs
                for env_idx in range(n_envs):
                    if env_idx not in done_set:
                        self._write_frame(camera_name, env_idx, _to_uint8(frames[env_idx]))

            if done_envs:
                self._finish_envs(done_envs)

        return result

    def _write_frame(self, camera_name: str, env_idx: int, frame: np.ndarray) -> None:
        """Append one frame to this (env, camera)'s episode video, opening the encoder if needed."""
        assert frame.ndim == 3 and frame.shape[2] == 3, (
            f"Camera '{camera_name}' produced a frame of shape {frame.shape}; expected (H, W, 3) RGB."
            " The encoder is configured for 3-channel input."
        )
        episode_writer = self.writers[camera_name][env_idx]
        if episode_writer is None:
            # The env's counter still names the episode now in progress; it is advanced on reset,
            # inside env.step.
            episode_num = self.unwrapped.get_episode_index(env_idx)
            path = os.path.join(
                self.video_folder,
                format_episode_video_filename(self.name_prefix, env_idx, camera_name, episode_num),
            )
            height, width, _ = frame.shape
            # We use one thread because frames arrive slower than a single thread is able to encode.
            episode_writer = EpisodeVideoWriter(
                writer=FFMPEG_VideoWriter(path, size=(width, height), fps=self.fps, threads=1),
                path=path,
            )
            self.writers[camera_name][env_idx] = episode_writer
        episode_writer.writer.write_frame(frame)

    def _finish_envs(self, env_ids: list[int]) -> None:
        """Finalise the mp4 of every stream open for each env that just reset."""
        for env_idx in env_ids:
            for env_writers in self.writers.values():
                episode_writer = env_writers[env_idx]
                if episode_writer is None:
                    continue
                episode_writer.writer.close()
                env_writers[env_idx] = None

    def close(self) -> None:
        # Partial episodes (cut off by num_steps rather than a real reset) are discarded: the
        # encoder is shut down and the incomplete file it was writing is removed.
        for env_writers in self.writers.values():
            for env_idx, episode_writer in enumerate(env_writers):
                if episode_writer is None:
                    continue
                episode_writer.writer.close()
                if os.path.exists(episode_writer.path):
                    os.remove(episode_writer.path)
                env_writers[env_idx] = None
        self.env.close()

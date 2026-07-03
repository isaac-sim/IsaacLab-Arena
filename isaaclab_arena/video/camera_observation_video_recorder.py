# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Gym wrapper that records one mp4 per (env, camera, episode) in ``obs['camera_obs']``.

Frames are streamed to disk and atomically published each time an environment
resets (terminated or truncated), so each output file corresponds to exactly
one complete episode.
Partial episodes cut off by ``num_steps`` are discarded on ``close()``.

Output filename: ``<name_prefix>-env<N>-<camera_name>-episode-<E>.mp4``

policy_runner.py wraps the env with this alongside ``RecordVideo`` so
the kit viewport mp4 (third-person scene view) and the embodiment-
mounted camera mp4s (what the policy actually sees) are written
together when ``--record_viewport_video --record_camera_video`` is set.

Each active camera uses one ffmpeg pipe. Frames are encoded incrementally, so
recorder memory is bounded by the active streams and encoder pipe buffers
rather than growing with episode length.
"""

from __future__ import annotations

import contextlib
import gymnasium as gym
import numpy as np
import os
import re
import torch
from dataclasses import dataclass

import imageio_ffmpeg

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


@dataclass
class _ActiveWriter:
    """One incrementally encoded, not-yet-published episode video."""

    writer: object
    temporary_path: str
    final_path: str


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


class CameraObsVideoRecorder(gym.Wrapper):
    """Record one mp4 per (env, camera, episode) in ``obs['camera_obs']``.

    Cameras are batched as ``[N_envs, H, W, C]``.  Each env is recorded
    independently; its buffer is flushed when that env resets (terminated
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

        # (camera_name, env_idx) -> active encoder for the current episode.
        self._writers: dict[tuple[str, int], _ActiveWriter] = {}

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
                for env_idx in range(n_envs):
                    if env_idx not in done_set:
                        self._write_frame(camera_name, env_idx, _to_uint8(frames[env_idx]))

            if done_envs:
                self._flush_envs(done_envs)

        return result

    def _write_frame(self, camera_name: str, env_idx: int, frame: np.ndarray) -> None:
        key = (camera_name, env_idx)
        active = self._writers.get(key)
        if active is None:
            active = self._open_writer(camera_name, env_idx, frame)
            self._writers[key] = active
        try:
            active.writer.send(np.ascontiguousarray(frame))
        except BaseException:
            self._discard_writer(key, active)
            raise

    def _open_writer(self, camera_name: str, env_idx: int, frame: np.ndarray) -> _ActiveWriter:
        if frame.ndim == 2:
            height, width = frame.shape
            pixel_format = "gray"
        elif frame.ndim == 3 and frame.shape[2] in (1, 3, 4):
            height, width, channels = frame.shape
            pixel_format = {1: "gray", 3: "rgb24", 4: "rgba"}[channels]
        else:
            raise ValueError(f"Camera frame must be HxW, HxWx1, HxWx3, or HxWx4; got {frame.shape}")

        episode_num = self.unwrapped.get_episode_index(env_idx)
        final_path = os.path.join(
            self.video_folder,
            format_episode_video_filename(self.name_prefix, env_idx, camera_name, episode_num),
        )
        temporary_path = f"{final_path}.part"
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary_path)
        writer = imageio_ffmpeg.write_frames(
            temporary_path,
            (width, height),
            pix_fmt_in=pixel_format,
            fps=self.fps,
            macro_block_size=1,
            output_params=["-f", "mp4"],
        )
        try:
            writer.send(None)
        except BaseException:
            try:
                writer.close()
            finally:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(temporary_path)
            raise
        return _ActiveWriter(writer=writer, temporary_path=temporary_path, final_path=final_path)

    def _flush_envs(self, env_ids: list[int]) -> None:
        for env_idx in env_ids:
            keys = [key for key in self._writers if key[1] == env_idx]
            for key in keys:
                active = self._writers.pop(key)
                try:
                    active.writer.close()
                    if not os.path.isfile(active.temporary_path) or os.path.getsize(active.temporary_path) == 0:
                        raise RuntimeError(f"ffmpeg produced no video: {active.temporary_path}")
                    os.replace(active.temporary_path, active.final_path)
                except BaseException:
                    with contextlib.suppress(FileNotFoundError):
                        os.unlink(active.temporary_path)
                    raise

    def _discard_writer(self, key: tuple[str, int], active: _ActiveWriter) -> None:
        self._writers.pop(key, None)
        try:
            active.writer.close()
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(active.temporary_path)

    def close(self) -> None:
        # Partial episodes (cut off by num_steps rather than a real reset) are discarded.
        try:
            for key, active in list(self._writers.items()):
                self._discard_writer(key, active)
        finally:
            self.env.close()

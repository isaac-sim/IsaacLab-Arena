# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Gym wrapper that records one mp4 per camera in ``obs['camera_obs']``.

Mirrors ``gymnasium.wrappers.RecordVideo`` — same ``step_trigger`` /
``video_length`` semantics and the same moviepy ``ImageSequenceClip``
encoder, but frames come from ``obs["camera_obs"][<cam>]`` rather than
``env.render()``. Output is one mp4 per camera under ``video_folder``,
named ``<name_prefix>-<cam>-step-<N>.mp4``.

policy_runner.py wraps the env with this alongside ``RecordVideo`` so
the kit viewport mp4 (third-person scene view) and the embodiment-
mounted camera mp4s (what the policy actually sees) are written
together when ``--video --camera_video`` is set.

Multi-env note: Arena batches camera observations as
``[N_envs, H, W, C]``. This wrapper records env 0 only (``frame[0]``);
other envs are silently dropped. ``RecordVideo`` for the kit viewport
is unaffected — the viewport shows the full scene regardless of
``num_envs``.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import os
import torch
from collections.abc import Callable

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

CAMERA_OBS_GROUP_KEY = "camera_obs"


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


def _sanitize_cam_key(key: str) -> str:
    """Strip path separators so a camera key can't escape video_folder."""
    return key.replace("/", "_").replace(os.sep, "_")


class CameraObsVideoRecorder(gym.Wrapper):
    """Record an mp4 per camera in ``obs['camera_obs']``.

    Records env 0 only: cameras are batched as ``[N_envs, H, W, C]`` and the
    wrapper picks ``frame[0]``. Intended for single-env evaluation rollouts.
    """

    def __init__(
        self,
        env: gym.Env,
        video_folder: str,
        step_trigger: Callable[[int], bool],
        video_length: int,
        name_prefix: str = "robot-cam",
        fps: int | None = None,
    ):
        super().__init__(env)
        os.makedirs(video_folder, exist_ok=True)
        self.video_folder = video_folder
        self.step_trigger = step_trigger
        self.video_length = video_length
        self.name_prefix = name_prefix
        self.fps = fps if fps is not None else int(env.metadata.get("render_fps", 30))

        self.step_id = -1
        self.recording = False
        self.recording_start_step = 0
        self.buffers: dict[str, list[np.ndarray]] = {}

    def step(self, action):
        result = self.env.step(action)
        self.step_id += 1
        obs = result[0]
        cam_obs = obs.get(CAMERA_OBS_GROUP_KEY, {}) if isinstance(obs, dict) else {}

        if not self.recording and self.step_trigger(self.step_id):
            self.recording = True
            self.recording_start_step = self.step_id
            self.buffers = {k: [] for k in cam_obs}

        if self.recording and cam_obs:
            for k, frame in cam_obs.items():
                # frame: [N_envs, H, W, C]; record env 0 only.
                self.buffers.setdefault(k, []).append(_to_uint8(frame[0]))
            if self.buffers and all(len(v) >= self.video_length for v in self.buffers.values()):
                self._flush()

        return result

    def _flush(self) -> None:
        for cam, frames in self.buffers.items():
            if not frames:
                continue
            path = os.path.join(
                self.video_folder,
                f"{self.name_prefix}-{_sanitize_cam_key(cam)}-step-{self.recording_start_step}.mp4",
            )
            clip = ImageSequenceClip(list(frames), fps=self.fps)
            clip.write_videofile(path, logger=None, audio=False)
            del clip
        self.recording = False
        self.buffers = {}

    def close(self) -> None:
        try:
            if self.recording and any(len(v) > 0 for v in self.buffers.values()):
                self._flush()
        finally:
            self.env.close()

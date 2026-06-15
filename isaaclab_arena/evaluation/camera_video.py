# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Gym wrapper that records one mp4 per (env, camera, episode) in ``obs['camera_obs']``.

Frames are flushed to disk each time an environment resets (terminated or
truncated), so each output file corresponds to exactly one episode.  A final
flush on ``close()`` captures any trailing incomplete episode.

Output filename: ``<name_prefix>-env<N>-<cam>-episode-<E>.mp4``

policy_runner.py wraps the env with this alongside ``RecordVideo`` so
the kit viewport mp4 (third-person scene view) and the embodiment-
mounted camera mp4s (what the policy actually sees) are written
together when ``--video --camera_video`` is set.

Memory note: each env buffers raw uint8 frames for its current episode before
encoding.  Peak RAM is N×L×H×W×C bytes where L is max episode length, not the
full rollout.  For 10 envs, 500-step episodes, 512×512×3 frames that is ~3.8 GB
of raw frames — the encoded mp4s are far smaller (H.264 compresses ~100:1).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import os
import torch

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
    """Record one mp4 per (env, camera, episode) in ``obs['camera_obs']``.

    Cameras are batched as ``[N_envs, H, W, C]``.  Each env is recorded
    independently; its buffer is flushed when that env resets (terminated
    or truncated), producing one file per completed episode:
    ``<name_prefix>-env<N>-<cam>-episode-<E>.mp4``.
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

        # cam_key -> list of per-env frame lists: buffers[cam][env_idx] = [frame, ...]
        self.buffers: dict[str, list[list[np.ndarray]]] = {}
        # How many episodes have been flushed for each env.
        self.episode_counts: list[int] = []
        self._n_envs: int | None = None

    def step(self, action):
        result = self.env.step(action)
        obs, _, terminated, truncated, _ = result
        cam_obs = obs.get(CAMERA_OBS_GROUP_KEY, {}) if isinstance(obs, dict) else {}

        if cam_obs:
            frames_sample = next(iter(cam_obs.values()))
            n_envs = frames_sample.shape[0]

            if self._n_envs is None:
                self._n_envs = n_envs
                self.episode_counts = [0] * n_envs

            # Determine done envs before appending frames. Isaac Lab auto-resets on
            # termination, so the obs returned for a done env is the post-reset first
            # frame of the new episode — discard it so it doesn't contaminate the
            # current episode or create a phantom one-frame episode on close().
            if isinstance(terminated, torch.Tensor):
                done_envs = (terminated | truncated).nonzero().flatten().tolist()
            else:
                done_envs = [0] if (terminated or truncated) else []
            done_set = set(done_envs)

            for k, frames in cam_obs.items():
                if k not in self.buffers:
                    self.buffers[k] = [[] for _ in range(n_envs)]
                for env_idx in range(n_envs):
                    if env_idx not in done_set:
                        self.buffers[k][env_idx].append(_to_uint8(frames[env_idx]))

            if done_envs:
                self._flush_envs(done_envs)

        return result

    def _flush_envs(self, env_ids: list[int]) -> None:
        for env_idx in env_ids:
            episode_num = self.episode_counts[env_idx]
            wrote_any = False
            for cam, env_frame_lists in self.buffers.items():
                frames = env_frame_lists[env_idx]
                if not frames:
                    continue
                path = os.path.join(
                    self.video_folder,
                    f"{self.name_prefix}-env{env_idx}-{_sanitize_cam_key(cam)}-episode-{episode_num}.mp4",
                )
                clip = ImageSequenceClip(list(frames), fps=self.fps)
                clip.write_videofile(path, logger=None, audio=False)
                del clip
                env_frame_lists[env_idx] = []
                wrote_any = True
            if wrote_any:
                self.episode_counts[env_idx] += 1

    def close(self) -> None:
        try:
            if self._n_envs is not None and self.buffers:
                envs_with_frames = [
                    i for i in range(self._n_envs) if any(len(self.buffers[cam][i]) > 0 for cam in self.buffers)
                ]
                if envs_with_frames:
                    self._flush_envs(envs_with_frames)
        finally:
            self.env.close()

# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for CameraObsVideoRecorder.

No Isaac Sim or GPU required. The moviepy encoder is replaced by a stand-in that encodes
nothing and only counts the frames it is handed, so tests run fast and on CPU-only machines.
"""

import contextlib
import gymnasium as gym
import os
import shutil
import torch
from unittest.mock import patch

import pytest

from isaaclab_arena.video.camera_observation_video_recorder import CAMERA_OBS_GROUP_KEY, CameraObsVideoRecorder

# ---------------------------------------------------------------------------
# Minimal gym.Env stub — satisfies gymnasium.Wrapper's isinstance check
# ---------------------------------------------------------------------------

H, W, C = 4, 4, 3
CAMERAS = ["front", "wrist"]


class _StubEnv(gym.Env):
    metadata = {"render_fps": 30}
    observation_space = gym.spaces.Dict({})
    action_space = gym.spaces.Discrete(1)

    def __init__(self):
        super().__init__()
        self._step_return = ({}, None, torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), None)
        # Per-env completed-episode counts, mirroring the Arena env's centralized episode index.
        self._episode_counts: dict[int, int] = {}

    def reset(self, **kwargs):
        return {}, {}

    def step(self, action):
        # Mirror the Arena env: advance the per-env episode index for each env that resets this
        # step (the real env does this within _reset_idx, before step() returns).
        _, _, terminated, truncated, _ = self._step_return
        for env_id in (terminated | truncated).nonzero().flatten().tolist():
            self._episode_counts[env_id] = self._episode_counts.get(env_id, 0) + 1
        return self._step_return

    def get_episode_index(self, env_id: int) -> int:
        """The current episode index for ``env_id`` (its count of completed episodes)."""
        return self._episode_counts.get(env_id, 0)

    def render(self):
        pass


def _make_env() -> _StubEnv:
    return _StubEnv()


def _configure_step(env: _StubEnv, done_envs: list[int] | None = None, n_envs: int = 2):
    """Set the next step return value with given terminations."""
    terminated = torch.zeros(n_envs, dtype=torch.bool)
    for idx in done_envs or []:
        terminated[idx] = True
    truncated = torch.zeros(n_envs, dtype=torch.bool)
    cam_obs = {cam: torch.zeros(n_envs, H, W, C, dtype=torch.uint8) for cam in CAMERAS}
    obs = {CAMERA_OBS_GROUP_KEY: cam_obs}
    env._step_return = (obs, None, terminated, truncated, None)


class _FakeVideoWriter:
    """Stand-in for moviepy's FFMPEG_VideoWriter that counts frames and touches its file.

    Creating the file mirrors ffmpeg, so tests can assert that a partial episode's file is
    removed rather than left behind.
    """

    def __init__(self, filename, size, fps, **kwargs):
        self.filename = filename
        self.size = size
        self.fps = fps
        self.frames_written = 0
        self.closed = False
        with open(filename, "wb"):
            pass

    def write_frame(self, frame):
        self.frames_written += 1

    def close(self):
        self.closed = True


@contextlib.contextmanager
def _patched_writers():
    """Replace the encoder with ``_FakeVideoWriter`` and yield the instances created."""
    instances: list[_FakeVideoWriter] = []

    def factory(*args, **kwargs):
        writer = _FakeVideoWriter(*args, **kwargs)
        instances.append(writer)
        return writer

    with patch(
        "isaaclab_arena.video.camera_observation_video_recorder.FFMPEG_VideoWriter",
        side_effect=factory,
    ):
        yield instances


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_video_files_written_on_termination(tmp_path):
    """A file per camera is written when an env terminates."""
    env = _make_env()
    with _patched_writers() as writers:
        recorder = CameraObsVideoRecorder(env, video_folder=str(tmp_path))

        _configure_step(env)
        recorder.step(None)  # stream one frame

        _configure_step(env, done_envs=[0])
        recorder.step(None)  # env 0 terminates → finalise

        finalised = [writer.filename for writer in writers if writer.closed]
        assert len(finalised) == len(CAMERAS)
        for cam in CAMERAS:
            assert os.path.join(str(tmp_path), f"robot-cam-env0-{cam}-episode-0.mp4") in finalised


def test_frames_are_streamed_not_buffered(tmp_path):
    """Every frame reaches the encoder as it arrives, and no frame list is retained."""
    env = _make_env()
    with _patched_writers() as writers:
        recorder = CameraObsVideoRecorder(env, video_folder=str(tmp_path))

        for _ in range(3):
            _configure_step(env)
            recorder.step(None)

        # One open encoder per (env, camera), each already handed all three frames.
        assert len(writers) == len(CAMERAS) * 2
        assert all(writer.frames_written == 3 for writer in writers)


def test_episode_counter_increments_per_env(tmp_path):
    """Each env tracks its own episode count independently via the env's centralized index."""
    env = _make_env()
    with _patched_writers():
        recorder = CameraObsVideoRecorder(env, video_folder=str(tmp_path))

        _configure_step(env)
        recorder.step(None)

        _configure_step(env, done_envs=[0])
        recorder.step(None)  # env 0: episode 0 done

        _configure_step(env)
        recorder.step(None)

        _configure_step(env, done_envs=[0])
        recorder.step(None)  # env 0: episode 1 done

        _configure_step(env, done_envs=[1])
        recorder.step(None)  # env 1: episode 0 done

        assert env.get_episode_index(0) == 2
        assert env.get_episode_index(1) == 1


def test_multiple_episodes_produce_sequential_filenames(tmp_path):
    """Consecutive episodes for an env are named episode-0, episode-1, ..."""
    env = _make_env()

    with _patched_writers() as writers:
        recorder = CameraObsVideoRecorder(env, video_folder=str(tmp_path))

        for _ in range(3):
            _configure_step(env, n_envs=1)
            recorder.step(None)
            _configure_step(env, done_envs=[0], n_envs=1)
            recorder.step(None)

    written_paths = [writer.filename for writer in writers]
    for episode in range(3):
        for cam in CAMERAS:
            assert os.path.join(str(tmp_path), f"robot-cam-env0-{cam}-episode-{episode}.mp4") in written_paths


def test_partial_episode_dropped_on_close(tmp_path):
    """Frames streamed without a termination leave no file behind on close()."""
    env = _make_env()
    with _patched_writers() as writers:
        recorder = CameraObsVideoRecorder(env, video_folder=str(tmp_path))

        _configure_step(env)
        recorder.step(None)  # stream frames, no termination

        recorder.close()

        # Every encoder was shut down and its incomplete file removed.
        assert writers and all(writer.closed for writer in writers)
        assert list(tmp_path.iterdir()) == []


def test_no_video_written_for_empty_episode(tmp_path):
    """An env terminating with no recorded frames writes no video; its episode index still advances."""
    env = _make_env()
    with _patched_writers() as writers:
        recorder = CameraObsVideoRecorder(env, video_folder=str(tmp_path))

        # Terminate on the very first step — no prior frames were recorded.
        _configure_step(env, done_envs=[0])
        recorder.step(None)

        # No encoder was ever opened for the empty episode, but the env's centralized index still
        # advanced past it (so a later episode's video number stays in lockstep with the
        # per-episode results record).
        assert not any(writer.filename.endswith("env0-front-episode-0.mp4") for writer in writers)
        assert env.get_episode_index(0) == 1


def test_post_reset_frame_not_recorded(tmp_path):
    """The obs on a terminal step (post-reset) is not recorded into either episode."""
    env = _make_env()
    with _patched_writers() as writers:
        recorder = CameraObsVideoRecorder(env, video_folder=str(tmp_path))

        _configure_step(env)
        recorder.step(None)  # 1 frame recorded for both envs

        _configure_step(env, done_envs=[0])
        recorder.step(None)  # env 0 terminates; post-reset frame discarded

        writer_by_filename = {writer.filename: writer for writer in writers}
        for cam in CAMERAS:
            # env 0's episode 0 closed holding only the single pre-termination frame, and no
            # encoder was opened for its next episode.
            env0_episode0 = writer_by_filename[os.path.join(str(tmp_path), f"robot-cam-env0-{cam}-episode-0.mp4")]
            assert env0_episode0.frames_written == 1
            assert env0_episode0.closed
            assert os.path.join(str(tmp_path), f"robot-cam-env0-{cam}-episode-1.mp4") not in writer_by_filename

            # env 1 recorded 2 frames (neither step was terminal for it) and is still open.
            env1_episode0 = writer_by_filename[os.path.join(str(tmp_path), f"robot-cam-env1-{cam}-episode-0.mp4")]
            assert env1_episode0.frames_written == 2
            assert not env1_episode0.closed


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")
def test_real_video_file_written_on_termination(tmp_path):
    """An actual mp4 file appears on disk when an env terminates (requires ffmpeg)."""
    env = _make_env()
    recorder = CameraObsVideoRecorder(env, video_folder=str(tmp_path))

    _configure_step(env)
    recorder.step(None)

    _configure_step(env, done_envs=[0])
    recorder.step(None)

    for cam in CAMERAS:
        path = os.path.join(str(tmp_path), f"robot-cam-env0-{cam}-episode-0.mp4")
        assert os.path.isfile(path), f"Expected video file not found: {path}"
        assert os.path.getsize(path) > 0

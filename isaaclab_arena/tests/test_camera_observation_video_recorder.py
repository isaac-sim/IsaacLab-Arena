# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for CameraObsVideoRecorder.

No Isaac Sim or GPU required. The incremental ffmpeg writer is mocked so tests
run fast and on CPU-only machines.
"""

import gymnasium as gym
import os
import shutil
import torch
from contextlib import ExitStack
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


class _FakeWriter:
    def __init__(self, path: str):
        self.path = path
        self.frame_count = 0
        self.closed = False

    def send(self, frame):
        if frame is not None:
            self.frame_count += 1

    def close(self):
        self.closed = True
        with open(self.path, "wb") as stream:
            stream.write(b"fake-mp4")


class _FakeWriterFactory:
    def __init__(self):
        self.writers: list[_FakeWriter] = []

    def __call__(self, path, *args, **kwargs):
        writer = _FakeWriter(path)
        self.writers.append(writer)
        return writer


def _mock_writers():
    factory = _FakeWriterFactory()
    stack = ExitStack()
    stack.enter_context(
        patch(
            "isaaclab_arena.video.camera_observation_video_recorder.imageio_ffmpeg.write_frames",
            side_effect=factory,
        )
    )
    stack.enter_context(patch("isaaclab_arena.video.camera_observation_video_recorder._validate_encoded_video"))
    return factory, stack


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_video_files_written_on_termination(tmp_path):
    """A file per camera is written when an env terminates."""
    env = _make_env()
    factory, mocked = _mock_writers()
    with mocked:
        recorder = CameraObsVideoRecorder(env, video_folder=str(tmp_path))

        _configure_step(env)
        recorder.step(None)  # accumulate one frame

        _configure_step(env, done_envs=[0])
        recorder.step(None)  # env 0 terminates → flush

        assert len(factory.writers) == 2 * len(CAMERAS)
        for cam in CAMERAS:
            path = os.path.join(str(tmp_path), f"robot-cam-env0-{cam}-episode-0.mp4")
            assert os.path.isfile(path)


def test_episode_counter_increments_per_env(tmp_path):
    """Each env tracks its own episode count independently via the env's centralized index."""
    env = _make_env()
    _, mocked = _mock_writers()
    with mocked:
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
    _, mocked = _mock_writers()
    with mocked:
        recorder = CameraObsVideoRecorder(env, video_folder=str(tmp_path))

        for _ in range(3):
            _configure_step(env, n_envs=1)
            recorder.step(None)
            _configure_step(env, done_envs=[0], n_envs=1)
            recorder.step(None)

    for episode in range(3):
        for cam in CAMERAS:
            assert os.path.isfile(os.path.join(str(tmp_path), f"robot-cam-env0-{cam}-episode-{episode}.mp4"))


def test_partial_episode_dropped_on_close(tmp_path):
    """Frames accumulated without termination are silently discarded on close()."""
    env = _make_env()
    factory, mocked = _mock_writers()
    with mocked:
        recorder = CameraObsVideoRecorder(env, video_folder=str(tmp_path))

        _configure_step(env)
        recorder.step(None)  # accumulate frames, no termination

        recorder.close()

        assert factory.writers and all(writer.closed for writer in factory.writers)
        assert list(tmp_path.iterdir()) == []


def test_no_video_written_for_empty_episode(tmp_path):
    """An env terminating with no buffered frames writes no video; its episode index still advances."""
    env = _make_env()
    factory, mocked = _mock_writers()
    with mocked:
        recorder = CameraObsVideoRecorder(env, video_folder=str(tmp_path))

        # Terminate on the very first step — no prior frames were recorded.
        _configure_step(env, done_envs=[0])
        recorder.step(None)

        # No video for the empty episode, but the env's centralized index still advanced past it
        # (so a later episode's video number stays in lockstep with the per-episode results record).
        assert len(factory.writers) == len(CAMERAS)
        assert not list(tmp_path.glob("robot-cam-env0-*.mp4"))
        assert env.get_episode_index(0) == 1


def test_post_reset_frame_not_appended(tmp_path):
    """The obs on a terminal step (post-reset) is not buffered for the next episode."""
    env = _make_env()
    factory, mocked = _mock_writers()
    with mocked:
        recorder = CameraObsVideoRecorder(env, video_folder=str(tmp_path))

        _configure_step(env)
        recorder.step(None)  # 1 frame buffered for both envs

        _configure_step(env, done_envs=[0])
        recorder.step(None)  # env 0 terminates; post-reset frame discarded

        # env 0's writers were finalized; its post-reset frame was discarded.
        for cam in CAMERAS:
            assert (cam, 0) not in recorder._writers

        # env 1 streamed two frames (neither step was terminal for it).
        for cam in CAMERAS:
            active = recorder._writers[(cam, 1)]
            assert active.writer.frame_count == 2

        assert sum(writer.frame_count for writer in factory.writers) == 6


def test_long_episode_does_not_buffer_frames(tmp_path):
    """Frame storage stays in ffmpeg rather than growing a Python list."""
    env = _make_env()
    factory, mocked = _mock_writers()
    with mocked:
        recorder = CameraObsVideoRecorder(env, video_folder=str(tmp_path))
        for _ in range(10_000):
            _configure_step(env, n_envs=1)
            recorder.step(None)

        assert not hasattr(recorder, "buffers")
        assert len(recorder._writers) == len(CAMERAS)
        assert [writer.frame_count for writer in factory.writers] == [10_000, 10_000]

        recorder.close()
        assert list(tmp_path.iterdir()) == []


def test_frame_shape_change_discards_partial_video(tmp_path):
    env = _make_env()
    _, mocked = _mock_writers()
    with mocked:
        recorder = CameraObsVideoRecorder(env, video_folder=str(tmp_path))
        _configure_step(env, n_envs=1)
        recorder.step(None)
        env._step_return = (
            {CAMERA_OBS_GROUP_KEY: {cam: torch.zeros(1, H + 2, W, C, dtype=torch.uint8) for cam in CAMERAS}},
            None,
            torch.zeros(1, dtype=torch.bool),
            torch.zeros(1, dtype=torch.bool),
            None,
        )

        with pytest.raises(ValueError, match="changed frame shape"):
            recorder.step(None)

        recorder.close()
        assert list(tmp_path.iterdir()) == []


def test_failed_completion_validation_never_publishes(tmp_path):
    env = _make_env()
    factory = _FakeWriterFactory()
    with (
        patch(
            "isaaclab_arena.video.camera_observation_video_recorder.imageio_ffmpeg.write_frames",
            side_effect=factory,
        ),
        patch(
            "isaaclab_arena.video.camera_observation_video_recorder._validate_encoded_video",
            side_effect=RuntimeError("truncated stream"),
        ),
    ):
        recorder = CameraObsVideoRecorder(env, video_folder=str(tmp_path))
        _configure_step(env, n_envs=1)
        recorder.step(None)
        _configure_step(env, done_envs=[0], n_envs=1)
        with pytest.raises(RuntimeError, match="truncated stream"):
            recorder.step(None)

    assert not list(tmp_path.glob("*.mp4"))
    assert not list(tmp_path.glob("*.part"))


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

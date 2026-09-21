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

from isaaclab_arena.utils.env_step_timer import EnvStepTimerWrapper
from isaaclab_arena.utils.timer import Timer, get_timer_stats, reset_timer_stats
from isaaclab_arena.video.camera_observation_video_recorder import CAMERA_OBS_GROUP_KEY, CameraObsVideoRecorder
from isaaclab_arena.video.video_recording import VideoRecordingCfg, wrap_env_for_video

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

    def reset(self, *, env_ids=None, **kwargs):
        reset_env_ids = range(len(self._step_return[2])) if env_ids is None else env_ids
        for env_id in reset_env_ids:
            env_id = int(env_id)
            self._episode_counts[env_id] = self._episode_counts.get(env_id, 0) + 1
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


def _configure_step(
    env: _StubEnv,
    done_envs: list[int] | None = None,
    n_envs: int = 2,
    final_camera_values: dict[str, list[int]] | None = None,
    truncated_envs: list[int] | None = None,
):
    """Set the next step return value with given terminations."""
    terminated = torch.zeros(n_envs, dtype=torch.bool)
    for idx in done_envs or []:
        terminated[idx] = True
    truncated = torch.zeros(n_envs, dtype=torch.bool)
    for idx in truncated_envs or []:
        truncated[idx] = True
    cam_obs = {cam: torch.zeros(n_envs, H, W, C, dtype=torch.uint8) for cam in CAMERAS}
    obs = {CAMERA_OBS_GROUP_KEY: cam_obs}
    info = None
    if final_camera_values is not None:
        final_camera_obs = {
            camera: torch.stack([torch.full((H, W, C), value, dtype=torch.uint8) for value in values])
            for camera, values in final_camera_values.items()
        }
        info = {"final_obs": {CAMERA_OBS_GROUP_KEY: final_camera_obs}}
    env._step_return = (obs, None, terminated, truncated, info)


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
        self.first_pixel_values = []
        self.closed = False
        with open(filename, "wb"):
            pass

    def write_frame(self, frame):
        self.frames_written += 1
        self.first_pixel_values.append(int(frame[0, 0, 0]))

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


def test_manual_reset_discards_partial_episode_and_starts_new_video(tmp_path):
    """An explicit reset closes interrupted encoders before starting a fresh episode."""
    env = _make_env()
    with _patched_writers() as writers:
        recorder = CameraObsVideoRecorder(env, video_folder=str(tmp_path))
        _configure_step(env, n_envs=1)
        recorder.step(None)
        recorder.step(None)
        interrupted_writers = list(writers)

        recorder.reset()

        assert all(writer.closed for writer in interrupted_writers)
        assert list(tmp_path.iterdir()) == []
        recorder.step(None)
        fresh_writers = writers[len(interrupted_writers) :]
        assert len(fresh_writers) == len(CAMERAS)
        assert all(writer.frames_written == 1 for writer in fresh_writers)
        assert all(writer.filename.endswith("episode-1.mp4") for writer in fresh_writers)
        assert all(writer.frames_written == 2 for writer in interrupted_writers)
        recorder.close()


@pytest.mark.parametrize("env_ids", [[1], (1,), torch.tensor([1]), []], ids=["list", "tuple", "tensor", "empty"])
def test_subset_reset_preserves_unaffected_video_streams(tmp_path, env_ids):
    """Reset only the requested streams and forward the environment selection unchanged."""
    env = _make_env()
    with _patched_writers() as writers:
        recorder = CameraObsVideoRecorder(env, video_folder=str(tmp_path))
        _configure_step(env)
        recorder.step(None)
        original_writers = list(writers)

        with patch.object(env, "reset", wraps=env.reset) as reset:
            recorder.reset(env_ids=env_ids)
            assert reset.call_args.kwargs["env_ids"] is env_ids

        for writer in original_writers:
            reset_requested = len(env_ids) > 0 and "-env1-" in writer.filename
            assert writer.closed == reset_requested
            assert os.path.exists(writer.filename) != reset_requested
        recorder.step(None)
        for writer in original_writers:
            assert writer.frames_written == (1 if writer.closed else 2)
        fresh_writers = writers[len(original_writers) :]
        assert len(fresh_writers) == len(CAMERAS) * len(env_ids)
        assert all(writer.frames_written == 1 for writer in fresh_writers)
        recorder.close()


def test_completed_videos_survive_manual_reset_and_close(tmp_path):
    """Discard interrupted streams without removing previously completed episode files."""
    env = _make_env()
    with _patched_writers() as writers:
        recorder = CameraObsVideoRecorder(env, video_folder=str(tmp_path))
        _configure_step(env)
        recorder.step(None)
        _configure_step(env, done_envs=[0])
        recorder.step(None)
        completed_writers = [writer for writer in writers if writer.closed]
        assert len(completed_writers) == len(CAMERAS)

        recorder.reset()
        assert all(os.path.isfile(writer.filename) for writer in completed_writers)
        _configure_step(env)
        recorder.step(None)
        recorder.close()

        assert sorted(os.listdir(tmp_path)) == sorted(os.path.basename(writer.filename) for writer in completed_writers)


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


def test_frame_writing_is_timed_separately_from_finalizing(tmp_path):
    """Frame writes are timed on every recorded step; finalizing is timed only on a reset."""
    reset_timer_stats()
    env = _make_env()
    with _patched_writers():
        recorder = CameraObsVideoRecorder(env, video_folder=str(tmp_path))

        _configure_step(env)
        recorder.step(None)
        recorder.step(None)

        stats = get_timer_stats()
        assert stats["record_camera_frames"].count == 2
        assert "record_camera_finalize" not in stats

        _configure_step(env, done_envs=[0])
        recorder.step(None)

        stats = get_timer_stats()
        assert stats["record_camera_frames"].count == 3
        assert stats["record_camera_finalize"].count == 1


def test_recording_stack_reports_its_costs_under_the_enclosing_step_timer(tmp_path):
    """Stepping the recording stack from inside a step timer reproduces the rollout's breakdown."""
    reset_timer_stats()
    env = _make_env()
    _configure_step(env)

    wrapped = wrap_env_for_video(
        env,
        VideoRecordingCfg(record_camera_video=True, video_base_dir=str(tmp_path)),
        num_steps=1,
        num_episodes=None,
    )

    assert isinstance(wrapped, CameraObsVideoRecorder)
    assert isinstance(wrapped.env, EnvStepTimerWrapper)
    assert wrapped.env.env is env

    with _patched_writers():
        with Timer("env_step"):  # Stands in for the rollout's env step timer.
            wrapped.step(None)

    # The sim step and the recording cost are siblings below the env step, so subtracting them
    # from it is what isolates the recording overhead.
    stats = get_timer_stats()
    assert stats["env_step"].count == 1
    assert stats["env_step/sim_step"].count == 1
    assert stats["env_step/record_camera_frames"].count == 1
    assert stats["env_step"].total_ms >= stats["env_step/sim_step"].total_ms


def test_wrap_env_for_video_adds_no_timer_when_recording_is_disabled(tmp_path):
    """With no recorder requested the env is returned unchanged and nothing is timed."""
    reset_timer_stats()
    env = _make_env()

    wrapped = wrap_env_for_video(env, VideoRecordingCfg(video_base_dir=str(tmp_path)), 1, None)

    assert wrapped is env
    assert get_timer_stats() == {}


def test_no_timing_recorded_without_camera_observations(tmp_path):
    """A step carrying no camera observations does no recording work, so nothing is timed."""
    reset_timer_stats()
    env = _make_env()
    with _patched_writers():
        recorder = CameraObsVideoRecorder(env, video_folder=str(tmp_path))

        recorder.step(None)  # _StubEnv returns an empty obs until _configure_step is called

        assert "record_camera_frames" not in get_timer_stats()


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


def test_final_observation_records_terminal_frame_and_preserves_live_streams(tmp_path):
    """Use pre-reset images only for done environments, retaining normal images for live ones."""
    env = _make_env()
    final_camera_values = {"front": [11, 12], "wrist": [21, 22]}
    with _patched_writers() as writers:
        recorder = CameraObsVideoRecorder(env, video_folder=str(tmp_path))
        _configure_step(env)
        recorder.step(None)
        _configure_step(env, done_envs=[0], final_camera_values=final_camera_values)
        recorder.step(None)

        writer_by_filename = {os.path.basename(writer.filename): writer for writer in writers}
        for camera in CAMERAS:
            completed_writer = writer_by_filename[f"robot-cam-env0-{camera}-episode-0.mp4"]
            live_writer = writer_by_filename[f"robot-cam-env1-{camera}-episode-0.mp4"]
            assert completed_writer.first_pixel_values == [0, final_camera_values[camera][0]]
            assert completed_writer.closed
            assert live_writer.first_pixel_values == [0, 0]
            assert not live_writer.closed

        _configure_step(env)
        recorder.step(None)
        fresh_writers = [writer for writer in writers if writer.filename.endswith("episode-1.mp4")]
        assert len(fresh_writers) == len(CAMERAS)
        assert all(writer.first_pixel_values == [0] for writer in fresh_writers)
        recorder.close()


@pytest.mark.parametrize("truncated", [False, True], ids=["termination", "truncation"])
def test_first_step_final_observation_uses_completed_episode_number(tmp_path, truncated):
    """A first-step terminal image belongs to the episode that just completed."""
    env = _make_env()
    final_camera_values = {"front": [31], "wrist": [41]}
    with _patched_writers() as writers:
        recorder = CameraObsVideoRecorder(env, video_folder=str(tmp_path))
        for episode_index in range(2):
            _configure_step(
                env,
                done_envs=[] if truncated else [0],
                truncated_envs=[0] if truncated else [],
                n_envs=1,
                final_camera_values=final_camera_values,
            )
            recorder.step(None)
            for camera in CAMERAS:
                filename = os.path.join(str(tmp_path), f"robot-cam-env0-{camera}-episode-{episode_index}.mp4")
                writer = next(writer for writer in writers if writer.filename == filename)
                assert writer.first_pixel_values == final_camera_values[camera]
                assert writer.closed and os.path.isfile(filename)
        recorder.close()
        assert len(list(tmp_path.glob("*.mp4"))) == 2 * len(CAMERAS)


@pytest.mark.parametrize("final_camera_values", [{}, {"front": [17]}], ids=["empty", "missing_camera"])
def test_missing_final_camera_keeps_skipping_post_reset_frame(tmp_path, final_camera_values):
    """Missing final camera images must not be replaced by the new episode's images."""
    env = _make_env()
    with _patched_writers() as writers:
        recorder = CameraObsVideoRecorder(env, video_folder=str(tmp_path))
        _configure_step(env, n_envs=1)
        recorder.step(None)
        _configure_step(env, done_envs=[0], n_envs=1, final_camera_values=final_camera_values)
        recorder.step(None)
        for camera in CAMERAS:
            writer = next(writer for writer in writers if f"-{camera}-" in writer.filename)
            assert writer.first_pixel_values == [0] + final_camera_values.get(camera, [])
            assert writer.closed
        recorder.close()


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

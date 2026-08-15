# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Capture short camera and viewport clips for every exact-pose RoboLab task."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

DEFAULT_TASKS_DIR = Path(__file__).parents[1] / "tasks"
DEFAULT_OUTPUT_DIR = Path("output/robolab_exact_capture")


def sanitize_video_name(name: str) -> str:
    """Return a filename-safe camera or task name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def camera_video_filename(env_index: int, camera_name: str) -> str:
    """Return the capture filename for one environment camera."""
    return f"env{env_index}_{sanitize_video_name(camera_name)}.mp4"


def _create_argument_parser() -> argparse.ArgumentParser:
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser

    parser = get_isaaclab_arena_cli_parser()
    parser.description = __doc__
    parser.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-envs", type=int, default=9)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Task stem to capture; repeat to select multiple tasks. The default captures every task.",
    )
    return parser


def _discover_task_paths(tasks_dir: Path, task_names: list[str]) -> list[Path]:
    task_paths = sorted(tasks_dir.glob("*.yaml"))
    if task_names:
        selected = set(task_names)
        task_paths = [path for path in task_paths if path.stem in selected]
        missing = selected - {path.stem for path in task_paths}
        assert not missing, f"Unknown task names: {sorted(missing)}"
    assert task_paths, f"No task YAMLs found in '{tasks_dir}'"
    return task_paths


class _CaptureWriters:
    """Incremental ffmpeg writers for one task's fixed-length clips."""

    def __init__(self, output_dir: Path, fps: int) -> None:
        self.output_dir = output_dir
        self.fps = fps
        self._writers = {}
        self._frame_counts: dict[Path, int] = {}
        self._closed = False

    def _write(self, path: Path, frame) -> None:
        import numpy as np
        import torch

        from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        if frame.dtype != np.uint8:
            scale = 255.0 if frame.dtype.kind == "f" and float(frame.max()) <= 1.0 else 1.0
            frame = np.clip(frame * scale, 0, 255).astype(np.uint8)
        assert frame.ndim == 3 and frame.shape[2] == 3, f"Expected an RGB frame, got shape {frame.shape}"
        writer = self._writers.get(path)
        if writer is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            height, width, _ = frame.shape
            writer = FFMPEG_VideoWriter(str(path), size=(width, height), fps=self.fps, threads=1)
            self._writers[path] = writer
            self._frame_counts[path] = 0
        writer.write_frame(frame)
        self._frame_counts[path] += 1

    def write_camera_observations(self, camera_observations: dict[str, object]) -> None:
        """Append one frame for every environment in every robot camera."""
        for camera_name, frames in camera_observations.items():
            for env_index in range(frames.shape[0]):
                self._write(self.output_dir / camera_video_filename(env_index, camera_name), frames[env_index])

    def write_viewport(self, frame) -> None:
        """Append one viewport frame."""
        self._write(self.output_dir / "viewport.mp4", frame)

    def close(self, expected_frames: int | None = None) -> None:
        """Finalize every clip and optionally assert that each has the requested length."""
        if self._closed:
            return
        for writer in self._writers.values():
            writer.close()
        self._closed = True
        if expected_frames is not None:
            bad_counts = {str(path): count for path, count in self._frame_counts.items() if count != expected_frames}
            assert not bad_counts, f"Capture videos have unexpected frame counts: {bad_counts}"
            assert self._writers, f"No video frames were captured in '{self.output_dir}'"


def _build_environment(task_path: Path, num_envs: int, device: str):
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import reapply_viewer_cfg

    arena_environment = ArenaEnvGraphSpec.from_yaml(task_path).to_arena_env(enable_cameras=True)
    builder = ArenaEnvBuilder(
        arena_environment,
        ArenaEnvBuilderCfg(num_envs=num_envs, device=device, solve_relations=False),
    )
    _, env_cfg, env_kwargs = builder.build_registered()
    env_cfg.recorders = {}
    env_cfg.episode_recorders = {}
    env = builder.make_registered(env_cfg, env_kwargs, render_mode="rgb_array")
    reapply_viewer_cfg(env)
    return env


def capture_task(task_path: Path, output_dir: Path, num_envs: int, num_steps: int, device: str) -> None:
    """Build one task and capture fixed-length zero-action clips."""
    import torch

    env = _build_environment(task_path, num_envs, device)
    writers = _CaptureWriters(output_dir / task_path.stem, int(env.metadata.get("render_fps", 30)))
    capture_completed = False
    try:
        env.reset()
        zero_actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        with torch.inference_mode():
            for _ in range(num_steps):
                observations, _, _, _, _ = env.step(zero_actions)
                camera_observations = observations.get("camera_obs", {})
                assert camera_observations, f"Task '{task_path.stem}' produced no camera observations"
                writers.write_camera_observations(camera_observations)
                writers.write_viewport(env.render())
        writers.close(expected_frames=num_steps)
        capture_completed = True
    finally:
        from isaaclab_arena.evaluation.resource_cleanup import close_environment

        if not capture_completed:
            writers.close()
        close_environment(env)


def main() -> None:
    """Launch one SimulationApp and capture all selected tasks sequentially."""
    args = _create_argument_parser().parse_args()
    assert args.num_envs > 0, "--num-envs must be positive"
    assert args.num_steps > 0, "--num-steps must be positive"
    args.enable_cameras = True
    task_paths = _discover_task_paths(args.tasks_dir, args.task)

    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    with SimulationAppContext(args):
        for index, task_path in enumerate(task_paths, start=1):
            print(f"[{index}/{len(task_paths)}] Capturing {task_path.stem}", flush=True)
            capture_task(task_path, args.output_dir, args.num_envs, args.num_steps, args.device)


if __name__ == "__main__":
    main()

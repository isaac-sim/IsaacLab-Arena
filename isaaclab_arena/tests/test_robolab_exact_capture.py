# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the exact-pose RoboLab specs and capture post-processing."""

from __future__ import annotations

import numpy as np
import yaml
from pathlib import Path

from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from PIL import Image

from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena_environments.robolab_exact.scripts.capture_task_videos import camera_video_filename
from isaaclab_arena_environments.robolab_exact.scripts.extract_middle_frames import (
    extract_middle_frame,
    output_filename_for_video,
)

_ROBOLAB_EXACT_DIR = Path(__file__).parents[2] / "isaaclab_arena_environments" / "robolab_exact"


def test_robolab_exact_specs_are_complete_and_parseable():
    scene_paths = sorted((_ROBOLAB_EXACT_DIR / "scenes").glob("*.yaml"))
    task_paths = sorted((_ROBOLAB_EXACT_DIR / "tasks").glob("*.yaml"))

    assert len(scene_paths) == 17
    assert len(task_paths) == 38
    for scene_path in scene_paths:
        scene = yaml.safe_load(scene_path.read_text(encoding="utf-8"))
        assert all(obj.get("initial_pose") is not None for obj in scene["objects"])
        assert "relations" not in scene
    for task_path in task_paths:
        ArenaEnvGraphSpec.from_yaml(task_path)


def test_capture_video_filenames_preserve_environment_and_camera():
    assert camera_video_filename(7, "wrist/camera_rgb") == "env7_wrist_camera_rgb.mp4"
    video_path = Path("/capture/banana_in_bowl/env7_wrist_camera_rgb.mp4")
    assert output_filename_for_video(video_path) == "banana_in_bowl_env7_wrist_camera_rgb.png"
    assert output_filename_for_video(video_path.parent / "viewport.mp4") == "banana_in_bowl_viewport.png"


def test_extract_middle_frame_uses_true_frame_count(tmp_path):
    video_path = tmp_path / "env0_camera.mp4"
    writer = FFMPEG_VideoWriter(str(video_path), size=(8, 8), fps=10, threads=1)
    for frame_index in range(10):
        value = 0 if frame_index < 5 else 255
        writer.write_frame(np.full((8, 8, 3), value, dtype=np.uint8))
    writer.close()

    output_path = tmp_path / "middle.png"
    assert extract_middle_frame(video_path, output_path) == 5
    assert np.asarray(Image.open(output_path)).mean() > 240

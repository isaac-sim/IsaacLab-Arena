# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.video.camera_observation_video_recorder import (
    format_episode_video_filename,
    parse_episode_video_filename,
)
from isaaclab_arena.visualization.report import build_report


def test_episode_video_filename_roundtrip_no_rebuild():
    name = format_episode_video_filename("robot-cam", 2, "wrist_cam", 5)
    parsed = parse_episode_video_filename(name)
    assert parsed is not None
    assert (parsed.prefix, parsed.env_index, parsed.camera_name, parsed.episode_index) == (
        "robot-cam",
        2,
        "wrist_cam",
        5,
    )
    assert parsed.rebuild_index is None
    # Re-formatting the recovered fields reproduces the original filename.
    assert (
        format_episode_video_filename(parsed.prefix, parsed.env_index, parsed.camera_name, parsed.episode_index) == name
    )


def test_build_report_smoke(tmp_path):
    # One job sub-directory (experiment_runner layout) with two cameras, two envs, two rebuilds, plus a
    # flat file (policy_runner layout) and stray files the scanner must ignore.
    job_dir = tmp_path / "pick_and_place"
    job_dir.mkdir()
    video_names = []
    for rebuild in (0, 1):
        for env in (0, 1):
            for camera in ("wrist_cam", "front_cam"):
                name = format_episode_video_filename(f"robot-cam-rebuild{rebuild}", env, camera, 0)
                (job_dir / name).write_bytes(b"")
                video_names.append(name)

    flat_name = format_episode_video_filename("robot-cam", 0, "wrist_cam", 0)
    (tmp_path / flat_name).write_bytes(b"")
    video_names.append(flat_name)
    (tmp_path / "notes.txt").write_text("ignore me")
    (tmp_path / "not-a-recorder-file.mp4").write_bytes(b"")

    report_path = build_report(tmp_path)

    assert report_path == tmp_path / "index.html"
    assert report_path.exists()
    html = report_path.read_text(encoding="utf-8")

    assert "Evaluation Report" in html
    assert "pick_and_place" in html
    assert "wrist_cam" in html and "front_cam" in html

    # Every matched video is referenced; the non-matching mp4 is not.
    for name in video_names:
        assert name in html
    assert "not-a-recorder-file.mp4" not in html

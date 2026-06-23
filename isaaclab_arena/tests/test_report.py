# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit + smoke tests for the evaluation report (pure Python, no simulation).

Covers the recorder<->report filename contract (``format_episode_video_filename`` /
``parse_episode_video_filename`` round-trip, including the ``-rebuild<N>`` prefix), the
rebuild-disambiguating contiguous renumbering in ``_scan_jobs``, and an end-to-end
``build_report`` smoke test that scans recorder-named mp4s into a self-contained ``index.html``.
"""

from isaaclab_arena.video.camera_observation_video_recorder import (
    format_episode_video_filename,
    parse_episode_video_filename,
)
from isaaclab_arena.visualization.report import _scan_jobs, build_report


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


def test_episode_video_filename_roundtrip_with_rebuild():
    # The eval runner bakes the rebuild into the name_prefix; parsing recovers it separately.
    name = format_episode_video_filename("robot-cam-rebuild3", 1, "front_cam", 0)
    parsed = parse_episode_video_filename(name)
    assert parsed is not None
    assert parsed.rebuild_index == 3
    assert (parsed.prefix, parsed.env_index, parsed.camera_name, parsed.episode_index) == (
        "robot-cam",
        1,
        "front_cam",
        0,
    )
    reformatted = format_episode_video_filename(
        f"{parsed.prefix}-rebuild{parsed.rebuild_index}", parsed.env_index, parsed.camera_name, parsed.episode_index
    )
    assert reformatted == name


def test_parse_episode_video_filename_rejects_non_recorder_names():
    assert parse_episode_video_filename("not-a-recorder-file.mp4") is None
    assert parse_episode_video_filename("index.html") is None


def test_scan_jobs_renumbers_rebuilds_contiguously(tmp_path):
    # Same (job, env) recorded across two rebuilds, each with episodes 0 and 1. Without rebuild
    # disambiguation the two episode-0 files would collide; the scan must renumber them into a
    # contiguous, collision-free range.
    job_dir = tmp_path / "jobA"
    job_dir.mkdir()
    for rebuild in (0, 1):
        for episode in (0, 1):
            name = format_episode_video_filename(f"robot-cam-rebuild{rebuild}", 0, "wrist_cam", episode)
            (job_dir / name).write_bytes(b"")

    jobs = _scan_jobs(tmp_path)

    assert len(jobs) == 1
    episodes = jobs[0].episodes
    assert all(ep.env_index == 0 for ep in episodes)
    indices = sorted(ep.episode_index for ep in episodes)
    assert indices == [0, 1, 2, 3], "episode indices must be contiguous and collision-free across rebuilds"


def test_build_report_smoke(tmp_path):
    # One job sub-directory (eval_runner layout) with two cameras, two envs, two rebuilds, plus a
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

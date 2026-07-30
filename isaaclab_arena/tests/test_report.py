# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json

from isaaclab_arena.video.camera_observation_video_recorder import (
    format_episode_video_filename,
    parse_episode_video_filename,
)
from isaaclab_arena.visualization.report import _split_job, build_report


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


def test_split_job():
    # Policy is the text after the final underscore; tasks may contain underscores themselves.
    assert _split_job("banana_in_bowl_pi0") == ("banana_in_bowl", "pi0")
    assert _split_job("food_packing_1_cans_cosmos") == ("food_packing_1_cans", "cosmos")
    # Nested job paths split on the leaf segment only.
    assert _split_job("sweep/banana_in_bowl_gr00t") == ("sweep/banana_in_bowl", "gr00t")
    # No underscore (e.g. the policy runner's single unnamed job) -> empty policy.
    assert _split_job("") == ("", "")
    assert _split_job("standalone") == ("standalone", "")


def _write_job(job_dir, success):
    """Create a one-episode job folder with a matching results record and video."""
    job_dir.mkdir()
    name = format_episode_video_filename("robot-cam-rebuild0", 0, "wrist_cam", 0)
    (job_dir / name).write_bytes(b"")
    record = {"env_id": 0, "episode_in_env": 0, "success": success, "job_name": job_dir.name}
    (job_dir / "episode_results_rebuild0.jsonl").write_text(json.dumps(record) + "\n")


def test_report_groups_by_task_and_policy_with_lazy_videos(tmp_path):
    # Two policies of the same task should collapse into one task section with two policy sub-sections.
    _write_job(tmp_path / "banana_in_bowl_pi0", success=True)
    _write_job(tmp_path / "banana_in_bowl_cosmos", success=False)

    html = build_report(tmp_path).read_text(encoding="utf-8")

    # One task grouping (banana_in_bowl), two policy sub-sections (pi0, cosmos).
    assert html.count('<details class="task"') == 1
    assert html.count('<details class="policy"') == 2
    assert ">banana_in_bowl " in html

    # Videos are lazy: sources carry data-src (no eager src=), and the template ships the loader.
    assert 'data-src="' in html
    assert "<source src=" not in html
    assert 'addEventListener("toggle"' in html

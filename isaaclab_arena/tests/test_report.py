# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test the hierarchical evaluation report written by ``build_report``."""

import json
import re

from isaaclab_arena.evaluation.arena_run import RunStatus
from isaaclab_arena.video.camera_observation_video_recorder import (
    format_episode_video_filename,
    parse_episode_video_filename,
)
from isaaclab_arena.visualization.report import RunExecutionReport, build_report

# Matches a real video element, as opposed to the word "video" in the page's script or prose.
_VIDEO_ELEMENT_PATTERN = re.compile(r"<video[\s>]")


def _write_run(experiment_dir, run_name: str, num_episodes: int = 2, cameras=("wrist_cam", "front_cam")):
    """Write one Run sub-directory with per-episode results and a video per (episode, camera)."""
    run_dir = experiment_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    records = [
        {"env_id": 0, "episode_in_env": index, "success": index == 0, "seed": 42} for index in range(num_episodes)
    ]
    (run_dir / "episode_results_rebuild0.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    video_names = []
    for record in records:
        for camera in cameras:
            name = format_episode_video_filename("robot-cam-rebuild0", 0, camera, record["episode_in_env"])
            (run_dir / name).write_bytes(b"")
            video_names.append(name)
    return video_names


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


def test_build_report_writes_a_page_per_task_and_run(tmp_path):
    for task in ("banana_in_bowl", "bowl_in_bin"):
        for policy in ("pi0", "cosmos"):
            _write_run(tmp_path, f"{task}_{policy}")
    (tmp_path / "notes.txt").write_text("ignore me")
    (tmp_path / "not-a-recorder-file.mp4").write_bytes(b"")

    report_path = build_report(tmp_path)

    assert report_path == tmp_path / "index.html"
    index = report_path.read_text(encoding="utf-8")
    assert "Evaluation Report" in index
    assert "Success rate by task and policy" in index
    assert "banana_in_bowl" in index and "bowl_in_bin" in index

    pages = tmp_path / "report"
    assert sorted(page.name for page in pages.glob("task_*.html")) == [
        "task_banana_in_bowl.html",
        "task_bowl_in_bin.html",
    ]
    assert len(list(pages.glob("job_*.html"))) == 4


def test_overview_and_task_pages_reference_no_video(tmp_path):
    # The overview is what a large experiment opens first, so it must stay free of video entirely:
    # emitting one element per recording is what made the previous single-page report unusable.
    for policy in ("pi0", "cosmos"):
        _write_run(tmp_path, f"banana_in_bowl_{policy}")
        _write_run(tmp_path, f"bowl_in_bin_{policy}")

    build_report(tmp_path)

    index = (tmp_path / "index.html").read_text(encoding="utf-8")
    task_page = (tmp_path / "report" / "task_banana_in_bowl.html").read_text(encoding="utf-8")
    assert not _VIDEO_ELEMENT_PATTERN.search(index)
    assert ".mp4" not in index
    assert not _VIDEO_ELEMENT_PATTERN.search(task_page)
    assert ".mp4" not in task_page


def test_run_page_defers_every_video_until_it_scrolls_into_view(tmp_path):
    video_names = _write_run(tmp_path, "banana_in_bowl_pi0", num_episodes=3)
    _write_run(tmp_path, "banana_in_bowl_cosmos")

    build_report(tmp_path)

    run_page = (tmp_path / "report" / "job_banana_in_bowl_pi0.html").read_text(encoding="utf-8")
    # Every recording is referenced, but as a slot the script mounts later, never as a <video> tag.
    assert not _VIDEO_ELEMENT_PATTERN.search(run_page)
    for name in video_names:
        assert f'data-video-src="../banana_in_bowl_pi0/{name}"' in run_page
    assert run_page.count("data-video-src=") == len(video_names)
    assert "not-a-recorder-file.mp4" not in run_page


def test_report_pages_link_down_and_back_up(tmp_path):
    for policy in ("pi0", "cosmos"):
        _write_run(tmp_path, f"banana_in_bowl_{policy}")
        _write_run(tmp_path, f"bowl_in_bin_{policy}")

    build_report(tmp_path)

    index = (tmp_path / "index.html").read_text(encoding="utf-8")
    task_page = (tmp_path / "report" / "task_banana_in_bowl.html").read_text(encoding="utf-8")
    run_page = (tmp_path / "report" / "job_banana_in_bowl_pi0.html").read_text(encoding="utf-8")

    assert 'href="report/task_banana_in_bowl.html"' in index
    assert 'href="job_banana_in_bowl_pi0.html"' in task_page
    assert 'href="../index.html"' in task_page
    assert 'href="../index.html"' in run_page
    assert 'href="task_banana_in_bowl.html"' in run_page
    # Episode chips deep-link to the matching card on the run page.
    assert 'href="job_banana_in_bowl_pi0.html#ep-0-0"' in task_page
    assert 'id="ep-0-0"' in run_page


def test_run_page_names_its_policy_throughout(tmp_path):
    # A run page is scrolled through hundreds of episodes, so the policy has to stay answerable
    # after the heading has scrolled away.
    _write_run(tmp_path, "banana_in_bowl_pi0", num_episodes=3)
    _write_run(tmp_path, "banana_in_bowl_cosmos")

    build_report(tmp_path)
    run_page = (tmp_path / "report" / "job_banana_in_bowl_pi0.html").read_text(encoding="utf-8")

    assert "<h1>pi0</h1>" in run_page
    assert '<span class="key">policy</span><span class="value">pi0</span>' in run_page
    assert '<span class="key">task</span><span class="value">banana_in_bowl</span>' in run_page
    # Repeated on every episode card, and marked as the current item in the sticky bar.
    assert run_page.count("policy <strong>pi0</strong>") == 3
    assert '<span class="current">pi0</span>' in run_page


def test_run_page_shows_which_success_signals_fired_and_which_did_not(tmp_path):
    def record(index: int, reached: int) -> dict:
        """One episode that fired the first ``reached`` predicates of a three-predicate sequence."""
        names = ["objects_settled", "object_is_above_height(object_name='banana')", "object_on_destination()"]
        objective = {"score": reached / 3, "is_complete": reached == 3, "total_groups": 1}
        if reached < 3:
            objective["active_predicates"] = {"default_group": names[reached]}
        return {
            "env_id": 0,
            "episode_in_env": index,
            "success": reached == 3,
            "progress": {
                "overall_score": reached / 3,
                "objectives": {"pick_and_place": objective},
                "events": [
                    {"objective": "pick_and_place", "predicate_index": i, "predicate_name": names[i], "step": 10 * i}
                    for i in range(reached)
                ],
            },
        }

    run_dir = tmp_path / "banana_in_bowl_pi0"
    run_dir.mkdir()
    (run_dir / "episode_results_rebuild0.jsonl").write_text(
        "\n".join(json.dumps(record(index, reached)) for index, reached in enumerate((3, 1))) + "\n",
        encoding="utf-8",
    )
    _write_run(tmp_path, "banana_in_bowl_cosmos")

    build_report(tmp_path)
    run_page = (tmp_path / "report" / "job_banana_in_bowl_pi0.html").read_text(encoding="utf-8")

    # The stalled episode still lists all three predicates: one fired, one waiting, one never reached.
    assert 'class="signal on"' in run_page and 'class="signal blocked"' in run_page
    assert 'class="signal off"' in run_page
    assert "step 10" in run_page and "waiting" in run_page
    # The predicate the run never reached is named even though no episode fired it in that episode.
    assert run_page.count("object_on_destination") >= 2


def test_every_page_below_the_overview_can_climb_back_up(tmp_path):
    _write_run(tmp_path, "banana_in_bowl_pi0")
    _write_run(tmp_path, "banana_in_bowl_cosmos")

    build_report(tmp_path)
    task_page = (tmp_path / "report" / "task_banana_in_bowl.html").read_text(encoding="utf-8")
    run_page = (tmp_path / "report" / "job_banana_in_bowl_pi0.html").read_text(encoding="utf-8")

    # A run page is long, so it carries the button both in the sticky bar and at the very end.
    assert run_page.count('<a class="upbutton" href="task_banana_in_bowl.html"') == 2
    assert 'class="upbutton" href="../index.html"' in run_page
    assert "Back to banana_in_bowl" in run_page
    assert task_page.count('<a class="upbutton" href="../index.html"') == 2
    assert "Back to the overview" in task_page


def test_task_page_names_the_policies_it_compares(tmp_path):
    _write_run(tmp_path, "banana_in_bowl_pi0")
    _write_run(tmp_path, "banana_in_bowl_cosmos")

    build_report(tmp_path)
    task_page = (tmp_path / "report" / "task_banana_in_bowl.html").read_text(encoding="utf-8")

    assert '<span class="key">comparing</span><span class="value">cosmos, pi0</span>' in task_page
    assert "cosmos episodes" in task_page and "pi0 episodes" in task_page


def test_rebuilding_removes_pages_for_runs_that_no_longer_exist(tmp_path):
    for policy in ("pi0", "cosmos"):
        _write_run(tmp_path, f"banana_in_bowl_{policy}")
        _write_run(tmp_path, f"bowl_in_bin_{policy}")
    build_report(tmp_path)
    assert (tmp_path / "report" / "task_bowl_in_bin.html").exists()

    for policy in ("pi0", "cosmos"):
        for path in (tmp_path / f"bowl_in_bin_{policy}").iterdir():
            path.unlink()
        (tmp_path / f"bowl_in_bin_{policy}").rmdir()
    build_report(tmp_path)

    assert (tmp_path / "report" / "task_banana_in_bowl.html").exists()
    assert not (tmp_path / "report" / "task_bowl_in_bin.html").exists()
    assert not (tmp_path / "report" / "job_bowl_in_bin_pi0.html").exists()


def test_build_report_on_an_empty_directory_writes_an_empty_report(tmp_path):
    report_path = build_report(tmp_path)

    assert report_path.exists()
    assert "No results recorded yet." in report_path.read_text(encoding="utf-8")


def test_build_report_with_supplied_run_execution_results(tmp_path):
    report_path = build_report(
        tmp_path,
        run_executions=[
            RunExecutionReport(run_name="completed-run", status=RunStatus.COMPLETED, process_exit_code=0),
            RunExecutionReport(run_name="failed-run", status=RunStatus.FAILED, process_exit_code=17),
        ],
    )

    report_contents = report_path.read_text(encoding="utf-8")
    assert "2 run(s) &middot; 1 completed &middot; 1 failed &middot; 0 episode(s)" in report_contents
    assert "Failed runs (1)" in report_contents
    assert "failed-run" in report_contents
    assert "<code>17</code>" in report_contents


def test_failed_runs_are_listed_but_excluded_from_the_results(tmp_path):
    _write_run(tmp_path, "banana_in_bowl_pi0")
    _write_run(tmp_path, "banana_in_bowl_cosmos")

    build_report(
        tmp_path,
        run_executions=[
            RunExecutionReport(run_name="banana_in_bowl_cosmos", status=RunStatus.FAILED, process_exit_code=9)
        ],
    )

    index = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "Failed runs (1)" in index
    assert not (tmp_path / "report" / "job_banana_in_bowl_cosmos.html").exists()
    assert (tmp_path / "report" / "job_banana_in_bowl_pi0.html").exists()


def test_ungrouped_results_still_produce_a_report(tmp_path):
    # The policy runner writes results directly into the output directory, with no per-run folder.
    (tmp_path / "episode_results_rank0.jsonl").write_text(
        json.dumps({"env_id": 0, "episode_in_env": 0, "success": True}) + "\n", encoding="utf-8"
    )
    name = format_episode_video_filename("robot-cam", 0, "wrist_cam", 0)
    (tmp_path / name).write_bytes(b"")

    report_path = build_report(tmp_path)

    index = report_path.read_text(encoding="utf-8")
    assert "Runs" in index
    assert "could not be grouped" in index
    assert not _VIDEO_ELEMENT_PATTERN.search(index)

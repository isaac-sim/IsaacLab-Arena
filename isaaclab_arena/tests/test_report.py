# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test the hierarchical evaluation report written by ``build_report``."""

import json
import re
import subprocess
import sys

from isaaclab_arena.visualization.episode_results_files import (
    format_episode_video_filename,
    parse_episode_video_filename,
)
from isaaclab_arena.visualization.report import RunExecutionReport, _resolve_results_dir, build_report
from isaaclab_arena.visualization.report_render import unique_slugs

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
    assert parsed.rebuild_index == 0
    assert (
        format_episode_video_filename(parsed.prefix, parsed.env_index, parsed.camera_name, parsed.episode_index) == name
    )


def test_unique_slugs_deduplicates_names_with_the_same_safe_form():
    slugs = unique_slugs(["my run", "my+run"])

    assert slugs == {"my run": "my_run", "my+run": "my_run_2"}


def test_resolve_results_dir_uses_the_latest_timestamped_child(tmp_path):
    old = tmp_path / "2026-01-01_00-00-00"
    latest = tmp_path / "2026-01-02_00-00-00"
    old.mkdir()
    latest.mkdir()
    (tmp_path / "not-a-run").mkdir()

    assert _resolve_results_dir(tmp_path) == latest


def test_resolve_results_dir_returns_input_when_there_are_no_timestamped_children(tmp_path):
    (tmp_path / "results").mkdir()
    missing = tmp_path / "missing"

    assert _resolve_results_dir(tmp_path) == tmp_path
    assert _resolve_results_dir(missing) == missing


def test_report_import_is_leaf_only():
    code = """
import sys
import isaaclab_arena.visualization.report
blocked = [
    name for name in sys.modules
    if name.startswith('isaaclab_arena.evaluation')
    or name.startswith('isaaclab_arena.video')
    or name.split('.')[0] in {'torch', 'gymnasium', 'moviepy'}
]
print('\\n'.join(sorted(blocked)))
raise SystemExit(1 if blocked else 0)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr


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


def test_build_report_writes_distinct_pages_for_slug_collisions(tmp_path):
    _write_run(tmp_path, "my run")
    _write_run(tmp_path, "my+run")

    build_report(tmp_path)

    pages = tmp_path / "report"
    assert (pages / "task_my_run.html").exists()
    assert (pages / "task_my_run_2.html").exists()
    assert (pages / "job_my_run.html").exists()
    assert (pages / "job_my_run_2.html").exists()
    assert "my run" in (pages / "job_my_run.html").read_text(encoding="utf-8")
    assert "my+run" in (pages / "job_my_run_2.html").read_text(encoding="utf-8")


def test_sparse_task_policy_matrix_stays_grouped(tmp_path):
    _write_run(tmp_path, "banana_in_bowl_pi0")
    _write_run(tmp_path, "banana_in_bowl_cosmos")
    _write_run(tmp_path, "bowl_in_bin_cosmos")

    build_report(tmp_path)

    index = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "Success rate by task and policy" in index
    assert "bowl_in_bin" in index
    assert '<td class="cell missing"' in index


def test_overview_and_task_pages_reference_no_video(tmp_path):
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
    (tmp_path / "banana_in_bowl_pi0" / "not-a-recorder-file.mp4").write_bytes(b"")

    build_report(tmp_path)

    run_page = (tmp_path / "report" / "job_banana_in_bowl_pi0.html").read_text(encoding="utf-8")
    assert not _VIDEO_ELEMENT_PATTERN.search(run_page)
    for name in video_names:
        assert f'data-video-src="../banana_in_bowl_pi0/{name}"' in run_page
    assert run_page.count("data-video-src=") == len(video_names)
    assert "not-a-recorder-file.mp4" not in run_page


def test_run_pages_are_paginated(tmp_path):
    _write_run(tmp_path, "banana_in_bowl_pi0", num_episodes=5, cameras=("wrist_cam",))
    _write_run(tmp_path, "banana_in_bowl_cosmos", num_episodes=1, cameras=("wrist_cam",))

    build_report(tmp_path, episodes_per_page=2)

    pages = tmp_path / "report"
    assert (pages / "job_banana_in_bowl_pi0_1.html").exists()
    assert (pages / "job_banana_in_bowl_pi0_2.html").exists()
    assert (pages / "job_banana_in_bowl_pi0_3.html").exists()
    first_page = (pages / "job_banana_in_bowl_pi0_1.html").read_text(encoding="utf-8")
    assert "Page 1 of 3" in first_page
    assert first_page.count("data-video-src=") == 2


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
    assert 'href="job_banana_in_bowl_pi0.html#ep-0-0"' in task_page
    assert 'id="ep-0-0"' in run_page


def test_run_page_names_its_policy_throughout(tmp_path):
    _write_run(tmp_path, "banana_in_bowl_pi0", num_episodes=3)
    _write_run(tmp_path, "banana_in_bowl_cosmos")

    build_report(tmp_path)
    run_page = (tmp_path / "report" / "job_banana_in_bowl_pi0.html").read_text(encoding="utf-8")

    assert "<h1>pi0</h1>" in run_page
    assert '<span class="key">policy</span><span class="value">pi0</span>' in run_page
    assert '<span class="key">task</span><span class="value">banana_in_bowl</span>' in run_page
    assert run_page.count("policy <strong>pi0</strong>") == 3
    assert '<span class="current">pi0</span>' in run_page


def test_run_page_shows_which_success_signals_fired_and_which_did_not(tmp_path):
    def record(index: int, reached: int) -> dict:
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

    assert 'class="signal on"' in run_page and 'class="signal blocked"' in run_page
    assert 'class="signal off"' in run_page
    assert "step 10" in run_page and "waiting" in run_page
    assert run_page.count("object_on_destination") >= 2


def test_unknown_blocked_predicate_is_shown_without_known_sequence(tmp_path):
    run_dir = tmp_path / "banana_in_bowl_pi0"
    run_dir.mkdir()
    (run_dir / "episode_results_rebuild0.jsonl").write_text(
        json.dumps({
            "env_id": 0,
            "episode_in_env": 0,
            "success": False,
            "progress": {
                "objectives": {
                    "pick": {
                        "score": 0.0,
                        "is_complete": False,
                        "total_groups": 1,
                        "active_predicates": {"default": "never_seen(arg=1)"},
                    }
                },
                "events": [],
            },
        })
        + "\n",
        encoding="utf-8",
    )
    _write_run(tmp_path, "banana_in_bowl_cosmos")

    build_report(tmp_path)

    run_page = (tmp_path / "report" / "job_banana_in_bowl_pi0.html").read_text(encoding="utf-8")
    assert "never_seen" in run_page


def test_run_page_reports_conflicting_objective_family_sequences(tmp_path):
    run_dir = tmp_path / "banana_in_bowl_pi0"
    run_dir.mkdir()
    (run_dir / "episode_results_rebuild0.jsonl").write_text(
        json.dumps({
            "env_id": 0,
            "episode_in_env": 0,
            "success": False,
            "progress": {
                "objectives": {
                    "subtask_0/pick": {"score": 0.0, "is_complete": False, "total_groups": 1},
                    "subtask_1/pick": {"score": 0.0, "is_complete": False, "total_groups": 1},
                },
                "events": [
                    {"objective": "subtask_0/pick", "predicate_index": 0, "predicate_name": "first_predicate"},
                    {"objective": "subtask_1/pick", "predicate_index": 0, "predicate_name": "other_predicate"},
                ],
            },
        })
        + "\n",
        encoding="utf-8",
    )
    _write_run(tmp_path, "banana_in_bowl_cosmos")

    build_report(tmp_path)

    run_page = (tmp_path / "report" / "job_banana_in_bowl_pi0.html").read_text(encoding="utf-8")
    assert "Data issues" in run_page
    assert "conflicting predicate sequences" in run_page


def test_every_page_below_the_overview_can_climb_back_up(tmp_path):
    _write_run(tmp_path, "banana_in_bowl_pi0")
    _write_run(tmp_path, "banana_in_bowl_cosmos")

    build_report(tmp_path)
    task_page = (tmp_path / "report" / "task_banana_in_bowl.html").read_text(encoding="utf-8")
    run_page = (tmp_path / "report" / "job_banana_in_bowl_pi0.html").read_text(encoding="utf-8")

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
            RunExecutionReport(run_name="completed-run", status="completed", process_exit_code=0),
            RunExecutionReport(run_name="failed-run", status="failed", process_exit_code=17),
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
        run_executions=[RunExecutionReport(run_name="banana_in_bowl_cosmos", status="failed", process_exit_code=9)],
    )

    index = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "Failed runs (1)" in index
    assert not (tmp_path / "report" / "job_banana_in_bowl_cosmos.html").exists()
    assert (tmp_path / "report" / "job_banana_in_bowl_pi0.html").exists()


def test_ungrouped_results_still_produce_a_report(tmp_path):
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


def test_overview_descends_to_a_task_page_when_runs_cannot_be_grouped(tmp_path):
    # A single-policy Experiment has no policy axis to factorize on, so its Runs stay ungrouped.
    # The overview must still descend to the task page: that is the level showing where episodes
    # got to, and skipping it strands the reader in the videos with no funnel.
    _write_run(tmp_path, "banana_in_bowl_pi0")
    _write_run(tmp_path, "banana_on_plate_pi0")

    build_report(tmp_path)
    index = (tmp_path / "index.html").read_text(encoding="utf-8")

    assert 'href="report/task_banana_in_bowl_pi0.html"' in index
    assert 'href="report/task_banana_on_plate_pi0.html"' in index
    # Never straight to the episodes.
    assert "report/job_" not in index
    # And the task page it lands on descends the rest of the way.
    task_page = (tmp_path / "report" / "task_banana_in_bowl_pi0.html").read_text(encoding="utf-8")
    assert 'href="job_banana_in_bowl_pi0.html"' in task_page


def test_every_written_page_is_reachable_from_the_overview(tmp_path):
    for run in ("banana_in_bowl_pi0", "banana_in_bowl_cosmos", "solo_run_pi0"):
        _write_run(tmp_path, run)

    build_report(tmp_path)
    pages = tmp_path / "report"
    linked = (tmp_path / "index.html").read_text(encoding="utf-8") + "".join(
        page.read_text(encoding="utf-8") for page in pages.glob("*.html")
    )

    orphans = [page.name for page in pages.glob("*.html") if page.name not in linked]
    assert not orphans, f"pages written but unreachable: {orphans}"


def test_malformed_jsonl_is_reported_without_crashing(tmp_path):
    run_dir = tmp_path / "banana_in_bowl_pi0"
    run_dir.mkdir()
    (run_dir / "episode_results_rebuild0.jsonl").write_text("{bad json\n", encoding="utf-8")
    _write_run(tmp_path, "banana_in_bowl_cosmos")

    report_path = build_report(tmp_path)

    index = report_path.read_text(encoding="utf-8")
    assert "Data issues" in index
    assert "invalid JSON" in index


def test_media_paths_are_url_quoted_and_text_is_escaped(tmp_path):
    run_dir = tmp_path / "task_pi0"
    run_dir.mkdir()
    (run_dir / "episode_results_rebuild0.jsonl").write_text(
        json.dumps({
            "env_id": 0,
            "episode_in_env": 0,
            "success": True,
            "language_instruction": "<script>alert(1)</script>",
        })
        + "\n",
        encoding="utf-8",
    )
    video_name = format_episode_video_filename("robot-cam-rebuild0", 0, "wrist cam?rgb", 0)
    (run_dir / video_name).write_bytes(b"")

    build_report(tmp_path)

    run_page = (tmp_path / "report" / "job_task_pi0.html").read_text(encoding="utf-8")
    assert "wrist%20cam%3Frgb" in run_page
    assert "<script>alert(1)</script>" not in run_page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in run_page

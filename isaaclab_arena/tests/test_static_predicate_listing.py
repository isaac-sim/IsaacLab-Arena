# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test the episode_results annotation logic without building a real Task (no Isaac Sim needed).

``build_task_from_job_name``/``static_predicates_for_job`` themselves need the Isaac Lab / pxr
environment and are exercised indirectly via the injectable ``lookup_fn`` on ``_JobPredicateCache``.
"""

import json

from isaaclab_arena.progress_tracking.static_predicate_listing import (
    STATIC_PREDICATES_SUFFIX,
    _JobPredicateCache,
    annotate_episode_results_dir,
    annotate_episode_results_file,
)

_FAKE_PREDICATES = {
    "pick_and_place": {
        "pick_up_object": "apple",
        "destination_location": "wooden_bowl",
        "groups": {"default_group": [{"index": 0, "predicate": "objects_settled", "score": 1.0}]},
    }
}


def _fake_lookup(job_name: str, env_package: str) -> dict:
    if job_name == "unknown_job":
        raise AssertionError(f"No task yaml for job_name={job_name!r} under env_package={env_package!r}")
    return _FAKE_PREDICATES


def _write_jsonl(path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_annotate_preserves_line_correspondence(tmp_path):
    source = tmp_path / "episode_results_rebuild0.jsonl"
    _write_jsonl(
        source,
        [
            json.dumps({"job_name": "apple_and_yogurt_in_bowl", "env_id": 0, "episode_in_env": 0}),
            "",
            json.dumps({"job_name": "apple_and_yogurt_in_bowl", "env_id": 1, "episode_in_env": 0}),
        ],
    )

    cache = _JobPredicateCache("robolab", lookup_fn=_fake_lookup)
    output_path = annotate_episode_results_file(source, "robolab", cache=cache)

    assert output_path == source.with_name(f"episode_results_rebuild0{STATIC_PREDICATES_SUFFIX}")
    output_lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(output_lines) == 3
    assert output_lines[1] == ""

    first = json.loads(output_lines[0])
    assert first == {
        "job_name": "apple_and_yogurt_in_bowl",
        "env_id": 0,
        "episode_in_env": 0,
        "predicates": _FAKE_PREDICATES,
    }
    third = json.loads(output_lines[2])
    assert third["env_id"] == 1


def test_annotate_reports_per_line_errors_without_aborting(tmp_path):
    source = tmp_path / "episode_results_rebuild0.jsonl"
    _write_jsonl(
        source,
        [
            "not valid json",
            json.dumps({"no_job_name_field": True}),
            json.dumps({"job_name": "unknown_job"}),
            json.dumps({"job_name": "apple_and_yogurt_in_bowl"}),
        ],
    )

    cache = _JobPredicateCache("robolab", lookup_fn=_fake_lookup)
    output_path = annotate_episode_results_file(source, "robolab", cache=cache)
    output_lines = output_path.read_text(encoding="utf-8").splitlines()

    assert len(output_lines) == 4
    for line in output_lines[:3]:
        assert "error" in json.loads(line)
    assert "predicates" in json.loads(output_lines[3])


def test_job_predicate_cache_is_reused_across_files(tmp_path):
    calls: list[str] = []

    def _counting_lookup(job_name: str, env_package: str) -> dict:
        calls.append(job_name)
        return _FAKE_PREDICATES

    first = tmp_path / "a" / "episode_results_rebuild0.jsonl"
    second = tmp_path / "b" / "episode_results_rebuild0.jsonl"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    for path in (first, second):
        _write_jsonl(path, [json.dumps({"job_name": "apple_and_yogurt_in_bowl", "env_id": 0})])

    cache = _JobPredicateCache("robolab", lookup_fn=_counting_lookup)
    annotate_episode_results_file(first, "robolab", cache=cache)
    annotate_episode_results_file(second, "robolab", cache=cache)

    assert calls == ["apple_and_yogurt_in_bowl"]


def test_annotate_dir_finds_and_annotates_every_results_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "isaaclab_arena.progress_tracking.static_predicate_listing.static_predicates_for_job", _fake_lookup
    )

    task_a = tmp_path / "task_a"
    task_b = tmp_path / "task_b"
    task_a.mkdir()
    task_b.mkdir()
    _write_jsonl(task_a / "episode_results_rebuild0.jsonl", [json.dumps({"job_name": "task_a", "env_id": 0})])
    _write_jsonl(task_b / "episode_results_rank1.jsonl", [json.dumps({"job_name": "task_b", "env_id": 0})])
    # A report page under a "report" dir must not be picked up as a source file.
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    (report_dir / "episode_results_rebuild0.jsonl").write_text("{}\n", encoding="utf-8")

    output_paths = annotate_episode_results_dir(tmp_path, "robolab")

    assert {path.name for path in output_paths} == {
        f"episode_results_rebuild0{STATIC_PREDICATES_SUFFIX}",
        f"episode_results_rank1{STATIC_PREDICATES_SUFFIX}",
    }

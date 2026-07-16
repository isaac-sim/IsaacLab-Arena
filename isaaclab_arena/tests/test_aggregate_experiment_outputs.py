# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify aggregation of independently executed Arena Run outputs."""

import json

import pytest

from isaaclab_arena.evaluation.aggregate_experiment_outputs import aggregate_experiment_outputs


def _write_run_output(input_root, timestamp: str, run_name: str, success: bool):
    run_output = input_root / timestamp / run_name
    run_output.mkdir(parents=True)
    record = {
        "job_name": run_name,
        "env_id": 0,
        "episode_in_env": 0,
        "success": success,
    }
    (run_output / "episode_results_rebuild0.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (input_root / timestamp / "index.html").write_text("singleton report", encoding="utf-8")
    return run_output


def test_aggregates_run_directories_and_builds_one_report(tmp_path):
    first_input = tmp_path / "input-0"
    second_input = tmp_path / "input-1"
    first_run_output = _write_run_output(first_input, "2026-07-16_10-00-00", "first", True)
    second_run_output = _write_run_output(second_input, "2026-07-16_10-00-01", "second", False)
    output_dir = tmp_path / "output"

    report_path = aggregate_experiment_outputs(
        {"first": first_input, "second": second_input},
        output_dir,
    )

    assert report_path == output_dir / "index.html"
    assert (output_dir / "first/episode_results_rebuild0.jsonl").read_text(encoding="utf-8") == (
        first_run_output / "episode_results_rebuild0.jsonl"
    ).read_text(encoding="utf-8")
    assert (output_dir / "second/episode_results_rebuild0.jsonl").read_text(encoding="utf-8") == (
        second_run_output / "episode_results_rebuild0.jsonl"
    ).read_text(encoding="utf-8")
    assert not (output_dir / "2026-07-16_10-00-00").exists()
    report = report_path.read_text(encoding="utf-8")
    assert "first" in report
    assert "second" in report
    assert "2 job(s)" in report


def test_rejects_missing_or_ambiguous_run_output(tmp_path):
    with pytest.raises(AssertionError, match="found 0"):
        aggregate_experiment_outputs({"missing": tmp_path / "missing"}, tmp_path / "output")

    input_root = tmp_path / "input"
    _write_run_output(input_root, "2026-07-16_10-00-00", "first", True)
    _write_run_output(input_root, "2026-07-16_10-00-01", "first", True)
    with pytest.raises(AssertionError, match="found 2"):
        aggregate_experiment_outputs({"first": input_root}, tmp_path / "second-output")

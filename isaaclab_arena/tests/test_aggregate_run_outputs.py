# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify aggregation of canonical Arena Run output directories."""

import json
from pathlib import Path

import pytest

from isaaclab_arena.evaluation.aggregate_run_outputs import aggregate_run_outputs


def _write_run_output(run_output_directory: Path, run_name: str, success: bool) -> None:
    run_output_directory.mkdir(parents=True)
    record = {
        "job_name": run_name,
        "env_id": 0,
        "episode_in_env": 0,
        "success": success,
    }
    (run_output_directory / "episode_results_rebuild0.jsonl").write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )


def test_aggregates_run_directories_and_builds_one_report(tmp_path):
    first_run_output_directory = tmp_path / "runner-0-output" / "first"
    second_run_output_directory = tmp_path / "runner-1-output" / "second"
    _write_run_output(first_run_output_directory, "first", True)
    _write_run_output(second_run_output_directory, "second", False)
    combined_experiment_output_directory = tmp_path / "combined-experiment-output"

    report_path = aggregate_run_outputs(
        {
            "first": first_run_output_directory,
            "second": second_run_output_directory,
        },
        combined_experiment_output_directory,
    )

    assert report_path == combined_experiment_output_directory / "index.html"
    assert (combined_experiment_output_directory / "first/episode_results_rebuild0.jsonl").read_text(
        encoding="utf-8"
    ) == (first_run_output_directory / "episode_results_rebuild0.jsonl").read_text(encoding="utf-8")
    assert (combined_experiment_output_directory / "second/episode_results_rebuild0.jsonl").read_text(
        encoding="utf-8"
    ) == (second_run_output_directory / "episode_results_rebuild0.jsonl").read_text(encoding="utf-8")
    report_contents = report_path.read_text(encoding="utf-8")
    assert "first" in report_contents
    assert "second" in report_contents
    assert "2 job(s)" in report_contents


def test_rejects_missing_run_output_directory(tmp_path):
    missing_run_output_directory = tmp_path / "missing-run-output"

    with pytest.raises(AssertionError, match="Run 'missing' output directory does not exist"):
        aggregate_run_outputs(
            {"missing": missing_run_output_directory},
            tmp_path / "combined-experiment-output",
        )


def test_rejects_existing_destination_run_directory(tmp_path):
    source_run_output_directory = tmp_path / "runner-output" / "first"
    _write_run_output(source_run_output_directory, "first", True)
    combined_experiment_output_directory = tmp_path / "combined-experiment-output"
    existing_destination_directory = combined_experiment_output_directory / "first"
    existing_destination_directory.mkdir(parents=True)

    with pytest.raises(AssertionError, match="Run output destination already exists"):
        aggregate_run_outputs(
            {"first": source_run_output_directory},
            combined_experiment_output_directory,
        )

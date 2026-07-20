# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify discovery and assembly of timestamped Experiment outputs staged by OSMO."""

import json
from pathlib import Path

import pytest

from osmo.scripts.aggregate_experiment_results import (
    build_experiment_output_from_staged_experiment_runner_outputs,
    load_staged_experiment_runner_output_directories_by_run_name,
    resolve_run_output_directories_from_staged_experiment_runner_outputs,
)


def _write_run_output(run_output_directory: Path, run_name: str, success: bool) -> None:
    run_output_directory.mkdir(parents=True)
    episode_result = {
        "job_name": run_name,
        "env_id": 0,
        "episode_in_env": 0,
        "success": success,
    }
    (run_output_directory / "episode_results_rebuild0.jsonl").write_text(
        json.dumps(episode_result) + "\n",
        encoding="utf-8",
    )


def test_loads_staged_experiment_runner_output_directories_as_paths(tmp_path):
    serialized_output_directories_path = tmp_path / "staged-experiment-runner-output-directories.json"
    staged_output_directory = tmp_path / "experiment-runner-0-output"
    serialized_output_directories_path.write_text(
        json.dumps({"first": str(staged_output_directory)}),
        encoding="utf-8",
    )

    staged_output_directories_by_run_name = load_staged_experiment_runner_output_directories_by_run_name(
        serialized_output_directories_path
    )

    assert staged_output_directories_by_run_name == {"first": staged_output_directory}


def test_resolves_run_directory_below_one_generated_experiment_directory(tmp_path):
    staged_output_directory = tmp_path / "experiment-runner-0-output"
    expected_run_output_directory = staged_output_directory / "generated-experiment-output" / "first"
    expected_run_output_directory.mkdir(parents=True)
    (staged_output_directory / "runner.log").write_text("complete\n", encoding="utf-8")

    run_output_directories_by_name = resolve_run_output_directories_from_staged_experiment_runner_outputs(
        {"first": staged_output_directory}
    )

    assert run_output_directories_by_name == {"first": expected_run_output_directory}


def test_rejects_missing_staged_experiment_runner_output_directory(tmp_path):
    missing_staged_output_directory = tmp_path / "missing-experiment-runner-output"

    with pytest.raises(AssertionError, match="Staged Experiment Runner output directory.*does not exist"):
        resolve_run_output_directories_from_staged_experiment_runner_outputs({"first": missing_staged_output_directory})


def test_rejects_staged_output_without_the_requested_run(tmp_path):
    staged_output_directory = tmp_path / "experiment-runner-0-output"
    (staged_output_directory / "2026-07-20_12-00-00" / "another-run").mkdir(parents=True)

    with pytest.raises(AssertionError, match="Expected exactly one Run output.*found 0"):
        resolve_run_output_directories_from_staged_experiment_runner_outputs({"first": staged_output_directory})


def test_does_not_search_below_the_generated_experiment_directory(tmp_path):
    staged_output_directory = tmp_path / "experiment-runner-0-output"
    deeply_nested_run_output_directory = (
        staged_output_directory / "unexpected-wrapper" / "2026-07-20_12-00-00" / "first"
    )
    deeply_nested_run_output_directory.mkdir(parents=True)

    with pytest.raises(AssertionError, match="Expected exactly one Run output.*found 0"):
        resolve_run_output_directories_from_staged_experiment_runner_outputs({"first": staged_output_directory})


def test_rejects_multiple_generated_experiment_directories_containing_the_run(tmp_path):
    staged_output_directory = tmp_path / "experiment-runner-0-output"
    (staged_output_directory / "2026-07-20_12-00-00" / "first").mkdir(parents=True)
    (staged_output_directory / "2026-07-20_12-01-00" / "first").mkdir(parents=True)

    with pytest.raises(AssertionError, match="Expected exactly one Run output.*found 2"):
        resolve_run_output_directories_from_staged_experiment_runner_outputs({"first": staged_output_directory})


def test_builds_experiment_output_from_separate_staged_experiment_runner_outputs(tmp_path):
    first_staged_output_directory = tmp_path / "experiment-runner-0-output"
    second_staged_output_directory = tmp_path / "experiment-runner-1-output"
    first_run_output_directory = first_staged_output_directory / "2026-07-20_12-00-00" / "first"
    second_run_output_directory = second_staged_output_directory / "2026-07-20_12-00-00" / "second"
    _write_run_output(first_run_output_directory, "first", True)
    _write_run_output(second_run_output_directory, "second", False)
    combined_experiment_output_directory = tmp_path / "combined-experiment-output"

    report_path = build_experiment_output_from_staged_experiment_runner_outputs(
        {
            "first": first_staged_output_directory,
            "second": second_staged_output_directory,
        },
        combined_experiment_output_directory,
    )

    assert report_path == combined_experiment_output_directory / "index.html"
    assert (combined_experiment_output_directory / "first/episode_results_rebuild0.jsonl").is_file()
    assert (combined_experiment_output_directory / "second/episode_results_rebuild0.jsonl").is_file()
    report_contents = report_path.read_text(encoding="utf-8")
    assert "first" in report_contents
    assert "second" in report_contents
    assert "2 job(s)" in report_contents

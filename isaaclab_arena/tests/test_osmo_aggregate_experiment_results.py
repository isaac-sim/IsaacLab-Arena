# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify assembly of exact single-Run Experiment outputs staged by OSMO."""

import json
from pathlib import Path

import pytest

from osmo.scripts.aggregate_experiment_results import (
    build_experiment_output_from_staged_experiment_runner_outputs,
    load_staged_experiment_runner_output_directories_by_run_name,
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


def test_rejects_staged_experiment_output_without_the_requested_run(tmp_path):
    staged_output_directory = tmp_path / "experiment-runner-0-output"
    (staged_output_directory / "another-run").mkdir(parents=True)

    with pytest.raises(AssertionError, match="Expected Run output directory for Run 'first'.*first"):
        build_experiment_output_from_staged_experiment_runner_outputs(
            {"first": staged_output_directory},
            tmp_path / "combined-experiment-output",
        )


def test_builds_experiment_output_from_separate_staged_experiment_runner_outputs(tmp_path):
    first_staged_output_directory = tmp_path / "experiment-runner-0-output"
    second_staged_output_directory = tmp_path / "experiment-runner-1-output"
    first_run_output_directory = first_staged_output_directory / "first"
    second_run_output_directory = second_staged_output_directory / "second"
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

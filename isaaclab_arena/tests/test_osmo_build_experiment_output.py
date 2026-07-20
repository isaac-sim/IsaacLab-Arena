# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify building one Experiment output from exact Experiment Runner task outputs."""

import json
from pathlib import Path

import pytest

from osmo.scripts.build_experiment_output import (
    build_experiment_output,
    collect_run_outputs_into_experiment_output,
    load_experiment_runner_output_directories_by_run_name,
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


def test_loads_experiment_runner_output_directories_as_paths(tmp_path):
    experiment_runner_output_directories_file_path = tmp_path / "experiment-runner-output-directories.json"
    experiment_runner_output_directory = tmp_path / "experiment-runner-0-output"
    experiment_runner_output_directories_file_path.write_text(
        json.dumps({"first": str(experiment_runner_output_directory)}),
        encoding="utf-8",
    )

    experiment_runner_output_directories_by_run_name = load_experiment_runner_output_directories_by_run_name(
        experiment_runner_output_directories_file_path
    )

    assert experiment_runner_output_directories_by_run_name == {"first": experiment_runner_output_directory}


def test_rejects_experiment_runner_output_without_the_requested_run(tmp_path):
    experiment_runner_output_directory = tmp_path / "experiment-runner-0-output"
    (experiment_runner_output_directory / "another-run").mkdir(parents=True)

    with pytest.raises(AssertionError, match="Expected Run output directory for Run 'first'.*first"):
        collect_run_outputs_into_experiment_output(
            {"first": experiment_runner_output_directory},
            tmp_path / "experiment-output",
        )


def test_collects_run_outputs_without_building_report(tmp_path):
    experiment_runner_output_directory = tmp_path / "experiment-runner-0-output"
    _write_run_output(experiment_runner_output_directory / "first", "first", True)
    experiment_output_directory = tmp_path / "experiment-output"

    collect_run_outputs_into_experiment_output(
        {"first": experiment_runner_output_directory},
        experiment_output_directory,
    )

    assert (experiment_output_directory / "first/episode_results_rebuild0.jsonl").is_file()
    assert not (experiment_output_directory / "index.html").exists()


def test_builds_experiment_output_from_separate_experiment_runner_outputs(tmp_path):
    first_experiment_runner_output_directory = tmp_path / "experiment-runner-0-output"
    second_experiment_runner_output_directory = tmp_path / "experiment-runner-1-output"
    first_run_output_directory = first_experiment_runner_output_directory / "first"
    second_run_output_directory = second_experiment_runner_output_directory / "second"
    _write_run_output(first_run_output_directory, "first", True)
    _write_run_output(second_run_output_directory, "second", False)
    experiment_output_directory = tmp_path / "experiment-output"

    report_path = build_experiment_output(
        {
            "first": first_experiment_runner_output_directory,
            "second": second_experiment_runner_output_directory,
        },
        experiment_output_directory,
    )

    assert report_path == experiment_output_directory / "index.html"
    assert (experiment_output_directory / "first/episode_results_rebuild0.jsonl").is_file()
    assert (experiment_output_directory / "second/episode_results_rebuild0.jsonl").is_file()
    report_contents = report_path.read_text(encoding="utf-8")
    assert "first" in report_contents
    assert "second" in report_contents
    assert "2 job(s)" in report_contents

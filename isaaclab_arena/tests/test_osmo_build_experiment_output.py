# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify building one Experiment output from exact Experiment Runner task outputs."""

import json
from pathlib import Path

import pytest

from isaaclab_arena.evaluation.arena_experiment_result import ARENA_EXPERIMENT_RESULT_FILENAME
from isaaclab_arena.evaluation.arena_run import RunStatus
from isaaclab_arena.visualization.report import RunExecutionReport
from osmo.scripts.build_experiment_output import (
    EXPERIMENT_RUNNER_RESULT_FILE_NAME,
    build_experiment_output,
    collect_run_outputs_into_experiment_output,
    load_experiment_runner_output_directories_by_run_name,
    load_experiment_runner_result,
)


def _run_metadata(environment_name: str, policy_variant: str) -> dict[str, object]:
    return {
        "environment": {
            "name": environment_name,
            "definition": f"definitions/{environment_name}.yaml",
        },
        "policy_variant": policy_variant,
    }


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


def _write_experiment_runner_result(
    experiment_runner_output_directory: Path,
    execution_status: RunStatus,
    process_exit_code: int,
    run_metadata_by_name: dict[str, dict[str, object]],
) -> None:
    experiment_runner_output_directory.mkdir(parents=True, exist_ok=True)
    (experiment_runner_output_directory / EXPERIMENT_RUNNER_RESULT_FILE_NAME).write_text(
        json.dumps({
            "execution_status": execution_status.value,
            "process_exit_code": process_exit_code,
            "runs": run_metadata_by_name,
        })
        + "\n",
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


def test_loads_experiment_runner_result(tmp_path):
    experiment_runner_output_directory = tmp_path / "experiment-runner-output"
    expected_run_metadata = _run_metadata("first-environment", "pi05")
    _write_experiment_runner_result(
        experiment_runner_output_directory,
        RunStatus.COMPLETED,
        0,
        {"first": expected_run_metadata},
    )

    experiment_runner_result, run_metadata = load_experiment_runner_result(
        experiment_runner_output_directory,
        "first",
    )

    assert experiment_runner_result == RunExecutionReport(
        run_name="first",
        status=RunStatus.COMPLETED,
        process_exit_code=0,
    )
    assert run_metadata == expected_run_metadata


def test_rejects_inconsistent_experiment_runner_result(tmp_path):
    experiment_runner_output_directory = tmp_path / "experiment-runner-output"
    _write_experiment_runner_result(
        experiment_runner_output_directory,
        RunStatus.COMPLETED,
        1,
        {"first": _run_metadata("first-environment", "pi05")},
    )

    with pytest.raises(AssertionError, match="Experiment Runner result for Run 'first' is inconsistent"):
        load_experiment_runner_result(experiment_runner_output_directory, "first")


def test_rejects_experiment_runner_result_for_another_run(tmp_path):
    experiment_runner_output_directory = tmp_path / "experiment-runner-output"
    _write_experiment_runner_result(
        experiment_runner_output_directory,
        RunStatus.COMPLETED,
        0,
        {"another-run": _run_metadata("another-environment", "cosmos")},
    )

    with pytest.raises(AssertionError, match="must contain metadata for exactly that Run"):
        load_experiment_runner_result(experiment_runner_output_directory, "first")


def test_rejects_completed_experiment_runner_output_without_the_requested_run(tmp_path):
    experiment_runner_output_directory = tmp_path / "experiment-runner-0-output"
    (experiment_runner_output_directory / "another-run").mkdir(parents=True)
    _write_experiment_runner_result(
        experiment_runner_output_directory,
        RunStatus.COMPLETED,
        0,
        {"first": _run_metadata("first-environment", "pi05")},
    )

    with pytest.raises(AssertionError, match="Completed Run 'first' is missing its expected output directory"):
        collect_run_outputs_into_experiment_output(
            {"first": experiment_runner_output_directory},
            tmp_path / "experiment-output",
        )


def test_collects_run_outputs_without_building_report(tmp_path):
    experiment_runner_output_directory = tmp_path / "experiment-runner-0-output"
    _write_run_output(experiment_runner_output_directory / "first", "first", True)
    first_run_metadata = _run_metadata("first-environment", "pi05")
    _write_experiment_runner_result(
        experiment_runner_output_directory,
        RunStatus.COMPLETED,
        0,
        {"first": first_run_metadata},
    )
    experiment_output_directory = tmp_path / "experiment-output"

    run_execution_reports, run_metadata_by_name = collect_run_outputs_into_experiment_output(
        {"first": experiment_runner_output_directory},
        experiment_output_directory,
    )

    assert run_execution_reports == [
        RunExecutionReport(run_name="first", status=RunStatus.COMPLETED, process_exit_code=0)
    ]
    assert run_metadata_by_name == {
        "first": {
            **first_run_metadata,
            "status": RunStatus.COMPLETED.value,
        }
    }
    assert (experiment_output_directory / "first/episode_results_rebuild0.jsonl").is_file()
    assert (experiment_output_directory / "first" / EXPERIMENT_RUNNER_RESULT_FILE_NAME).is_file()
    assert not (experiment_output_directory / "index.html").exists()


def test_builds_experiment_output_from_separate_experiment_runner_outputs(tmp_path):
    first_experiment_runner_output_directory = tmp_path / "experiment-runner-0-output"
    second_experiment_runner_output_directory = tmp_path / "experiment-runner-1-output"
    first_run_output_directory = first_experiment_runner_output_directory / "first"
    second_run_output_directory = second_experiment_runner_output_directory / "second"
    _write_run_output(first_run_output_directory, "first", True)
    _write_run_output(second_run_output_directory, "second", False)
    _write_experiment_runner_result(
        first_experiment_runner_output_directory,
        RunStatus.COMPLETED,
        0,
        {"first": _run_metadata("first-environment", "pi05")},
    )
    _write_experiment_runner_result(
        second_experiment_runner_output_directory,
        RunStatus.COMPLETED,
        0,
        {"second": _run_metadata("second-environment", "cosmos")},
    )
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
    experiment_result = json.loads(
        (experiment_output_directory / ARENA_EXPERIMENT_RESULT_FILENAME).read_text(encoding="utf-8")
    )
    assert list(experiment_result["runs"]) == ["first", "second"]
    first_run_result = experiment_result["runs"]["first"]
    assert first_run_result["environment"] == {
        "name": "first-environment",
        "definition": "definitions/first-environment.yaml",
    }
    assert first_run_result["policy_variant"] == "pi05"
    assert first_run_result["status"] == "completed"
    assert first_run_result["rebuilds"] == [{
        "index": 0,
        "episodes": [{
            "job_name": "first",
            "env_id": 0,
            "episode_in_env": 0,
            "success": True,
        }],
    }]
    assert set(first_run_result) == {"environment", "policy_variant", "status", "rebuilds"}
    assert experiment_result["runs"]["second"]["policy_variant"] == "cosmos"
    report_contents = report_path.read_text(encoding="utf-8")
    assert "first" in report_contents
    assert "second" in report_contents
    assert "2 run(s) &middot; 2 completed &middot; 0 failed &middot; 2 episode(s)" in report_contents
    assert "Failed runs" not in report_contents


def test_reports_failed_runner_without_its_partial_artifacts(tmp_path):
    completed_runner_output_directory = tmp_path / "completed-runner-output"
    failed_runner_output_directory = tmp_path / "failed-runner-output"
    _write_run_output(completed_runner_output_directory / "completed-run", "completed-run", True)
    _write_run_output(failed_runner_output_directory / "failed-run", "failed-run", False)
    _write_experiment_runner_result(
        completed_runner_output_directory,
        RunStatus.COMPLETED,
        0,
        {"completed-run": _run_metadata("completed-environment", "pi05")},
    )
    failed_run_metadata = _run_metadata("failed-environment", "cosmos")
    _write_experiment_runner_result(
        failed_runner_output_directory,
        RunStatus.FAILED,
        17,
        {"failed-run": failed_run_metadata},
    )
    experiment_output_directory = tmp_path / "experiment-output"

    report_path = build_experiment_output(
        {
            "completed-run": completed_runner_output_directory,
            "failed-run": failed_runner_output_directory,
        },
        experiment_output_directory,
    )

    assert (experiment_output_directory / "completed-run/episode_results_rebuild0.jsonl").is_file()
    assert not (experiment_output_directory / "failed-run/episode_results_rebuild0.jsonl").exists()
    failed_result_path = experiment_output_directory / "failed-run" / EXPERIMENT_RUNNER_RESULT_FILE_NAME
    assert json.loads(failed_result_path.read_text(encoding="utf-8")) == {
        "execution_status": RunStatus.FAILED.value,
        "process_exit_code": 17,
        "runs": {"failed-run": failed_run_metadata},
    }
    experiment_result = json.loads(
        (experiment_output_directory / ARENA_EXPERIMENT_RESULT_FILENAME).read_text(encoding="utf-8")
    )
    assert experiment_result["runs"]["completed-run"]["status"] == "completed"
    failed_run_result = experiment_result["runs"]["failed-run"]
    assert failed_run_result["status"] == "failed"
    assert failed_run_result["rebuilds"] == []
    assert set(failed_run_result) == {"environment", "policy_variant", "status", "rebuilds"}
    report_contents = report_path.read_text(encoding="utf-8")
    assert "completed-run" in report_contents
    assert "failed-run" in report_contents
    assert "2 run(s) &middot; 1 completed &middot; 1 failed &middot; 1 episode(s)" in report_contents
    assert "Failed runs (1)" in report_contents
    assert "<code>17</code>" in report_contents
    assert "These runs did not complete and are excluded from episode results." in report_contents


def test_builds_failure_report_when_every_runner_failed(tmp_path):
    first_runner_output_directory = tmp_path / "first-runner-output"
    second_runner_output_directory = tmp_path / "second-runner-output"
    _write_experiment_runner_result(
        first_runner_output_directory,
        RunStatus.FAILED,
        1,
        {"first": _run_metadata("first-environment", "pi05")},
    )
    _write_experiment_runner_result(
        second_runner_output_directory,
        RunStatus.FAILED,
        2,
        {"second": _run_metadata("second-environment", "cosmos")},
    )
    experiment_output_directory = tmp_path / "experiment-output"

    report_path = build_experiment_output(
        {
            "first": first_runner_output_directory,
            "second": second_runner_output_directory,
        },
        experiment_output_directory,
    )

    assert (experiment_output_directory / "first" / EXPERIMENT_RUNNER_RESULT_FILE_NAME).is_file()
    assert (experiment_output_directory / "second" / EXPERIMENT_RUNNER_RESULT_FILE_NAME).is_file()
    experiment_result = json.loads(
        (experiment_output_directory / ARENA_EXPERIMENT_RESULT_FILENAME).read_text(encoding="utf-8")
    )
    assert experiment_result["runs"]["first"]["status"] == "failed"
    assert experiment_result["runs"]["second"]["status"] == "failed"
    assert all("process_exit_code" not in run_result for run_result in experiment_result["runs"].values())
    report_contents = report_path.read_text(encoding="utf-8")
    assert "first" in report_contents
    assert "second" in report_contents
    assert "2 run(s) &middot; 0 completed &middot; 2 failed &middot; 0 episode(s)" in report_contents
    assert "Failed runs (2)" in report_contents
    assert "<code>1</code>" in report_contents
    assert "<code>2</code>" in report_contents
    assert "No results recorded yet." not in report_contents

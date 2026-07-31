# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify coordinating one OSMO report from successful Experiment Runs."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from osmo.scripts import coordinate_experiment_output as coordinator

EXPERIMENT_RUNNER_TASK_NAMES_BY_RUN_NAME = {
    "first": "experiment-runner-0",
    "second": "experiment-runner-1",
    "third": "experiment-runner-2",
}
SOURCE_WORKFLOW_OUTPUT_URL = "https://storage.example.com/v1/team/workflows/source-workflow-7"


def _workflow_with_runner_statuses(
    workflow_status: str,
    statuses_by_task_name: dict[str, str],
    output_url: str = SOURCE_WORKFLOW_OUTPUT_URL,
) -> dict:
    groups = [
        {
            "name": f"arena-run-{run_index}",
            "tasks": [{"name": task_name, "status": task_status}],
        }
        for run_index, (task_name, task_status) in enumerate(statuses_by_task_name.items())
    ]
    return {"status": workflow_status, "outputs": output_url, "groups": groups}


@pytest.mark.parametrize(
    ("statuses_by_task_name", "expected_completed_runs"),
    [
        (
            {
                "experiment-runner-0": "COMPLETED",
                "experiment-runner-1": "COMPLETED",
                "experiment-runner-2": "COMPLETED",
            },
            EXPERIMENT_RUNNER_TASK_NAMES_BY_RUN_NAME,
        ),
        (
            {
                "experiment-runner-0": "COMPLETED",
                "experiment-runner-1": "FAILED",
                "experiment-runner-2": "FAILED_IMAGE_PULL",
            },
            {"first": "experiment-runner-0"},
        ),
        (
            {
                "experiment-runner-0": "FAILED_EVICTED",
                "experiment-runner-2": "FAILED_CANCELED",
            },
            {},
        ),
    ],
)
def test_selects_only_completed_experiment_runners(statuses_by_task_name, expected_completed_runs):
    workflow = _workflow_with_runner_statuses("FAILED", statuses_by_task_name)

    actual_statuses_by_task_name = coordinator.get_experiment_runner_statuses(
        workflow,
        EXPERIMENT_RUNNER_TASK_NAMES_BY_RUN_NAME,
    )

    assert actual_statuses_by_task_name == statuses_by_task_name
    assert (
        coordinator.find_completed_experiment_runners(
            EXPERIMENT_RUNNER_TASK_NAMES_BY_RUN_NAME,
            actual_statuses_by_task_name,
        )
        == expected_completed_runs
    )


def test_waits_for_source_workflow_terminal_status_instead_of_every_task(monkeypatch):
    source_workflow_responses = iter([
        _workflow_with_runner_statuses(
            "RUNNING",
            {"experiment-runner-0": "COMPLETED", "experiment-runner-1": "RUNNING"},
        ),
        _workflow_with_runner_statuses(
            "FAILED_SUBMISSION",
            {"experiment-runner-0": "COMPLETED"},
        ),
    ])
    sleep_intervals = []
    monkeypatch.setattr(coordinator, "query_workflow", lambda _workflow_id: next(source_workflow_responses))
    monkeypatch.setattr(coordinator.time, "sleep", sleep_intervals.append)

    terminal_workflow = coordinator.wait_for_source_workflow_to_finish(
        "source-workflow-7",
        EXPERIMENT_RUNNER_TASK_NAMES_BY_RUN_NAME,
        poll_interval_seconds=7,
    )

    assert terminal_workflow["status"] == "FAILED_SUBMISSION"
    assert sleep_intervals == [7]
    assert coordinator.find_completed_experiment_runners(
        EXPERIMENT_RUNNER_TASK_NAMES_BY_RUN_NAME,
        coordinator.get_experiment_runner_statuses(
            terminal_workflow,
            EXPERIMENT_RUNNER_TASK_NAMES_BY_RUN_NAME,
        ),
    ) == {"first": "experiment-runner-0"}


def test_retries_transient_osmo_query_failures(monkeypatch):
    command_results = iter([
        SimpleNamespace(returncode=1, stdout="", stderr="temporary service error"),
        SimpleNamespace(returncode=0, stdout='{"status": "RUNNING"}', stderr=""),
    ])
    sleep_intervals = []
    monkeypatch.setattr(coordinator.subprocess, "run", lambda command, **kwargs: next(command_results))
    monkeypatch.setattr(coordinator.time, "sleep", sleep_intervals.append)

    response = coordinator.run_osmo_json_command(
        ["osmo", "workflow", "query", "source-workflow-7"],
        maximum_attempts=2,
        retry_delay_seconds=3,
    )

    assert response == {"status": "RUNNING"}
    assert sleep_intervals == [3]


def test_builds_completed_runner_output_url():
    assert (
        coordinator.experiment_runner_output_url(
            f"{SOURCE_WORKFLOW_OUTPUT_URL}/",
            "experiment-runner-2",
        )
        == f"{SOURCE_WORKFLOW_OUTPUT_URL}/experiment-runner-2/"
    )

    with pytest.raises(AssertionError, match=r"HTTP\(S\)"):
        coordinator.experiment_runner_output_url("swift://bucket/workflow", "experiment-runner-2")


def test_downloads_runner_output_without_preserving_remote_prefix(monkeypatch, tmp_path):
    command_results = iter([
        SimpleNamespace(returncode=4, stdout="", stderr="temporary network error"),
        SimpleNamespace(returncode=0, stdout="", stderr=""),
    ])
    commands = []
    sleep_intervals = []

    def capture_download(command, **kwargs):
        commands.append(command)
        return next(command_results)

    monkeypatch.setattr(coordinator.subprocess, "run", capture_download)
    monkeypatch.setattr(coordinator.time, "sleep", sleep_intervals.append)
    destination_directory = tmp_path / "runner-output"

    coordinator.download_experiment_runner_output(
        f"{SOURCE_WORKFLOW_OUTPUT_URL}/experiment-runner-0/",
        destination_directory,
        maximum_attempts=2,
        retry_delay_seconds=4,
    )

    assert destination_directory.is_dir()
    assert commands[0] == commands[1]
    assert commands[0][:4] == ["wget", "--recursive", "--no-parent", "--no-host-directories"]
    assert "--cut-dirs=5" in commands[0]
    assert f"--directory-prefix={destination_directory}" in commands[0]
    assert commands[0][-1] == f"{SOURCE_WORKFLOW_OUTPUT_URL}/experiment-runner-0/"
    assert sleep_intervals == [4]


def test_downloads_only_completed_runner_outputs(monkeypatch, tmp_path):
    downloads = []

    def capture_download(output_url, destination_directory):
        downloads.append((output_url, destination_directory))
        run_name = "first" if output_url.endswith("experiment-runner-0/") else "third"
        (destination_directory / run_name).mkdir(parents=True)

    monkeypatch.setattr(coordinator, "download_experiment_runner_output", capture_download)

    downloaded_directories = coordinator.download_completed_experiment_runner_outputs(
        SOURCE_WORKFLOW_OUTPUT_URL,
        {"first": "experiment-runner-0", "third": "experiment-runner-2"},
        tmp_path / "downloads",
    )

    assert downloads == [
        (
            f"{SOURCE_WORKFLOW_OUTPUT_URL}/experiment-runner-0/",
            tmp_path / "downloads/experiment-runner-0",
        ),
        (
            f"{SOURCE_WORKFLOW_OUTPUT_URL}/experiment-runner-2/",
            tmp_path / "downloads/experiment-runner-2",
        ),
    ]
    assert downloaded_directories == {
        "first": tmp_path / "downloads/experiment-runner-0",
        "third": tmp_path / "downloads/experiment-runner-2",
    }


def test_skips_unavailable_or_incomplete_completed_runner_outputs(monkeypatch, tmp_path):
    def capture_download(output_url, destination_directory):
        if output_url.endswith("experiment-runner-0/"):
            raise RuntimeError("stored output is unavailable")
        destination_directory.mkdir(parents=True)
        if output_url.endswith("experiment-runner-2/"):
            (destination_directory / "third").mkdir()

    monkeypatch.setattr(coordinator, "download_experiment_runner_output", capture_download)

    downloaded_directories = coordinator.download_completed_experiment_runner_outputs(
        SOURCE_WORKFLOW_OUTPUT_URL,
        EXPERIMENT_RUNNER_TASK_NAMES_BY_RUN_NAME,
        tmp_path / "downloads",
    )

    assert downloaded_directories == {"third": tmp_path / "downloads/experiment-runner-2"}


def test_runs_builder_with_downloaded_output_mapping(monkeypatch, tmp_path):
    captured_command = None
    captured_output_directories = None

    def capture_builder(command, **kwargs):
        nonlocal captured_command, captured_output_directories
        captured_command = command
        mapping_file_path = Path(command[command.index("--experiment-runner-output-directories-file") + 1])
        captured_output_directories = json.loads(mapping_file_path.read_text(encoding="utf-8"))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(coordinator.subprocess, "run", capture_builder)
    experiment_output_directory = tmp_path / "combined-output"
    builder_script_path = tmp_path / "build-output.py"
    coordinator.build_experiment_output(
        {"first": tmp_path / "runner-0", "third": tmp_path / "runner-2"},
        experiment_output_directory,
        builder_script_path,
        tmp_path,
    )

    assert captured_output_directories == {
        "first": str(tmp_path / "runner-0"),
        "third": str(tmp_path / "runner-2"),
    }
    assert captured_command[-2:] == ["--experiment-output-directory", str(experiment_output_directory)]


@pytest.mark.parametrize(
    ("terminal_workflow", "expected_completed_runs"),
    [
        (
            _workflow_with_runner_statuses(
                "FAILED",
                {
                    "experiment-runner-0": "COMPLETED",
                    "experiment-runner-1": "FAILED_IMAGE_PULL",
                    "experiment-runner-2": "COMPLETED",
                },
            ),
            {"first": "experiment-runner-0", "third": "experiment-runner-2"},
        ),
        (
            _workflow_with_runner_statuses("FAILED_SUBMISSION", {}, output_url=""),
            {},
        ),
    ],
)
def test_coordinates_partial_or_empty_successes(
    monkeypatch,
    tmp_path,
    terminal_workflow,
    expected_completed_runs,
):
    settings_file_path = tmp_path / "coordinator-settings.json"
    settings_file_path.write_text(
        json.dumps({
            "experiment_runner_task_names_by_run_name": EXPERIMENT_RUNNER_TASK_NAMES_BY_RUN_NAME,
            "poll_interval_seconds": 5,
        }),
        encoding="utf-8",
    )
    downloaded_runs = None
    built_output_directories = None

    monkeypatch.setattr(
        coordinator,
        "wait_for_source_workflow_to_finish",
        lambda *_args, **_kwargs: terminal_workflow,
    )

    def capture_downloads(source_output_url, completed_runs, download_root_directory):
        nonlocal downloaded_runs
        assert source_output_url == SOURCE_WORKFLOW_OUTPUT_URL
        downloaded_runs = dict(completed_runs)
        return {run_name: download_root_directory / run_name for run_name in completed_runs}

    def capture_build(output_directories, experiment_output_directory, build_script_path, temporary_directory):
        nonlocal built_output_directories
        built_output_directories = dict(output_directories)
        assert experiment_output_directory == tmp_path / "combined-output"
        assert build_script_path == tmp_path / "build-output.py"
        assert temporary_directory.is_dir()

    monkeypatch.setattr(coordinator, "download_completed_experiment_runner_outputs", capture_downloads)
    monkeypatch.setattr(coordinator, "build_experiment_output", capture_build)

    completed_runs = coordinator.coordinate_experiment_output(
        "source-workflow-7",
        settings_file_path,
        tmp_path / "build-output.py",
        tmp_path / "combined-output",
    )

    assert completed_runs == expected_completed_runs
    if expected_completed_runs:
        assert downloaded_runs == expected_completed_runs
        assert set(built_output_directories) == set(expected_completed_runs)
    else:
        assert downloaded_runs is None
        assert built_output_directories == {}

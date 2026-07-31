# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build one report from every successful Run in an OSMO Experiment workflow.

The NVIDIA OSMO deployment exposes internally stored task outputs as HTTP directories under the workflow's
``outputs`` URL. The coordinator uses that directory tree to collect completed Runs without declaring them as task
dependencies.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

_ACTIVE_WORKFLOW_STATUSES = frozenset({"PENDING", "RUNNING", "WAITING"})
_EXPERIMENT_RUNNER_OUTPUT_DIRECTORIES_FILE_NAME = "experiment_runner_output_directories.json"


def run_osmo_json_command(
    command: list[str],
    maximum_attempts: int = 5,
    retry_delay_seconds: float = 5,
) -> dict[str, Any]:
    """Run an OSMO command and retry transient failures before returning its JSON response."""
    assert maximum_attempts > 0, "OSMO command maximum attempts must be positive"
    assert retry_delay_seconds >= 0, "OSMO command retry delay must not be negative"

    failure_message = ""
    for attempt_number in range(1, maximum_attempts + 1):
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            try:
                response = json.loads(result.stdout)
            except json.JSONDecodeError:
                failure_message = f"OSMO command returned invalid JSON: {shlex.join(command)}"
            else:
                assert isinstance(response, dict), f"OSMO command must return a JSON object: {shlex.join(command)}"
                return response
        else:
            failure_message = (
                f"OSMO command failed with exit code {result.returncode}: {shlex.join(command)}\n"
                f"{result.stderr.strip()}"
            )

        if attempt_number < maximum_attempts:
            print(
                f"OSMO query attempt {attempt_number} failed; retrying in {retry_delay_seconds:g} seconds.",
                flush=True,
            )
            time.sleep(retry_delay_seconds)

    raise RuntimeError(failure_message)


def query_workflow(workflow_id: str) -> dict[str, Any]:
    """Return one workflow's current OSMO status response."""
    return run_osmo_json_command([
        "osmo",
        "workflow",
        "query",
        workflow_id,
        "--format-type",
        "json",
    ])


def get_experiment_runner_statuses(
    workflow: Mapping[str, Any],
    experiment_runner_task_names_by_run_name: Mapping[str, str],
) -> dict[str, str]:
    """Return current OSMO statuses for the expected Experiment Runner tasks."""
    expected_task_names = set(experiment_runner_task_names_by_run_name.values())
    statuses_by_task_name: dict[str, str] = {}
    for group in workflow.get("groups", []):
        for task in group.get("tasks", []):
            task_name = task.get("name")
            if task_name not in expected_task_names:
                continue
            assert task_name not in statuses_by_task_name, f"Workflow contains duplicate task name '{task_name}'"
            task_status = task.get("status")
            assert isinstance(task_status, str) and task_status, f"Task '{task_name}' has no OSMO status"
            statuses_by_task_name[task_name] = task_status
    return statuses_by_task_name


def find_completed_experiment_runners(
    experiment_runner_task_names_by_run_name: Mapping[str, str],
    statuses_by_task_name: Mapping[str, str],
) -> dict[str, str]:
    """Return Run names mapped to Experiment Runner tasks that OSMO completed successfully."""
    return {
        run_name: task_name
        for run_name, task_name in experiment_runner_task_names_by_run_name.items()
        if statuses_by_task_name.get(task_name) == "COMPLETED"
    }


def wait_for_source_workflow_to_finish(
    source_workflow_id: str,
    experiment_runner_task_names_by_run_name: Mapping[str, str],
    poll_interval_seconds: float,
) -> dict[str, Any]:
    """Wait for the source workflow to finish and return its terminal OSMO response."""
    previous_workflow_status: str | None = None
    previous_runner_statuses: dict[str, str] | None = None
    while True:
        workflow = query_workflow(source_workflow_id)
        workflow_status = workflow.get("status")
        assert isinstance(workflow_status, str) and workflow_status, "Source workflow has no OSMO status"
        runner_statuses = get_experiment_runner_statuses(workflow, experiment_runner_task_names_by_run_name)

        if workflow_status != previous_workflow_status or runner_statuses != previous_runner_statuses:
            runner_status_summary = ", ".join(
                f"{run_name}={runner_statuses.get(task_name, 'NOT_FOUND')}"
                for run_name, task_name in experiment_runner_task_names_by_run_name.items()
            )
            print(f"Source workflow status: {workflow_status}; Runs: {runner_status_summary}", flush=True)
            previous_workflow_status = workflow_status
            previous_runner_statuses = runner_statuses

        if workflow_status not in _ACTIVE_WORKFLOW_STATUSES:
            return workflow
        time.sleep(poll_interval_seconds)


def experiment_runner_output_url(source_workflow_output_url: str, experiment_runner_task_name: str) -> str:
    """Return the OSMO-hosted output URL for one completed Experiment Runner task."""
    parsed_source_url = urlsplit(source_workflow_output_url)
    assert (
        parsed_source_url.scheme in {"http", "https"} and parsed_source_url.netloc
    ), f"OSMO source workflow outputs must be exposed through an HTTP(S) URL; got '{source_workflow_output_url}'"
    encoded_task_name = quote(experiment_runner_task_name, safe="-_.~")
    return f"{source_workflow_output_url.rstrip('/')}/{encoded_task_name}/"


def download_experiment_runner_output(
    experiment_runner_output_url: str,
    destination_directory: Path,
    maximum_attempts: int = 3,
    retry_delay_seconds: float = 10,
) -> None:
    """Recursively download one completed Experiment Runner output from OSMO storage."""
    assert maximum_attempts > 0, "Output download maximum attempts must be positive"
    assert retry_delay_seconds >= 0, "Output download retry delay must not be negative"
    parsed_output_url = urlsplit(experiment_runner_output_url)
    path_segment_count = len([segment for segment in parsed_output_url.path.split("/") if segment])
    destination_directory.mkdir(parents=True, exist_ok=False)
    command = [
        "wget",
        "--recursive",
        "--no-parent",
        "--no-host-directories",
        f"--cut-dirs={path_segment_count}",
        f"--directory-prefix={destination_directory}",
        "--continue",
        "--quiet",
        experiment_runner_output_url,
    ]

    failure_message = ""
    for attempt_number in range(1, maximum_attempts + 1):
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            return
        failure_message = (
            f"Experiment Runner output download failed with exit code {result.returncode}: {shlex.join(command)}\n"
            f"{result.stderr.strip()}"
        )
        if attempt_number < maximum_attempts:
            print(
                f"Output download attempt {attempt_number} failed; retrying in {retry_delay_seconds:g} seconds.",
                flush=True,
            )
            time.sleep(retry_delay_seconds)
    raise RuntimeError(failure_message)


def download_completed_experiment_runner_outputs(
    source_workflow_output_url: str,
    completed_experiment_runner_task_names_by_run_name: Mapping[str, str],
    download_root_directory: Path,
) -> dict[str, Path]:
    """Download each available completed Run output and skip outputs that cannot be collected."""
    downloaded_output_directories_by_run_name: dict[str, Path] = {}
    for run_name, task_name in completed_experiment_runner_task_names_by_run_name.items():
        destination_directory = download_root_directory / task_name
        output_url = experiment_runner_output_url(source_workflow_output_url, task_name)
        print(f"Downloading completed Run '{run_name}' from {output_url}", flush=True)
        try:
            download_experiment_runner_output(output_url, destination_directory)
        except RuntimeError as error:
            print(f"Skipping Run '{run_name}' because its output could not be downloaded:\n{error}", flush=True)
            continue

        expected_run_output_directory = destination_directory / run_name
        if not expected_run_output_directory.is_dir():
            print(
                f"Skipping Run '{run_name}' because its downloaded output does not contain "
                f"'{expected_run_output_directory}'.",
                flush=True,
            )
            continue
        downloaded_output_directories_by_run_name[run_name] = destination_directory
    return downloaded_output_directories_by_run_name


def build_experiment_output(
    experiment_runner_output_directories_by_run_name: Mapping[str, Path],
    experiment_output_directory: Path,
    build_experiment_output_script_path: Path,
    temporary_directory: Path,
) -> None:
    """Run the embedded builder over the downloaded successful Run outputs."""
    output_directories_file_path = temporary_directory / _EXPERIMENT_RUNNER_OUTPUT_DIRECTORIES_FILE_NAME
    output_directories_file_path.write_text(
        json.dumps(
            {run_name: str(path) for run_name, path in experiment_runner_output_directories_by_run_name.items()}
        ),
        encoding="utf-8",
    )
    command = [
        "/isaac-sim/python.sh",
        str(build_experiment_output_script_path),
        "--experiment-runner-output-directories-file",
        str(output_directories_file_path),
        "--experiment-output-directory",
        str(experiment_output_directory),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Experiment output builder failed with exit code {result.returncode}: {shlex.join(command)}\n"
            f"{result.stderr.strip()}"
        )


def load_coordinator_settings(settings_file_path: Path) -> dict[str, Any]:
    """Load the embedded Experiment output coordinator settings."""
    settings = json.loads(settings_file_path.read_text(encoding="utf-8"))
    assert isinstance(settings, dict), "Experiment output coordinator settings must be a JSON object"
    return settings


def coordinate_experiment_output(
    source_workflow_id: str,
    settings_file_path: Path,
    build_experiment_output_script_path: Path,
    experiment_output_directory: Path,
) -> dict[str, str]:
    """Wait for all Runs, download successful outputs, and build their combined report."""
    settings = load_coordinator_settings(settings_file_path)
    experiment_runner_task_names_by_run_name = settings["experiment_runner_task_names_by_run_name"]
    poll_interval_seconds = settings["poll_interval_seconds"]

    terminal_source_workflow = wait_for_source_workflow_to_finish(
        source_workflow_id,
        experiment_runner_task_names_by_run_name,
        poll_interval_seconds,
    )
    terminal_runner_statuses = get_experiment_runner_statuses(
        terminal_source_workflow,
        experiment_runner_task_names_by_run_name,
    )
    completed_experiment_runner_task_names_by_run_name = find_completed_experiment_runners(
        experiment_runner_task_names_by_run_name,
        terminal_runner_statuses,
    )
    print(
        f"Found {len(completed_experiment_runner_task_names_by_run_name)} of "
        f"{len(experiment_runner_task_names_by_run_name)} completed Runs.",
        flush=True,
    )

    with tempfile.TemporaryDirectory(prefix="arena_experiment_outputs_") as temporary_directory_string:
        temporary_directory = Path(temporary_directory_string)
        downloaded_output_directories_by_run_name: dict[str, Path] = {}
        if completed_experiment_runner_task_names_by_run_name:
            source_workflow_output_url = terminal_source_workflow.get("outputs")
            assert (
                isinstance(source_workflow_output_url, str) and source_workflow_output_url
            ), "OSMO source workflow has completed Run outputs but exposes no workflow output URL"
            downloaded_output_directories_by_run_name = download_completed_experiment_runner_outputs(
                source_workflow_output_url,
                completed_experiment_runner_task_names_by_run_name,
                temporary_directory / "downloads",
            )
        collected_experiment_runner_task_names_by_run_name = {
            run_name: completed_experiment_runner_task_names_by_run_name[run_name]
            for run_name in downloaded_output_directories_by_run_name
        }
        print(
            f"Building the report from {len(collected_experiment_runner_task_names_by_run_name)} available Run "
            "outputs.",
            flush=True,
        )
        build_experiment_output(
            downloaded_output_directories_by_run_name,
            experiment_output_directory,
            build_experiment_output_script_path,
            temporary_directory,
        )

    return collected_experiment_runner_task_names_by_run_name


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-workflow-id", required=True, help="OSMO workflow containing the Experiment Runs")
    parser.add_argument("--settings-file", required=True, type=Path, help="embedded coordinator settings JSON")
    parser.add_argument(
        "--build-experiment-output-script",
        required=True,
        type=Path,
        help="embedded script that builds the combined Experiment output",
    )
    parser.add_argument(
        "--experiment-output-directory",
        required=True,
        type=Path,
        help="OSMO output directory for the combined Experiment report",
    )
    return parser.parse_args()


def main() -> None:
    """Coordinate one aggregated Experiment output from all successful Runs."""
    arguments = _parse_arguments()
    coordinate_experiment_output(
        arguments.source_workflow_id,
        arguments.settings_file,
        arguments.build_experiment_output_script,
        arguments.experiment_output_directory,
    )


if __name__ == "__main__":
    main()

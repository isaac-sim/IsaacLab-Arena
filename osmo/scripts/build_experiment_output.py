# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build one Arena Experiment output from independently executed Experiment Runner tasks.

The input JSON maps each Run name to its Experiment Runner task's output directory. Each runner result selects whether
its ``<run-name>/...`` output is included. Completed Run directories are copied into the
``<experiment-output>/<run-name>`` layout, while failed results are preserved without partial Run artifacts. The
``arena_experiment_result.json`` and ``index.html`` report list every Run execution and include episode details only
from completed Runs.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Mapping
from pathlib import Path

from isaaclab_arena.evaluation.arena_experiment_result import ArenaExperimentResult
from isaaclab_arena.evaluation.arena_run import RunStatus
from isaaclab_arena.visualization.report import RunExecutionReport, build_report

EXPERIMENT_RUNNER_RESULT_FILE_NAME = "experiment_runner_result.json"


def load_experiment_runner_result(
    experiment_runner_output_directory: Path,
    run_name: str,
) -> tuple[RunExecutionReport, dict[str, object]]:
    """Load and validate an Experiment Runner result.

    Args:
        experiment_runner_output_directory: Root of one Experiment Runner task output.
        run_name: Run associated with the task output, used in validation messages.

    Returns:
        Validated Run execution result and its environment and policy metadata.
    """
    experiment_runner_result_path = experiment_runner_output_directory / EXPERIMENT_RUNNER_RESULT_FILE_NAME
    experiment_runner_result = json.loads(experiment_runner_result_path.read_text(encoding="utf-8"))
    execution_status = RunStatus(experiment_runner_result["execution_status"])
    process_exit_code = experiment_runner_result["process_exit_code"]
    assert (execution_status is RunStatus.COMPLETED) == (
        process_exit_code == 0
    ), f"Experiment Runner result for Run '{run_name}' is inconsistent: '{experiment_runner_result_path}'"
    run_metadata_by_name = experiment_runner_result["runs"]
    assert isinstance(run_metadata_by_name, dict) and set(run_metadata_by_name) == {
        run_name
    }, f"Experiment Runner result for Run '{run_name}' must contain metadata for exactly that Run"
    run_metadata = run_metadata_by_name[run_name]
    assert isinstance(run_metadata, dict), f"Experiment Runner metadata for Run '{run_name}' must be a JSON object"
    return (
        RunExecutionReport(
            run_name=run_name,
            status=execution_status,
            process_exit_code=process_exit_code,
        ),
        run_metadata,
    )


def load_experiment_runner_output_directories_by_run_name(
    experiment_runner_output_directories_file_path: Path,
) -> dict[str, Path]:
    """Load ``run-name -> Experiment Runner task output directory`` from JSON.

    Args:
        experiment_runner_output_directories_file_path: JSON file containing one Experiment Runner output directory
            per Run.

    Returns:
        Run names mapped to Experiment Runner output directories.
    """
    with experiment_runner_output_directories_file_path.open(
        encoding="utf-8"
    ) as experiment_runner_output_directories_file:
        runner_output_directory_strings_by_run_name = json.load(experiment_runner_output_directories_file)

    assert (
        isinstance(runner_output_directory_strings_by_run_name, dict) and runner_output_directory_strings_by_run_name
    ), "Experiment Runner output directories must be a non-empty JSON mapping"
    experiment_runner_output_directories_by_run_name: dict[str, Path] = {}
    for run_name, runner_output_directory_string in runner_output_directory_strings_by_run_name.items():
        assert isinstance(run_name, str) and run_name, "Run names must be non-empty strings"
        assert (
            isinstance(runner_output_directory_string, str) and runner_output_directory_string
        ), f"Experiment Runner output directory for Run '{run_name}' must be a non-empty string"
        experiment_runner_output_directories_by_run_name[run_name] = Path(runner_output_directory_string)
    return experiment_runner_output_directories_by_run_name


def collect_run_outputs_into_experiment_output(
    experiment_runner_output_directories_by_run_name: Mapping[str, Path],
    experiment_output_directory: Path,
) -> tuple[list[RunExecutionReport], dict[str, dict[str, object]]]:
    """Collect completed Run outputs and preserve failed execution results.

    Args:
        experiment_runner_output_directories_by_run_name: Run names mapped to Experiment Runner task output
            directories.
        experiment_output_directory: Destination Experiment directory containing one subdirectory per Run.

    Returns:
        Validated execution results and Run metadata for the aggregated Experiment result.
    """
    assert experiment_runner_output_directories_by_run_name, "At least one Experiment Runner output is required"
    run_execution_reports: list[RunExecutionReport] = []
    run_metadata_by_name: dict[str, dict[str, object]] = {}
    for run_name, experiment_runner_output_directory in experiment_runner_output_directories_by_run_name.items():
        ArenaExperimentResult.assert_run_name_is_safe_path_component(run_name)
        run_execution_report, run_metadata = load_experiment_runner_result(
            experiment_runner_output_directory,
            run_name,
        )
        run_execution_reports.append(run_execution_report)
        run_metadata_by_name[run_name] = {
            **run_metadata,
            "status": run_execution_report.status.value,
        }
        destination_run_output_directory = experiment_output_directory / run_name
        experiment_runner_result_path = experiment_runner_output_directory / EXPERIMENT_RUNNER_RESULT_FILE_NAME
        if run_execution_report.status is RunStatus.FAILED:
            print(
                f"[WARNING] Excluding failed Run '{run_name}' with process exit code "
                f"{run_execution_report.process_exit_code}"
            )
            destination_run_output_directory.mkdir(parents=True)
            shutil.copy2(
                experiment_runner_result_path,
                destination_run_output_directory / EXPERIMENT_RUNNER_RESULT_FILE_NAME,
            )
            continue

        source_run_output_directory = experiment_runner_output_directory / run_name
        assert (
            source_run_output_directory.is_dir()
        ), f"Completed Run '{run_name}' is missing its expected output directory: '{source_run_output_directory}'"
        shutil.copytree(
            source_run_output_directory,
            destination_run_output_directory,
        )
        shutil.copy2(
            experiment_runner_result_path,
            destination_run_output_directory / EXPERIMENT_RUNNER_RESULT_FILE_NAME,
        )
    return (
        sorted(run_execution_reports, key=lambda run_execution_report: run_execution_report.run_name),
        run_metadata_by_name,
    )


def build_experiment_output(
    experiment_runner_output_directories_by_run_name: Mapping[str, Path],
    experiment_output_directory: Path,
) -> Path:
    """Build one complete Experiment output from Experiment Runner task outputs.

    Args:
        experiment_runner_output_directories_by_run_name: Run names mapped to Experiment Runner task output
            directories.
        experiment_output_directory: Experiment output directory containing one subdirectory per Run and
            ``index.html``.

    Returns:
        Path to the generated Experiment report.
    """
    run_execution_reports, run_metadata_by_name = collect_run_outputs_into_experiment_output(
        experiment_runner_output_directories_by_run_name,
        experiment_output_directory,
    )
    ArenaExperimentResult(experiment_output_directory, run_metadata_by_name).write()
    return build_report(experiment_output_directory, run_executions=run_execution_reports)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-runner-output-directories-file",
        required=True,
        type=Path,
        help="JSON mapping of each Run name to its Experiment Runner task output directory",
    )
    parser.add_argument(
        "--experiment-output-directory",
        required=True,
        type=Path,
        help="Arena Experiment output containing one directory per Run and index.html",
    )
    return parser.parse_args()


def main() -> None:
    """Build one Experiment output from the Experiment Runner outputs described on the command line."""
    parsed_arguments = _parse_arguments()
    experiment_runner_output_directories_by_run_name = load_experiment_runner_output_directories_by_run_name(
        parsed_arguments.experiment_runner_output_directories_file
    )
    build_experiment_output(
        experiment_runner_output_directories_by_run_name,
        parsed_arguments.experiment_output_directory,
    )


if __name__ == "__main__":
    main()

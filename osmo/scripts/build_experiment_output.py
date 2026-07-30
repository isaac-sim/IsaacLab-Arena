# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build one Arena Experiment output from independently executed Experiment Runner tasks.

The input JSON maps each Run name to its Experiment Runner task's output directory. Each task output must contain
``<run-name>/...``. Those Run directories are copied into the ``<experiment-output>/<run-name>`` layout, where one
``index.html`` report is generated.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Mapping
from pathlib import Path

from isaaclab_arena.visualization.report import build_report


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
) -> None:
    """Collect each Experiment Runner task's Run directory into one Experiment output directory.

    Args:
        experiment_runner_output_directories_by_run_name: Run names mapped to Experiment Runner task output
            directories. Each task output directory must contain a child directory with the corresponding Run name.
        experiment_output_directory: Destination Experiment directory containing one subdirectory per Run.
    """
    assert experiment_runner_output_directories_by_run_name, "At least one Experiment Runner output is required"
    for run_name, experiment_runner_output_directory in experiment_runner_output_directories_by_run_name.items():
        source_run_output_directory = experiment_runner_output_directory / run_name
        assert source_run_output_directory.is_dir(), (
            f"Expected Run output directory for Run '{run_name}' does not exist or is not a directory: "
            f"'{source_run_output_directory}'"
        )
        shutil.copytree(
            source_run_output_directory,
            experiment_output_directory / run_name,
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
    collect_run_outputs_into_experiment_output(
        experiment_runner_output_directories_by_run_name,
        experiment_output_directory,
    )
    return build_report(experiment_output_directory)


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

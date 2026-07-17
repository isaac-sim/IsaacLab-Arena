# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Combine canonical Arena Run output directories into one Experiment report."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Mapping
from pathlib import Path

from isaaclab_arena.visualization.report import build_report


def aggregate_run_outputs(
    run_output_directories_by_name: Mapping[str, Path],
    combined_experiment_output_directory: Path,
) -> Path:
    """Copy canonical Run outputs into one Experiment directory and build its report.

    Args:
        run_output_directories_by_name: Run names mapped to their exact output directories.
        combined_experiment_output_directory: Destination for the combined Run directories and report.

    Returns:
        Path to the generated Experiment report.
    """
    assert run_output_directories_by_name, "At least one Run output is required"
    combined_experiment_output_directory.mkdir(parents=True, exist_ok=True)

    for run_name, source_run_output_directory in run_output_directories_by_name.items():
        assert (
            source_run_output_directory.is_dir()
        ), f"Run '{run_name}' output directory does not exist: '{source_run_output_directory}'"
        destination_run_output_directory = combined_experiment_output_directory / run_name
        assert (
            not destination_run_output_directory.exists()
        ), f"Run output destination already exists: '{destination_run_output_directory}'"
        shutil.copytree(source_run_output_directory, destination_run_output_directory)

    return build_report(combined_experiment_output_directory)


def _load_run_output_directories_by_name(serialized_run_output_directories_path: Path) -> dict[str, Path]:
    """Load a JSON mapping of Run names to exact output directories."""
    with serialized_run_output_directories_path.open(encoding="utf-8") as serialized_run_output_directories_file:
        serialized_run_output_directories_by_name = json.load(serialized_run_output_directories_file)

    assert (
        isinstance(serialized_run_output_directories_by_name, dict) and serialized_run_output_directories_by_name
    ), "Run output directories must be a non-empty JSON mapping"
    run_output_directories_by_name: dict[str, Path] = {}
    for run_name, serialized_run_output_directory in serialized_run_output_directories_by_name.items():
        assert isinstance(run_name, str) and run_name, "Run output directory names must be non-empty strings"
        assert (
            isinstance(serialized_run_output_directory, str) and serialized_run_output_directory
        ), f"Run '{run_name}' output directory must be a non-empty string"
        run_output_directories_by_name[run_name] = Path(serialized_run_output_directory)
    return run_output_directories_by_name


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine canonical Arena Run outputs into an Experiment report.")
    parser.add_argument(
        "--run-output-directories-file",
        required=True,
        type=Path,
        help="JSON mapping of Run names to their exact output directories",
    )
    parser.add_argument(
        "--combined-experiment-output-directory",
        required=True,
        type=Path,
        help="Destination directory for the combined Experiment output",
    )
    return parser.parse_args()


def main() -> None:
    """Load exact Run output directories and aggregate them."""
    parsed_arguments = _parse_args()
    run_output_directories_by_name = _load_run_output_directories_by_name(parsed_arguments.run_output_directories_file)
    aggregate_run_outputs(
        run_output_directories_by_name,
        parsed_arguments.combined_experiment_output_directory,
    )


if __name__ == "__main__":
    main()

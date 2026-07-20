# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build one Arena Experiment output from independently staged OSMO Run outputs.

The input JSON maps each Run name to its Experiment Runner task's staged output root. Each root must contain
exactly one ``<timestamped-experiment-output>/<run-name>`` directory. The Run directories are copied into the
``<combined-experiment-output>/<run-name>`` layout, where one ``index.html`` report is generated.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Mapping
from pathlib import Path

from isaaclab_arena.visualization.report import build_report


def load_staged_experiment_runner_output_directories_by_run_name(
    serialized_output_directories_path: Path,
) -> dict[str, Path]:
    """Load ``run-name -> staged Experiment Runner task output root`` from JSON.

    Args:
        serialized_output_directories_path: JSON file containing one staged task output root per Run.

    Returns:
        Run names mapped to staged Experiment Runner task output roots.
    """
    with serialized_output_directories_path.open(encoding="utf-8") as serialized_output_directories_file:
        serialized_output_directories_by_run_name = json.load(serialized_output_directories_file)

    assert (
        isinstance(serialized_output_directories_by_run_name, dict) and serialized_output_directories_by_run_name
    ), "Staged Experiment Runner output directories must be a non-empty JSON mapping"
    staged_output_directories_by_run_name: dict[str, Path] = {}
    for run_name, serialized_output_directory in serialized_output_directories_by_run_name.items():
        assert isinstance(run_name, str) and run_name, "Run names must be non-empty strings"
        assert (
            isinstance(serialized_output_directory, str) and serialized_output_directory
        ), f"Staged Experiment Runner output directory for Run '{run_name}' must be a non-empty string"
        staged_output_directories_by_run_name[run_name] = Path(serialized_output_directory)
    return staged_output_directories_by_run_name


def resolve_run_output_directories_from_staged_experiment_runner_outputs(
    staged_output_directories_by_run_name: Mapping[str, Path],
) -> dict[str, Path]:
    """Resolve ``<staged-task-output>/<timestamped-experiment-output>/<run-name>`` for every Run.

    Args:
        staged_output_directories_by_run_name: Run names mapped to staged Experiment Runner task outputs.

    Returns:
        Run names mapped to the exact Run output directories within those task outputs.
    """
    assert staged_output_directories_by_run_name, "At least one staged Experiment Runner output is required"

    run_output_directories_by_name: dict[str, Path] = {}
    for run_name, staged_output_directory in staged_output_directories_by_run_name.items():
        assert staged_output_directory.is_dir(), (
            f"Staged Experiment Runner output directory for Run '{run_name}' does not exist or is not a directory: "
            f"'{staged_output_directory}'"
        )
        matching_run_output_directories = [
            experiment_output_directory / run_name
            for experiment_output_directory in sorted(staged_output_directory.iterdir())
            if experiment_output_directory.is_dir() and (experiment_output_directory / run_name).is_dir()
        ]
        assert len(matching_run_output_directories) == 1, (
            "Expected exactly one Run output matching "
            f"'<staged-output>/<timestamped-experiment-output>/{run_name}' below "
            f"'{staged_output_directory}', found {len(matching_run_output_directories)}: "
            f"{[str(run_output_directory) for run_output_directory in matching_run_output_directories]}"
        )
        run_output_directories_by_name[run_name] = matching_run_output_directories[0]

    return run_output_directories_by_name


def build_experiment_output_from_staged_experiment_runner_outputs(
    staged_output_directories_by_run_name: Mapping[str, Path],
    combined_experiment_output_directory: Path,
) -> Path:
    """Copy every staged Run directory into one Experiment output and generate its report.

    Args:
        staged_output_directories_by_run_name: Run names mapped to staged Experiment Runner task outputs.
        combined_experiment_output_directory: Combined Experiment directory containing one subdirectory per Run and
            ``index.html``.

    Returns:
        Path to the generated Experiment report.
    """
    run_output_directories_by_name = resolve_run_output_directories_from_staged_experiment_runner_outputs(
        staged_output_directories_by_run_name
    )
    for run_name, source_run_output_directory in run_output_directories_by_name.items():
        shutil.copytree(
            source_run_output_directory,
            combined_experiment_output_directory / run_name,
        )
    return build_report(combined_experiment_output_directory)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staged-experiment-runner-output-directories-file",
        required=True,
        type=Path,
        help="JSON mapping of each Run name to its staged OSMO Experiment Runner task output root",
    )
    parser.add_argument(
        "--combined-experiment-output-directory",
        required=True,
        type=Path,
        help="Combined Arena Experiment output containing one directory per Run and index.html",
    )
    return parser.parse_args()


def main() -> None:
    """Build one Experiment output from the staged Run outputs described on the command line."""
    parsed_arguments = _parse_arguments()
    staged_output_directories_by_run_name = load_staged_experiment_runner_output_directories_by_run_name(
        parsed_arguments.staged_experiment_runner_output_directories_file
    )
    build_experiment_output_from_staged_experiment_runner_outputs(
        staged_output_directories_by_run_name,
        parsed_arguments.combined_experiment_output_directory,
    )


if __name__ == "__main__":
    main()

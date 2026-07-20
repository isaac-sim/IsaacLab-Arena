# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Combine canonical Arena Run output directories into one Experiment report."""

from __future__ import annotations

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

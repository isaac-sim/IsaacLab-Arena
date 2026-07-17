# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Define the filesystem layout shared by Arena Experiment output producers and consumers."""

from pathlib import Path


def get_experiment_run_output_directory(experiment_output_directory: Path, run_name: str) -> Path:
    """Return the canonical output directory for one named Run in an Experiment.

    Args:
        experiment_output_directory: Root directory containing the Experiment output.
        run_name: Name identifying the Run within the Experiment.

    Returns:
        The Run-specific subdirectory below the Experiment output directory.
    """
    return experiment_output_directory / run_name

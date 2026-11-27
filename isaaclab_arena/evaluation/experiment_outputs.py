# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Write the Experiment-level files that combine the per-Run timings and resource usage files."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from isaaclab_arena.evaluation.experiment_resource_usage import aggregate_experiment_resource_usage
from isaaclab_arena.evaluation.experiment_timings import aggregate_experiment_timings


def aggregate_experiment_run_outputs(experiment_output_directory: Path, run_names: Iterable[str]) -> list[Path]:
    """Combine the named Runs' timings and resource files into Experiment-level files and print their paths.

    Args:
        experiment_output_directory: Experiment output holding one output directory per Run.
        run_names: Runs to include, typically the completed ones.

    Returns:
        Paths to the written Experiment timings file and Experiment resources file, in that order.
    """
    run_names = list(run_names)
    timings_path = aggregate_experiment_timings(experiment_output_directory, run_names)
    print(f"Wrote Arena Experiment timings to: {timings_path}")
    resource_usage_path = aggregate_experiment_resource_usage(experiment_output_directory, run_names)
    print(f"Wrote Arena Experiment resources to: {resource_usage_path}")
    return [timings_path, resource_usage_path]

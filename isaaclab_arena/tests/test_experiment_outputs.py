# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit test for combining the per-Run outputs of an Experiment. No Isaac Sim or GPU required."""

from isaaclab_arena.evaluation.arena_experiment_result import (
    ARENA_AGGREGATED_RESOURCE_USAGE_FILENAME,
    ARENA_AGGREGATED_TIMINGS_FILENAME,
)
from isaaclab_arena.evaluation.experiment_outputs import aggregate_experiment_run_outputs
from isaaclab_arena.evaluation.experiment_resource_usage import write_run_resource_usage
from isaaclab_arena.evaluation.experiment_timings import write_run_timings
from isaaclab_arena.utils.compute_resource.compute_resource_usage import ResourceUsage


def test_writes_both_experiment_files_for_the_named_runs(tmp_path):
    for run_name in ["first", "second"]:
        write_run_timings(tmp_path / run_name)
        write_run_resource_usage(tmp_path / run_name, ResourceUsage())

    paths = aggregate_experiment_run_outputs(tmp_path, ["first", "second"])

    assert paths == [tmp_path / ARENA_AGGREGATED_TIMINGS_FILENAME, tmp_path / ARENA_AGGREGATED_RESOURCE_USAGE_FILENAME]
    assert all(path.is_file() for path in paths)

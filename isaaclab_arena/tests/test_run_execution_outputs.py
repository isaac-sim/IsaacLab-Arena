# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests of the files a completed Run writes. No Isaac Sim or GPU required."""

from isaaclab_arena.evaluation.arena_experiment_result import (
    ARENA_SINGLE_RUN_RESOURCE_USAGE_FILENAME,
    ARENA_SINGLE_RUN_TIMINGS_FILENAME,
)
from isaaclab_arena.evaluation.arena_run import ArenaRunResult
from isaaclab_arena.evaluation.run_execution import _write_run_outputs
from isaaclab_arena.evaluation.run_status import RunStatus
from isaaclab_arena.utils.resource_usage import ResourceUsage


def test_write_run_outputs_writes_timings_and_resources(tmp_path):
    usage = ResourceUsage()
    usage.vram.update(123.0)
    result = ArenaRunResult(run_name="run", status=RunStatus.COMPLETED, start_timestamp_ns=0, end_timestamp_ns=1)

    _write_run_outputs(tmp_path / "run", result, usage)

    assert (tmp_path / "run" / ARENA_SINGLE_RUN_TIMINGS_FILENAME).is_file()
    assert (tmp_path / "run" / ARENA_SINGLE_RUN_RESOURCE_USAGE_FILENAME).is_file()

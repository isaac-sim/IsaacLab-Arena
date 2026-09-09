# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Write Arena Experiment timings as one file per Run plus one combined file for the Experiment.

Timers accumulate in a process-wide registry, so the Experiment Runner clears it between Runs and
writes each Run's timings into that Run's output directory. OSMO gets the same per-Run split for
free by running each Run in its own process. Both then combine those files with the same function,
so an Experiment produces the same timings layout whichever way it was run.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from isaaclab_arena.evaluation.arena_experiment_result import ARENA_EXPERIMENT_TIMINGS_FILENAME
from isaaclab_arena.utils.timer import merge_timer_stats_json, write_timer_stats_json

EXPERIMENT_RUNNER_APP_NAME = "experiment_runner"
"""Identifier recorded in every timing entry the Experiment Runner writes."""


def write_run_timings(run_output_directory: Path) -> Path:
    """Write the timers recorded so far into one Run's output directory.

    Args:
        run_output_directory: The Run's output directory, created if it does not already exist.

    Returns:
        The path that was written.
    """
    run_output_directory.mkdir(parents=True, exist_ok=True)
    return write_timer_stats_json(
        run_output_directory / ARENA_EXPERIMENT_TIMINGS_FILENAME,
        app_name=EXPERIMENT_RUNNER_APP_NAME,
    )


def aggregate_experiment_timings(experiment_output_directory: Path, run_names: Iterable[str]) -> Path:
    """Combine the named Runs' timings files into one Experiment timings file.

    Args:
        experiment_output_directory: Experiment output holding one output directory per Run.
        run_names: Runs to include, normally the completed ones. Included in sorted order so that
            the combined file does not depend on the order the Runs were executed or collected in.

    Returns:
        Path to the written Experiment timings file.
    """
    run_records: list[dict[str, object]] = []
    for run_name in sorted(run_names):
        run_timings_path = experiment_output_directory / run_name / ARENA_EXPERIMENT_TIMINGS_FILENAME
        assert run_timings_path.is_file(), f"Run '{run_name}' is missing its timings file: '{run_timings_path}'"
        run_records.extend(
            {"run_name": run_name, **record} for record in json.loads(run_timings_path.read_text(encoding="utf-8"))
        )

    experiment_timings_path = experiment_output_directory / ARENA_EXPERIMENT_TIMINGS_FILENAME
    experiment_timings_path.write_text(
        json.dumps({"totals": merge_timer_stats_json(run_records), "runs": run_records}, allow_nan=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return experiment_timings_path

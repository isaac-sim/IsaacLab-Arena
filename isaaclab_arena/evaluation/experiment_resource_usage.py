# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Write Arena Experiment resource usage as one file per Run plus one combined file for the Experiment."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

from isaaclab_arena.evaluation.arena_experiment_result import (
    ARENA_AGGREGATED_RESOURCE_USAGE_FILENAME,
    ARENA_SINGLE_RUN_RESOURCE_USAGE_FILENAME,
)
from isaaclab_arena.evaluation.experiment_timings import EXPERIMENT_RUNNER_APP_NAME
from isaaclab_arena.utils.resource_usage import ResourceUsage, merge_resource_records


def write_run_resource_usage(run_output_directory: Path, usage: ResourceUsage) -> Path:
    """Write a Run's resource usage into its output directory, returning the path written.

    Args:
        run_output_directory: The Run's output directory, created if needed.
        usage: Resource usage measured during the Run.
    """
    run_output_directory.mkdir(parents=True, exist_ok=True)
    text = json.dumps(usage.to_json_records(app_name=EXPERIMENT_RUNNER_APP_NAME), allow_nan=False, indent=2)
    run_resource_usage_path = run_output_directory / ARENA_SINGLE_RUN_RESOURCE_USAGE_FILENAME
    run_resource_usage_path.write_text(_put_samples_on_one_line(text) + "\n", encoding="utf-8")
    return run_resource_usage_path


def aggregate_experiment_resource_usage(experiment_output_directory: Path, run_names: Iterable[str]) -> Path:
    """Combine the named Runs' resource files into one Experiment resources file.

    The combined file holds "totals", one entry per resource summed or maximized over every Run, and "runs",
    every Run's own entries with the Run's name added to each as "run_name".

    Args:
        experiment_output_directory: Experiment output holding one output directory per Run.
        run_names: Runs to include, sorted so the file does not depend on the order given.

    Returns:
        Path to the written Experiment resources file.
    """
    run_records: list[dict[str, object]] = []
    for run_name in sorted(run_names):
        run_resource_usage_path = experiment_output_directory / run_name / ARENA_SINGLE_RUN_RESOURCE_USAGE_FILENAME
        assert (
            run_resource_usage_path.is_file()
        ), f"Run '{run_name}' is missing its resources file: '{run_resource_usage_path}'"
        run_records.extend(
            {"run_name": run_name, **record}
            for record in json.loads(run_resource_usage_path.read_text(encoding="utf-8"))
        )

    # The samples only feed the totals' percentiles; each Run's own record keeps its percentiles.
    totals = merge_resource_records(run_records, remove_samples_from_input_records=True)

    experiment_resource_usage_path = experiment_output_directory / ARENA_AGGREGATED_RESOURCE_USAGE_FILENAME
    experiment_resource_usage_path.write_text(
        json.dumps({"totals": totals, "runs": run_records}, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return experiment_resource_usage_path


def _put_samples_on_one_line(indented_json: str) -> str:
    """Collapse each indented "samples_<unit>" list in JSON text onto a single line."""
    return re.sub(
        r'("samples_\w+": )\[([^\]]*)\]',
        lambda match: f"{match.group(1)}[{', '.join(part.strip() for part in match.group(2).split(','))}]",
        indented_json,
    )

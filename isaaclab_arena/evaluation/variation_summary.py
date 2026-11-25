# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Summarize recorded episode outcomes by categorical variation value."""

from __future__ import annotations

import json
from collections import defaultdict
from prettytable import PrettyTable
from typing import TYPE_CHECKING

from isaaclab_arena.evaluation.report_data import success_rate_from_values
from isaaclab_arena.evaluation.run_status import RunStatus

if TYPE_CHECKING:
    from isaaclab_arena.evaluation.arena_experiment_result import ArenaExperimentResult


def build_categorical_variation_table(result: ArenaExperimentResult) -> PrettyTable:
    """Count episodes and successes for each observed string or boolean variation value.

    Args:
        result: Collected Experiment episodes, grouped by Run and rebuild.

    Returns:
        One row per completed Run, variation, and observed value, aggregating all rebuilds. Failed
        Runs and values no episode recorded have no row. Success rate excludes unknown outcomes and
        is N/A when no boolean outcomes are recorded. Each variation is grouped independently;
        numeric and vector values are omitted.
    """
    success_values_by_group: dict[tuple[str, str, str], list[bool | None]] = defaultdict(list)
    for run_name, run in result.runs.items():
        # A failed Run can keep the episodes it finished before failing; leave them out.
        if run["status"] == RunStatus.COMPLETED.value:
            for rebuild in run["rebuilds"]:
                for episode in rebuild["episodes"]:
                    for variation, value in episode.get("variations", {}).items():
                        if isinstance(value, (str, bool)):
                            # JSON labels distinguish a boolean from a string with the same spelling.
                            label = json.dumps(value, ensure_ascii=False)
                            success_values_by_group[(run_name, variation, label)].append(episode.get("success"))

    table = PrettyTable(["Run", "Variation", "Value", "Episodes", "Successes", "Unknown", "Success Rate"])
    for (run_name, variation, label), success_values in sorted(success_values_by_group.items()):
        group_success_rate = success_rate_from_values(success_values)
        table.add_row([
            run_name,
            variation,
            label,
            len(success_values),
            sum(success is True for success in success_values),
            sum(success is None for success in success_values),
            "N/A" if group_success_rate is None else f"{group_success_rate:.1%}",
        ])
    return table

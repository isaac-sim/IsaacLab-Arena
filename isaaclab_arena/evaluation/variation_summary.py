# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Summarize recorded episode outcomes by categorical variation value."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from prettytable import PrettyTable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_arena.evaluation.arena_experiment_result import ArenaExperimentResultData


@dataclass
class _EpisodeCounts:
    num_episodes: int = 0
    num_successes: int = 0
    num_unknown: int = 0


def build_categorical_variation_table(result: ArenaExperimentResultData) -> PrettyTable:
    """Count episodes and successes for each observed string or boolean variation value.

    Args:
        result: Collected Experiment episodes, grouped by Run and rebuild.

    Returns:
        One row per completed Run, variation, and observed value, aggregating all rebuilds. Failed
        Runs and values no episode recorded have no row. Success rate excludes unknown outcomes and
        is N/A when no boolean outcomes are recorded. Each variation is grouped independently;
        numeric and vector values are omitted.
    """
    counts: dict[tuple[str, str, str], _EpisodeCounts] = defaultdict(_EpisodeCounts)
    for run_name, run in result["runs"].items():
        # A failed Run can keep the episodes it finished before failing; leave them out.
        if run["status"] == "completed":
            for rebuild in run["rebuilds"]:
                for episode in rebuild["episodes"]:
                    for variation, value in episode.get("variations", {}).items():
                        if isinstance(value, (str, bool)):
                            # JSON labels distinguish a boolean from a string with the same spelling.
                            label = json.dumps(value, ensure_ascii=False)
                            group = counts[(run_name, variation, label)]
                            group.num_episodes += 1
                            success = episode.get("success")
                            if success is True:
                                group.num_successes += 1
                            elif success is not False:
                                group.num_unknown += 1

    table = PrettyTable(["Run", "Variation", "Value", "Episodes", "Successes", "Unknown", "Success Rate"])
    for (run_name, variation, label), group in sorted(counts.items()):
        num_known = group.num_episodes - group.num_unknown
        success_rate = f"{group.num_successes / num_known:.1%}" if num_known else "N/A"
        table.add_row([
            run_name,
            variation,
            label,
            group.num_episodes,
            group.num_successes,
            group.num_unknown,
            success_rate,
        ])
    return table

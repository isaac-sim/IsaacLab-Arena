# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Represent declarative and executable Arena Experiments."""

from dataclasses import dataclass
from typing import TypeAlias

from isaaclab_arena.evaluation.arena_run import ArenaRunCfg


@dataclass
class ArenaExperimentDefinitionCfg:
    """Declare the named Runs that form one Arena Experiment."""

    runs: dict[str, ArenaRunCfg]
    """Runs keyed by the names used for overrides, execution, and results."""

    def __post_init__(self) -> None:
        assert isinstance(self.runs, dict) and self.runs, "Experiment Definition must contain at least one Run"
        for run_name, run_cfg in self.runs.items():
            assert isinstance(run_name, str) and run_name, "Experiment Definition Run names must be non-empty strings"
            assert isinstance(run_cfg, ArenaRunCfg), f"Experiment Definition Run '{run_name}' must be an ArenaRunCfg"
            assert run_cfg.name == run_name, (
                "Run name is defined by its Experiment Definition mapping key and cannot be overridden: "
                f"expected '{run_name}', got '{run_cfg.name}'"
            )


ArenaExperiment: TypeAlias = list[ArenaRunCfg]
"""An Experiment expressed as its ordered list of Runs."""

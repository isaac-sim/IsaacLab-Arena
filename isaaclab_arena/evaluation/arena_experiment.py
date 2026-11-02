# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Represent Arena Experiment configurations."""

from dataclasses import dataclass

from isaaclab_arena.evaluation.arena_run import ArenaRunCfg
from isaaclab_arena.evaluation.single_line_string import is_single_line_nonempty_string


@dataclass
class ArenaExperimentCfg:
    """Configure the named Runs that form one Arena Experiment."""

    runs: dict[str, ArenaRunCfg]
    """Runs keyed by the names used for overrides, execution, and results."""

    name: str | None = None
    """Name of the Experiment, used as the `benchmark_id` of its KPI records.

    A command-line override must quote a name that looks like a boolean or a number, for example
    ``name='true'`` or ``name='1e3'``; Hydra otherwise converts the value and the name becomes ``True`` or ``1000.0``.
    """

    def __post_init__(self) -> None:
        assert self.name is None or is_single_line_nonempty_string(
            self.name
        ), f"Experiment name must be a non-empty single-line string, got {self.name!r}"
        assert isinstance(self.runs, dict) and self.runs, "Experiment Definition must contain at least one Run"
        for run_name, run_cfg in self.runs.items():
            assert isinstance(run_name, str) and run_name, "Experiment Definition Run names must be non-empty strings"
            assert isinstance(run_cfg, ArenaRunCfg), f"Experiment Definition Run '{run_name}' must be an ArenaRunCfg"
            assert run_cfg.name == run_name, (
                "Run name is defined by its Experiment Definition mapping key and cannot be overridden: "
                f"expected '{run_name}', got '{run_cfg.name}'"
            )

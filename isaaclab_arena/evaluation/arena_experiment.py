# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Represent Arena Experiment configurations."""

from dataclasses import dataclass, field

from isaaclab_arena.evaluation.arena_run import ArenaRunCfg
from isaaclab_arena.evaluation.single_line_string import is_single_line_nonempty_string
from isaaclab_arena_benchmarking.benchmarks.registry import assert_valid_kpi_exporter_names


@dataclass
class ArenaExperimentCfg:
    """Configure the named Runs that form one Arena Experiment."""

    runs: dict[str, ArenaRunCfg]
    """Runs keyed by the names used for overrides, execution, and results."""

    name: str
    """Name of the Experiment, a non-empty single-line string."""

    kpi_exporters: list[str] = field(default_factory=lambda: ["timing"])
    """Names of the registered KPI exporters that write the Experiment's KPI records once every Run completed.
    Each name appears once; an empty list exports no KPIs."""

    def __post_init__(self) -> None:
        assert is_single_line_nonempty_string(
            self.name
        ), f"Experiment name must be a non-empty single-line string, got {self.name!r}"
        assert isinstance(self.kpi_exporters, list) and all(
            isinstance(name, str) for name in self.kpi_exporters
        ), f"Experiment kpi_exporters must be a list of names, got {self.kpi_exporters!r}"
        assert_valid_kpi_exporter_names(self.kpi_exporters)
        assert isinstance(self.runs, dict) and self.runs, "Experiment Definition must contain at least one Run"
        for run_name, run_cfg in self.runs.items():
            assert isinstance(run_name, str) and run_name, "Experiment Definition Run names must be non-empty strings"
            assert isinstance(run_cfg, ArenaRunCfg), f"Experiment Definition Run '{run_name}' must be an ArenaRunCfg"
            assert run_cfg.name == run_name, (
                "Run name is defined by its Experiment Definition mapping key and cannot be overridden: "
                f"expected '{run_name}', got '{run_cfg.name}'"
            )

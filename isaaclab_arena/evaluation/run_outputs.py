# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Write the files of an Arena Run's output directory."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab_arena.evaluation.arena_experiment_result import ARENA_SINGLE_RUN_ENVIRONMENT_DEFINITION_FILENAME
from isaaclab_arena.evaluation.experiment_timings import write_run_timings
from isaaclab_arena.evaluation.legacy_graph_environment_cli import LegacyGraphEnvironmentCfg, load_graph_spec_from_graph

if TYPE_CHECKING:
    from isaaclab_arena.evaluation.arena_run import ArenaRunCfg, ArenaRunResult


def write_run_outputs(run_cfg: ArenaRunCfg, run_output_dir: Path, result: ArenaRunResult) -> None:
    """Write a completed Run's environment definition and timings to its output directory."""
    write_run_environment_definition(run_cfg, run_output_dir)
    write_run_timings(run_output_dir)


def write_run_environment_definition(run_cfg: ArenaRunCfg, run_output_dir: Path) -> Path | None:
    """Write the graph spec a Run builds its environment from, with per-Run overrides applied, to its output directory."""
    # Python environments are built from a registered factory and typed config, so they have no graph spec to write.
    # Their config is already recorded in the Experiment definition.
    if not isinstance(run_cfg.environment, LegacyGraphEnvironmentCfg):
        return None
    run_output_dir.mkdir(parents=True, exist_ok=True)
    environment_definition_path = run_output_dir / ARENA_SINGLE_RUN_ENVIRONMENT_DEFINITION_FILENAME
    load_graph_spec_from_graph(run_cfg.environment).write_yaml(environment_definition_path)
    return environment_definition_path

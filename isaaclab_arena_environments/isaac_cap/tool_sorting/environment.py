# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Registered graph entry points for CAP easy tool sorting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


# TODO(qianl): Move these settings into the graph YAML once it supports full ObjectPlacerParams.
def configure_tool_sort_placement(arena_env: IsaacLabArenaEnvironment) -> IsaacLabArenaEnvironment:
    """Apply the source layout-sampling policy for easy tool sorting."""
    arena_env.placer_params.random_yaw_init = True
    arena_env.placer_params.allow_best_loss_fallbacks = False
    return arena_env


@dataclass
class ToolSortEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure one easy tool-sorting environment."""

    use_tiled_cameras: bool = False


@dataclass
class ToolSortingEasy1EnvironmentCfg(ToolSortEnvironmentCfg):
    """Configure the first easy tool-sorting level."""


@dataclass
class ToolSortingEasy2EnvironmentCfg(ToolSortEnvironmentCfg):
    """Configure the second easy tool-sorting level."""


@dataclass
class ToolSortingEasy3EnvironmentCfg(ToolSortEnvironmentCfg):
    """Configure the third easy tool-sorting level."""


class ToolSortEnvironment(ArenaEnvironmentFactory[ToolSortEnvironmentCfg]):
    """Build one easy level from its graph YAML and tool-sort placement defaults."""

    scene_spec: Path
    _legacy_argparse_cfg_type = ToolSortEnvironmentCfg

    def build(self, cfg: ToolSortEnvironmentCfg) -> IsaacLabArenaEnvironment:
        from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec

        arena_env = ArenaEnvGraphSpec.from_yaml(str(self.scene_spec)).to_arena_env(enable_cameras=cfg.enable_cameras)
        arena_env.embodiment.set_use_tiled_cameras(cfg.use_tiled_cameras)
        return configure_tool_sort_placement(arena_env)


class ToolSortingEasy1Environment(ToolSortEnvironment):
    """Build the first easy tool-sorting level."""

    name = "vabar_tool_sorting_easy_1"
    _legacy_argparse_cfg_type = ToolSortingEasy1EnvironmentCfg
    scene_spec = Path(__file__).with_name("tool_sorting_easy_1.yaml")


class ToolSortingEasy2Environment(ToolSortEnvironment):
    """Build the second easy tool-sorting level."""

    name = "vabar_tool_sorting_easy_2"
    _legacy_argparse_cfg_type = ToolSortingEasy2EnvironmentCfg
    scene_spec = Path(__file__).with_name("tool_sorting_easy_2.yaml")


class ToolSortingEasy3Environment(ToolSortEnvironment):
    """Build the third easy tool-sorting level."""

    name = "vabar_tool_sorting_easy_3"
    _legacy_argparse_cfg_type = ToolSortingEasy3EnvironmentCfg
    scene_spec = Path(__file__).with_name("tool_sorting_easy_3.yaml")

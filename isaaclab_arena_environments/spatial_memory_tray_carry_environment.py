# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentFactory
from isaaclab_arena_environments.spatial_memory.environment import SpatialMemoryEnvironmentCfg

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@dataclass
class SpatialMemoryTrayCarryEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure task 3 of the spatial-memory suite; ``difficulty`` is the box height in metres."""


@register_environment
class SpatialMemoryTrayCarryEnvironment(ArenaEnvironmentFactory[SpatialMemoryTrayCarryEnvironmentCfg]):
    """Spatial-memory task 3: carry a box past a chair and a counter corner (G1 humanoid, head camera)."""

    name = "spatial_memory_tray_carry"
    _legacy_argparse_cfg_type = SpatialMemoryTrayCarryEnvironmentCfg

    def build(self, cfg: SpatialMemoryTrayCarryEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        from isaaclab_arena_environments.spatial_memory.environment import build_spatial_memory_environment
        from isaaclab_arena_environments.spatial_memory.g1_adapter import G1Adapter
        from isaaclab_arena_environments.spatial_memory.tasks.tray_carry import TrayCarryLogic

        logic = TrayCarryLogic(cfg.difficulty, G1Adapter())
        return build_spatial_memory_environment(self.name, logic, cfg)

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
class SpatialMemoryRestockSlotEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure task 9 of the spatial-memory suite; ``difficulty`` is the number of empty bins (1 to 3)."""


@register_environment
class SpatialMemoryRestockSlotEnvironment(ArenaEnvironmentFactory[SpatialMemoryRestockSlotEnvironmentCfg]):
    """Spatial-memory task 9: restock a new item into an empty bin (Franka, wrist camera)."""

    name = "spatial_memory_restock_slot"
    _legacy_argparse_cfg_type = SpatialMemoryRestockSlotEnvironmentCfg

    def build(self, cfg: SpatialMemoryRestockSlotEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        from isaaclab_arena_environments.spatial_memory.embodiment_adapter import FrankaAdapter
        from isaaclab_arena_environments.spatial_memory.environment import build_spatial_memory_environment
        from isaaclab_arena_environments.spatial_memory.tasks.restock_slot import (
            LOOK_JOINT_POSE,
            REST_JOINT_POSE,
            RestockSlotLogic,
        )

        logic = RestockSlotLogic(
            cfg.difficulty, FrankaAdapter(initial_joint_pose=LOOK_JOINT_POSE, rest_joint_pose=REST_JOINT_POSE)
        )
        return build_spatial_memory_environment(self.name, logic, cfg)

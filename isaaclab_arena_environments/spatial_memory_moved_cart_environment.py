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
class SpatialMemoryMovedCartEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure task 10 of the spatial-memory suite; ``difficulty`` is the re-observation window in seconds."""


@register_environment
class SpatialMemoryMovedCartEnvironment(ArenaEnvironmentFactory[SpatialMemoryMovedCartEnvironmentCfg]):
    """Spatial-memory task 10: put the mug away after a cart was moved out of view (Franka, wrist camera)."""

    name = "spatial_memory_moved_cart"
    _legacy_argparse_cfg_type = SpatialMemoryMovedCartEnvironmentCfg

    def build(self, cfg: SpatialMemoryMovedCartEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        from isaaclab_arena_environments.spatial_memory.embodiment_adapter import FrankaAdapter
        from isaaclab_arena_environments.spatial_memory.environment import build_spatial_memory_environment
        from isaaclab_arena_environments.spatial_memory.tasks.moved_cart import REST_JOINT_POSE, MovedCartLogic

        logic = MovedCartLogic(cfg.difficulty, FrankaAdapter(rest_joint_pose=REST_JOINT_POSE))
        return build_spatial_memory_environment(self.name, logic, cfg)

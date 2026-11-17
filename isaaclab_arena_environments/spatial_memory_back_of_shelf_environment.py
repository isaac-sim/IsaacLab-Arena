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
class SpatialMemoryBackOfShelfEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure task 5 of the spatial-memory suite; ``difficulty`` is the gap width in metres."""


@register_environment
class SpatialMemoryBackOfShelfEnvironment(ArenaEnvironmentFactory[SpatialMemoryBackOfShelfEnvironmentCfg]):
    """Spatial-memory task 5: take the item from the back of the shelf (Franka, wrist camera)."""

    name = "spatial_memory_back_of_shelf"
    _legacy_argparse_cfg_type = SpatialMemoryBackOfShelfEnvironmentCfg

    def build(self, cfg: SpatialMemoryBackOfShelfEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        from isaaclab_arena_environments.spatial_memory.embodiment_adapter import FrankaAdapter
        from isaaclab_arena_environments.spatial_memory.environment import build_spatial_memory_environment
        from isaaclab_arena_environments.spatial_memory.tasks.back_of_shelf import REST_JOINT_POSE, BackOfShelfLogic

        logic = BackOfShelfLogic(cfg.difficulty, FrankaAdapter(rest_joint_pose=REST_JOINT_POSE))
        return build_spatial_memory_environment(self.name, logic, cfg)

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
class SpatialMemoryKittingEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure task 8 of the spatial-memory suite; ``difficulty`` is the number of bins (2 to 6)."""


@register_environment
class SpatialMemoryKittingEnvironment(ArenaEnvironmentFactory[SpatialMemoryKittingEnvironmentCfg]):
    """Spatial-memory task 8: kit requested parts from a bin rack into a tray (Franka, wrist camera)."""

    name = "spatial_memory_kitting"
    _legacy_argparse_cfg_type = SpatialMemoryKittingEnvironmentCfg

    def build(self, cfg: SpatialMemoryKittingEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        from isaaclab_arena_environments.spatial_memory.embodiment_adapter import FrankaAdapter
        from isaaclab_arena_environments.spatial_memory.environment import build_spatial_memory_environment
        from isaaclab_arena_environments.spatial_memory.tasks.kitting import (
            LOOK_JOINT_POSE,
            REST_JOINT_POSE,
            KittingLogic,
        )

        logic = KittingLogic(
            cfg.difficulty, FrankaAdapter(initial_joint_pose=LOOK_JOINT_POSE, rest_joint_pose=REST_JOINT_POSE)
        )
        return build_spatial_memory_environment(self.name, logic, cfg)

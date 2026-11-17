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
class SpatialMemoryFetchBehindEnvironmentCfg(SpatialMemoryEnvironmentCfg):
    """Configure task 2 of the spatial-memory suite; ``difficulty`` is the number of mugs (2 to 8)."""


@register_environment
class SpatialMemoryFetchBehindEnvironment(ArenaEnvironmentFactory[SpatialMemoryFetchBehindEnvironmentCfg]):
    """Spatial-memory task 2: fetch the instructed mug from the shelf behind you (Franka, wrist camera)."""

    name = "spatial_memory_fetch_behind"
    _legacy_argparse_cfg_type = SpatialMemoryFetchBehindEnvironmentCfg

    def build(self, cfg: SpatialMemoryFetchBehindEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        from isaaclab_arena_environments.spatial_memory.embodiment_adapter import FrankaAdapter
        from isaaclab_arena_environments.spatial_memory.environment import build_spatial_memory_environment
        from isaaclab_arena_environments.spatial_memory.tasks.fetch_behind import (
            LOOK_JOINT_POSE,
            REST_JOINT_POSE,
            FetchBehindLogic,
        )

        logic = FetchBehindLogic(
            cfg.difficulty, FrankaAdapter(initial_joint_pose=LOOK_JOINT_POSE, rest_joint_pose=REST_JOINT_POSE)
        )
        return build_spatial_memory_environment(self.name, logic, cfg)

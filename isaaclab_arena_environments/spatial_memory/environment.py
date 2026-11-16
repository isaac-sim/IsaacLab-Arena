# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared configuration and builder for the registered spatial-memory environments."""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena_environments.spatial_memory.task_logic import SpatialMemoryTaskLogic


@dataclass
class SpatialMemoryEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure a spatial-memory environment."""

    observation_variant: Literal["current", "history_k", "oracle_map", "mapped"] = "current"
    """One of ``current``, ``history_k``, ``oracle_map`` or ``mapped``."""

    history_k: int | None = None
    """Frames stacked by ``history_k``; None covers 3 seconds of control steps."""

    difficulty: float | None = None
    """Value of the task's difficulty knob; None uses the task default."""


def build_spatial_memory_environment(
    name: str, logic: SpatialMemoryTaskLogic, cfg: SpatialMemoryEnvironmentCfg
) -> IsaacLabArenaEnvironment:
    """Assemble the Arena environment for ``logic``: primitive parts, ground, light, embodiment and task."""
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena_environments.spatial_memory.layout_utils import FLOOR_Z_M
    from isaaclab_arena_environments.spatial_memory.metrics import record_spatial_memory_episode
    from isaaclab_arena_environments.spatial_memory.observations import ObservationVariant, apply_observation_variant
    from isaaclab_arena_environments.spatial_memory.parts import make_part_object
    from isaaclab_arena_environments.spatial_memory.task import SpatialMemoryTask

    variant = ObservationVariant(cfg.observation_variant)
    asset_registry = AssetRegistry()
    # Spawn every part at the layout of a fixed sample so that the first frames are physically sensible; the reset
    # event writes the real per-episode layout.
    nominal = logic.sample_layout(torch.Generator().manual_seed(0)).part_poses_e
    parts = [
        make_part_object(spec, tuple(nominal[index, :3].tolist())) for index, spec in enumerate(logic.part_specs())
    ]
    ground = asset_registry.get_asset_by_name("ground_plane")(initial_pose=Pose(position_xyz=(0.0, 0.0, FLOOR_Z_M)))
    light = asset_registry.get_asset_by_name("light")()
    embodiment = logic.embodiment.make_embodiment(enable_cameras=cfg.enable_cameras)
    embodiment.set_initial_pose(Pose.identity())

    def env_cfg_callback(env_cfg):
        return apply_observation_variant(env_cfg, logic, variant, cfg.history_k)

    return IsaacLabArenaEnvironment(
        name=name,
        embodiment=embodiment,
        scene=Scene(assets=[ground, light, *parts]),
        task=SpatialMemoryTask(logic, variant),
        env_cfg_callback=env_cfg_callback,
        episode_recorder_terms={"spatial_memory": EpisodeRecorderTermCfg(func=record_spatial_memory_episode)},
    )

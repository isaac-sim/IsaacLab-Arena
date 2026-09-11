# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registered Newton environments for the gear-insertion tasks."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import IsaacLabArenaManagerBasedRLEnvCfg


_MEDIUM_SCENE_SPEC = Path(__file__).with_name("gear_medium.yaml")
_EASY_SCENE_SPEC = Path(__file__).with_name("gear_easy.yaml")


def gear_insertion_physics_cfg():
    """Build the tuned Newton configuration for the Factory gear geometry."""
    from isaaclab_newton.physics import NewtonCollisionPipelineCfg

    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import ArenaPhysicsCfg

    physics = deepcopy(ArenaPhysicsCfg().newton)
    physics.num_substeps = 4
    physics.collision_decimation = 1
    physics.solver_cfg.njmax = 8192
    physics.solver_cfg.nconmax = 4096
    physics.solver_cfg.use_mujoco_contacts = False
    physics.solver_cfg.update_data_interval = 1
    physics.default_shape_cfg.ke = 60_000.0
    physics.default_shape_cfg.kd = 500.0
    physics.collision_cfg = NewtonCollisionPipelineCfg(
        reduce_contacts=True,
        rigid_contact_max=4096,
        max_triangle_pairs=1_000_000,
    )
    return physics


def _configure_gear_insertion_physics(
    env_cfg: IsaacLabArenaManagerBasedRLEnvCfg,
    *,
    replicate_physics: bool = False,
) -> IsaacLabArenaManagerBasedRLEnvCfg:
    """Apply task-owned Newton tuning without a runner-level preset."""
    env_cfg.sim.physics = gear_insertion_physics_cfg()
    env_cfg.scene.replicate_physics = replicate_physics
    return env_cfg


@dataclass
class GearInsertionNewtonEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the registered gear-insertion environment."""

    enable_cameras: bool = False
    use_tiled_cameras: bool = False
    replicate_physics: bool = False
    episode_length_s: float | None = None


@dataclass
class GearInsertionEasyNewtonEnvironmentCfg(GearInsertionNewtonEnvironmentCfg):
    """Configure the registered easy gear-insertion environment."""


class GearInsertionNewtonEnvironment(ArenaEnvironmentFactory[GearInsertionNewtonEnvironmentCfg]):
    """Build gear insertion from its graph and task-owned Newton profile."""

    name = "vabar_contact_rich_insertion__gear_medium"
    _legacy_argparse_cfg_type = GearInsertionNewtonEnvironmentCfg
    scene_spec = _MEDIUM_SCENE_SPEC

    def build(self, cfg: GearInsertionNewtonEnvironmentCfg) -> IsaacLabArenaEnvironment:
        from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec

        spec = ArenaEnvGraphSpec.from_yaml(str(self.scene_spec))
        arena_env = spec.to_arena_env(enable_cameras=cfg.enable_cameras)
        arena_env.embodiment.set_use_tiled_cameras(cfg.use_tiled_cameras)
        if cfg.episode_length_s is not None:
            if cfg.episode_length_s <= 0:
                raise ValueError("episode_length_s must be positive")
            arena_env.task.episode_length_s = cfg.episode_length_s
        arena_env.env_cfg_callback = partial(
            _configure_gear_insertion_physics,
            replicate_physics=cfg.replicate_physics,
        )
        return arena_env


class GearInsertionEasyNewtonEnvironment(GearInsertionNewtonEnvironment):
    """Build the two-gear easy variant with the shared Newton profile."""

    name = "vabar_contact_rich_insertion__gear_easy"
    _legacy_argparse_cfg_type = GearInsertionEasyNewtonEnvironmentCfg
    scene_spec = _EASY_SCENE_SPEC

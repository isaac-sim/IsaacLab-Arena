# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Newton configuration for the DisplayPort-insertion environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import IsaacLabArenaManagerBasedRLEnvCfg


def displayport_insertion_newton_env_cfg_callback(
    env_cfg: IsaacLabArenaManagerBasedRLEnvCfg,
) -> IsaacLabArenaManagerBasedRLEnvCfg:
    """Configure Newton for the contact-rich DisplayPort insertion loop."""
    from isaaclab_newton.physics import NewtonCollisionPipelineCfg
    from isaaclab_newton.sim.schemas import NewtonMaterialPropertiesCfg

    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import ArenaPhysicsCfg

    env_cfg.sim.dt = 1.0 / 200.0
    env_cfg.decimation = 7
    env_cfg.sim.render_interval = 7
    env_cfg.sim.physics_material = NewtonMaterialPropertiesCfg(
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
    )
    env_cfg.sim.physics = ArenaPhysicsCfg().newton
    env_cfg.sim.physics.num_substeps = 4
    env_cfg.sim.physics.solver_cfg.ls_iterations = 50
    env_cfg.sim.physics.solver_cfg.ccd_iterations = 35
    env_cfg.sim.physics.solver_cfg.njmax = 4096
    env_cfg.sim.physics.solver_cfg.nconmax = 4096
    env_cfg.sim.physics.collision_cfg = NewtonCollisionPipelineCfg(
        reduce_contacts=True,
        max_triangle_pairs=2**25,
    )
    env_cfg.scene.env_spacing = 2.5
    env_cfg.scene.replicate_physics = True
    return env_cfg

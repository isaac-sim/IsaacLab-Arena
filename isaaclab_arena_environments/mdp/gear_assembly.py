# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Newton configuration for the gear-assembly environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import IsaacLabArenaManagerBasedRLEnvCfg


def gear_assembly_newton_env_cfg_callback(
    env_cfg: IsaacLabArenaManagerBasedRLEnvCfg,
) -> IsaacLabArenaManagerBasedRLEnvCfg:
    """Configure Newton for a stable 30 Hz contact-rich manipulation loop."""
    from isaaclab_newton.physics import HydroelasticSDFCfg, NewtonCollisionPipelineCfg

    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import ArenaPhysicsCfg

    env_cfg.sim.dt = 1.0 / 240.0
    env_cfg.decimation = 8
    env_cfg.sim.render_interval = 8
    env_cfg.sim.physics = ArenaPhysicsCfg().newton
    env_cfg.sim.physics.num_substeps = 4
    env_cfg.sim.physics.default_shape_cfg.gap = 0.0
    env_cfg.sim.physics.solver_cfg.ls_iterations = 50
    env_cfg.sim.physics.solver_cfg.ccd_iterations = 35
    env_cfg.sim.physics.solver_cfg.njmax = 4096
    env_cfg.sim.physics.solver_cfg.nconmax = 4096
    env_cfg.sim.physics.collision_cfg = NewtonCollisionPipelineCfg(
        sdf_hydroelastic_config=HydroelasticSDFCfg(reduce_contacts=True, normal_matching=True)
    )
    env_cfg.scene.env_spacing = 1.5
    env_cfg.scene.replicate_physics = True
    env_cfg.events.randomize_franka_joint_state = None
    return env_cfg

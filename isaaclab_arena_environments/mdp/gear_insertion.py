# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Newton physics configuration for the gear insertion environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_experimental.envs import MDP2RLEnvCfg

    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import IsaacLabArenaManagerBasedRLEnvCfg


def _set_gear_insertion_newton_physics(env_cfg: IsaacLabArenaManagerBasedRLEnvCfg | MDP2RLEnvCfg) -> None:
    """Set the Newton contact dynamics, control rate, and robot friction of gear insertion in place."""
    from isaaclab_newton.physics import NewtonCfg

    assert isinstance(env_cfg.sim.physics, NewtonCfg), "gear_insertion requires the Newton physics backend"

    # Keep contact integration at 960 Hz without running Python managers at that rate.
    env_cfg.sim.dt = 1.0 / 60.0
    env_cfg.decimation = 2
    env_cfg.sim.render_interval = 2
    env_cfg.sim.physics.num_substeps = 16
    env_cfg.sim.physics.default_shape_cfg.gap = 0.0
    # The gear and base use convex pieces, including an open bore. MuJoCo-Warp
    # recomputes their contact manifolds at each solver substep.
    env_cfg.sim.physics.solver_cfg.use_mujoco_contacts = True
    env_cfg.sim.physics.solver_cfg.enable_multiccd = False
    env_cfg.sim.physics.solver_cfg.impratio = 1.0
    env_cfg.sim.physics.solver_cfg.iterations = 50
    env_cfg.sim.physics.solver_cfg.ls_iterations = 50
    env_cfg.sim.physics.solver_cfg.ccd_iterations = 35
    env_cfg.sim.physics.solver_cfg.njmax = 2048
    env_cfg.sim.physics.solver_cfg.nconmax = 512
    env_cfg.sim.physics.collision_cfg = None
    env_cfg.scene.env_spacing = 1.5
    env_cfg.scene.robot.spawn.physics_material.static_friction = 0.8
    env_cfg.scene.robot.spawn.physics_material.dynamic_friction = 0.8
    env_cfg.scene.robot.spawn.physics_material.torsional_friction = 0.005


def gear_insertion_newton_env_cfg_callback(
    env_cfg: IsaacLabArenaManagerBasedRLEnvCfg,
) -> IsaacLabArenaManagerBasedRLEnvCfg:
    """Configure Newton contact dynamics for gear insertion."""
    _set_gear_insertion_newton_physics(env_cfg)
    env_cfg.events.randomize_franka_joint_state = None
    return env_cfg


def gear_insertion_newton_mdp2_env_cfg_callback(env_cfg: MDP2RLEnvCfg) -> MDP2RLEnvCfg:
    """Configure Newton contact dynamics for the MDP2 version of gear insertion."""
    _set_gear_insertion_newton_physics(env_cfg)
    return env_cfg

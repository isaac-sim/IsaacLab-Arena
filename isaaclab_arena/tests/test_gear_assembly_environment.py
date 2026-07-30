# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for the Arena Gear Assembly scene setup."""

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function


def _test_gear_assembly_scene_and_newton_cfg(simulation_app):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena_environments.gear_assembly_environment import (
        GearAssemblyEnvironment,
        GearAssemblyEnvironmentCfg,
    )

    arena_env = GearAssemblyEnvironment().build(GearAssemblyEnvironmentCfg())

    assert arena_env.name == "gear_assembly"
    assert arena_env.embodiment.name == "droid_abs_joint_pos"
    assert arena_env.rl_framework_entry_point is None
    assert arena_env.rl_policy_cfg is None

    assets = arena_env.scene.assets
    assert list(assets) == [
        "ground",
        "factory_gear_base",
        "factory_gear_small",
        "factory_gear_medium",
        "factory_gear_large",
        "light",
    ]
    assert assets["factory_gear_base"].object_cfg.spawn.activate_contact_sensors is False
    assert assets["factory_gear_base"].object_cfg.spawn.rigid_props.kinematic_enabled is True
    assert assets["factory_gear_small"].object_cfg.spawn.activate_contact_sensors is False
    assert assets["factory_gear_small"].object_cfg.spawn.rigid_props.kinematic_enabled is False

    builder = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=1))
    env_cfg, _ = builder.compose_manager_cfg()
    solver_cfg = env_cfg.sim.physics.solver_cfg

    assert type(env_cfg.sim.physics).__name__ == "NewtonCfg"
    assert type(solver_cfg).__name__ == "MJWarpSolverCfg"
    assert solver_cfg.solver == "newton"
    assert solver_cfg.integrator == "implicitfast"
    assert solver_cfg.use_mujoco_contacts is False
    assert env_cfg.scene.replicate_physics is True
    assert env_cfg.sim.dt == 1.0 / 120.0
    assert env_cfg.decimation == 4
    assert env_cfg.episode_length_s == 6.66
    return True


def test_gear_assembly_scene_and_newton_cfg():
    assert run_simulation_app_function(_test_gear_assembly_scene_and_newton_cfg, headless=True)

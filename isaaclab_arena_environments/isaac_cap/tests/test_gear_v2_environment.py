# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the gear-mesh v2 graph-owned environment configuration."""

from pathlib import Path

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

pytestmark = pytest.mark.isaac_cap


def _test_gear_graph_owns_physics_configuration(_simulation_app) -> bool:
    from copy import deepcopy

    from isaaclab_newton.physics import NewtonCfg, NewtonCollisionPipelineCfg

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environment_spec.env_cfg_override import apply_env_cfg_override
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import (
        ArenaPhysicsCfg,
        IsaacLabArenaManagerBasedRLEnvCfg,
    )
    from isaaclab_arena.utils.physics_backend import PhysicsBackend
    from isaaclab_arena_environments.isaac_cap import register_components
    from isaaclab_arena_environments.isaac_cap.gear_insertion_v2 import gear_mesh_environment

    register_components()
    graph_paths = (
        Path(gear_mesh_environment.__file__).with_name("gear_easy.yaml"),
        Path(gear_mesh_environment.__file__).with_name("gear_easy_pair.yaml"),
        Path(gear_mesh_environment.__file__).with_name("gear_medium_train.yaml"),
    )
    specs = [ArenaEnvGraphSpec.from_yaml(path) for path in graph_paths]
    assert all(spec.default_physics_backend is PhysicsBackend.NEWTON for spec in specs)
    assert all(spec.env_cfg_override == specs[0].env_cfg_override for spec in specs)

    env_cfg = IsaacLabArenaManagerBasedRLEnvCfg()
    env_cfg.sim.physics = deepcopy(ArenaPhysicsCfg().newton)
    apply_env_cfg_override(env_cfg, specs[0].env_cfg_override)

    physics = env_cfg.sim.physics
    assert isinstance(physics, NewtonCfg)
    assert env_cfg.sim.dt == pytest.approx(1.0 / 60.0)
    assert env_cfg.decimation == 1
    assert physics.num_substeps == 16
    assert physics.collision_decimation == 1
    assert not physics.use_cuda_graph
    assert physics.default_shape_cfg.ke == 60_000.0
    assert physics.default_shape_cfg.kd == 500.0
    assert physics.default_shape_cfg.gap == 5.0e-5
    assert physics.solver_cfg.update_data_interval == 1
    assert physics.solver_cfg.disable_sensors
    assert physics.solver_cfg.njmax == 32768
    assert physics.solver_cfg.nconmax == 16384
    assert not physics.solver_cfg.use_mujoco_contacts
    assert physics.solver_cfg.iterations == 100
    assert physics.solver_cfg.ls_iterations == 50
    assert physics.solver_cfg.cone == "elliptic"
    assert physics.solver_cfg.impratio == 10.0
    assert isinstance(physics.collision_cfg, NewtonCollisionPipelineCfg)
    assert not physics.collision_cfg.reduce_contacts
    assert physics.collision_cfg.rigid_contact_max == 32768
    assert physics.collision_cfg.max_triangle_pairs == 1_000_000
    return True


def test_gear_graph_owns_physics_configuration() -> None:
    assert run_function_with_persistent_simulation_app(_test_gear_graph_owns_physics_configuration)

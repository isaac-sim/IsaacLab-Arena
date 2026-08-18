# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Registration, configuration, and Newton smoke tests for the deformable-cube environment."""

from __future__ import annotations

import torch

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

ENVIRONMENT_NAME = "franka_deformable_cube_pick_and_place"


def _test_franka_deformable_cube_environment_registration_and_config(simulation_app) -> bool:
    """The registered factory builds the intended fixed-pose Arena description."""
    from isaaclab.sim.spawners.meshes.meshes_cfg import MeshCuboidCfg

    from isaaclab_arena.assets.deformable_object import DeformableObject
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.assets.registries import EnvironmentRegistry
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena_environments.cli import ensure_environments_registered
    from isaaclab_arena_environments.franka_deformable_cube_pick_and_place_environment import (
        FrankaDeformableCubePickAndPlaceEnvironment,
        FrankaDeformableCubePickAndPlaceEnvironmentCfg,
    )

    ensure_environments_registered()
    registry = EnvironmentRegistry()
    factory_type = registry.get_component_by_name(ENVIRONMENT_NAME)
    assert factory_type is FrankaDeformableCubePickAndPlaceEnvironment
    assert registry.get_environment_cfg_type(factory_type) is FrankaDeformableCubePickAndPlaceEnvironmentCfg

    arena_env = factory_type().build(FrankaDeformableCubePickAndPlaceEnvironmentCfg())
    assert arena_env.default_physics_preset == "newton_mjwarp_vbd"
    assert arena_env.embodiment.name == "franka_ik"
    assert isinstance(arena_env.task, PickAndPlaceTask)

    cube = arena_env.scene.assets["deformable_cube"]
    destination = arena_env.scene.assets["plate"]
    table = arena_env.scene.assets["maple_table_robolab"]
    assert isinstance(cube, DeformableObject)
    assert isinstance(cube._source, MeshCuboidCfg)
    assert cube.object_type is ObjectType.DEFORMABLE
    assert destination.object_type is ObjectType.RIGID
    assert isinstance(cube.get_initial_pose(), Pose)
    assert isinstance(destination.get_initial_pose(), Pose)
    assert arena_env.task.pick_up_object is cube
    assert arena_env.task.destination_location is destination
    assert arena_env.task.background_scene is table
    return True


def test_franka_deformable_cube_environment_registration_and_config() -> None:
    """The registered factory builds the intended fixed-pose Arena description."""
    assert run_function_with_persistent_simulation_app(
        _test_franka_deformable_cube_environment_registration_and_config,
        headless=True,
    )


def _test_franka_deformable_cube_newton_smoke(simulation_app) -> bool:
    """Spawn, reset, step, evaluate support, and close through the runner construction path."""
    from isaaclab_contrib.deformable.newton_manager_cfg import CoupledMJWarpVBDSolverCfg

    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser

    env = None
    try:
        parser = get_isaaclab_arena_environments_cli_parser(get_isaaclab_arena_cli_parser())
        args_cli = parser.parse_args(["--num_envs", "1", ENVIRONMENT_NAME])
        builder = get_arena_builder_from_cli(args_cli)
        env_cfg, env_kwargs = builder.compose_manager_cfg()
        assert isinstance(env_cfg.sim.physics.solver_cfg, CoupledMJWarpVBDSolverCfg)
        assert env_cfg.scene.replicate_physics

        env = builder.make_registered(env_cfg, env_kwargs)
        observation, _ = env.reset()
        assert observation is not None
        assert {
            "robot",
            "deformable_cube",
            "plate",
            "maple_table_robolab",
        } <= set(env.unwrapped.scene.keys())

        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        with torch.inference_mode():
            env.step(actions)
            env.step(actions)

            placement_cfg = builder.arena_env.task.get_termination_cfg().success.params["predicates"][0]
            placement_result = placement_cfg.func(env, **placement_cfg.params)
        assert placement_result.shape == (1,)
        assert placement_result.dtype == torch.bool
        return True
    finally:
        if env is not None:
            env.close()


def test_franka_deformable_cube_newton_smoke() -> None:
    """The registered environment completes its Newton lifecycle without errors."""
    assert run_function_with_persistent_simulation_app(
        _test_franka_deformable_cube_newton_smoke,
        headless=True,
    )

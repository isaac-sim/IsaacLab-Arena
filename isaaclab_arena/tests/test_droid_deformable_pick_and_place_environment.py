# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Registration, configuration, and PhysX smoke tests for the deformable-object environment."""

from __future__ import annotations

import torch

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

ENVIRONMENT_NAME = "droid_deformable_pick_and_place"


def _test_droid_deformable_environment_registration_and_config(simulation_app) -> bool:
    """The registered factory builds the selected deformable object and DROID embodiment."""
    import isaaclab.sim as sim_utils

    from isaaclab_arena.assets.deformable_object import DeformableObject
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.assets.registries import EnvironmentRegistry
    from isaaclab_arena.relations.relations import NextTo, On, Side, get_relation
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena_environments.cli import (
        ensure_environments_registered,
        get_arena_builder_from_cli,
        get_isaaclab_arena_environments_cli_parser,
    )
    from isaaclab_arena_environments.droid_deformable_pick_and_place_environment import (
        DroidDeformablePickAndPlaceEnvironment,
        DroidDeformablePickAndPlaceEnvironmentCfg,
    )

    ensure_environments_registered()
    registry = EnvironmentRegistry()
    factory_type = registry.get_component_by_name(ENVIRONMENT_NAME)
    assert factory_type is DroidDeformablePickAndPlaceEnvironment
    assert registry.get_environment_cfg_type(factory_type) is DroidDeformablePickAndPlaceEnvironmentCfg

    arena_env = factory_type().build(DroidDeformablePickAndPlaceEnvironmentCfg())
    assert arena_env.embodiment.name == "droid_abs_joint_pos"
    assert isinstance(arena_env.task, PickAndPlaceTask)

    pick_object = arena_env.scene.assets["pick_object"]
    destination = arena_env.scene.assets["plate"]
    table = arena_env.scene.assets["maple_table_robolab"]
    table_reference = arena_env.scene.assets["table"]
    assert isinstance(pick_object, DeformableObject)
    assert isinstance(pick_object.spawner_cfg, sim_utils.MeshCuboidCfg)
    assert pick_object.spawner_cfg.size == (0.15, 0.04, 0.04)
    assert pick_object.object_type is ObjectType.DEFORMABLE
    assert destination.object_type is ObjectType.RIGID
    assert isinstance(table_reference, ObjectReference)
    assert table_reference.is_anchor
    assert get_relation(pick_object, On).parent is table_reference
    assert get_relation(destination, On).parent is table_reference
    next_to = get_relation(pick_object, NextTo)
    assert next_to.parent is destination
    assert next_to.side is Side.POSITIVE_Y
    assert arena_env.task.pick_up_object is pick_object
    assert arena_env.task.destination_location is destination
    assert arena_env.task.background_scene is table
    assert arena_env.task.contact_sensor_name is None

    parser = get_isaaclab_arena_environments_cli_parser()
    args_cli = parser.parse_args(
        [ENVIRONMENT_NAME, "--pick_object", "teddy_bear", "--embodiment", "droid_differential_ik"]
    )
    teddy_env = get_arena_builder_from_cli(args_cli).arena_env
    assert teddy_env.embodiment.name == "droid_differential_ik"
    assert isinstance(teddy_env.scene.assets["pick_object"].spawner_cfg, sim_utils.UsdFileCfg)

    args_cli = parser.parse_args([ENVIRONMENT_NAME, "--pick_object", "surface"])
    surface_env = get_arena_builder_from_cli(args_cli).arena_env
    assert isinstance(surface_env.scene.assets["pick_object"].spawner_cfg, sim_utils.MeshRectangleCfg)
    assert surface_env.scene.assets["pick_object"].spawner_cfg.size == (0.2, 0.2)
    assert surface_env.scene.assets["pick_object"].spawner_cfg.resolution == (30, 30)
    return True


def test_droid_deformable_environment_registration_and_config() -> None:
    """The registered factory builds each supported CLI-selected object."""
    assert run_function_with_persistent_simulation_app(
        _test_droid_deformable_environment_registration_and_config,
        headless=True,
    )


def _test_droid_deformable_physx_smoke(simulation_app) -> bool:
    """Spawn, reset, step, evaluate placement, and close through the runner construction path."""
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser

    env = None
    try:
        parser = get_isaaclab_arena_environments_cli_parser(get_isaaclab_arena_cli_parser())
        args_cli = parser.parse_args(["--num_envs", "1", ENVIRONMENT_NAME])
        builder = get_arena_builder_from_cli(args_cli)
        env_cfg, env_kwargs = builder.compose_manager_cfg()

        env = builder.make_registered(env_cfg, env_kwargs)
        assert "isaaclab_physx" in str(env.unwrapped.sim.physics_manager)
        observation, _ = env.reset()
        assert observation is not None
        assert {
            "robot",
            "pick_object",
            "plate",
            "table",
            "maple_table_robolab",
        } <= set(env.unwrapped.scene.keys())

        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        with torch.inference_mode():
            env.step(actions)
            env.step(actions)

            success_term = builder.arena_env.task.get_termination_cfg().success
            placement_result = success_term.func(env, **success_term.params)
        assert placement_result.shape == (1,)
        assert placement_result.dtype == torch.bool
        return True
    finally:
        if env is not None:
            env.close()


def test_droid_deformable_physx_smoke() -> None:
    """The registered environment completes its PhysX lifecycle without errors."""
    assert run_function_with_persistent_simulation_app(
        _test_droid_deformable_physx_smoke,
        headless=True,
    )

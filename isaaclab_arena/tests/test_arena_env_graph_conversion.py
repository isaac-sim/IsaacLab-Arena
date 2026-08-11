# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test for graph-spec -> live IsaacLabArenaEnvironment conversion.

Lives apart from in-process graph-spec validation tests on purpose: those call
``spec.validate()``, which transitively imports ``pxr`` (relation-class resolution). The
persistent in-process ``SimulationApp`` here cannot start if ``pxr`` was imported first, so the
sim test must not share a process with those pxr-importing tests. Keeping it solo lets the app
launch cleanly before any ``pxr`` import.
"""

from pathlib import Path

import pytest

from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
from isaaclab_arena.environment_spec.arena_env_graph_types import (
    AssetSpec,
    CompositeTaskSpec,
    TaskCompositionType,
    TaskSpec,
)

TEST_DATA_DIR = Path(__file__).parent / "test_data"


def _test_arena_env_graph_conversion_builds_sequential_pick_and_place_task(simulation_app):
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase

    spec = ArenaEnvGraphSpec.from_yaml(TEST_DATA_DIR / "pick_and_place_maple_table_env_graph.yaml")
    arena_env = spec.to_arena_env()

    assert arena_env.name == "pick_and_place_maple_table_default"
    assert isinstance(arena_env.task, SequentialTaskBase)
    assert arena_env.task.desired_subtask_success_state is None
    assert len(arena_env.task.subtasks) == 2
    assert all(isinstance(subtask, PickAndPlaceTask) for subtask in arena_env.task.subtasks)
    assert arena_env.task.subtasks[0].pick_up_object.name == "rubiks_cube_hot3d_robolab"
    assert arena_env.task.subtasks[1].pick_up_object.name == "mug_ycb_robolab"
    assert all(subtask.destination_location.name == "bowl_ycb_robolab" for subtask in arena_env.task.subtasks)
    assert all(subtask.background_scene.name == "maple_table_robolab" for subtask in arena_env.task.subtasks)

    return True


def test_arena_env_graph_conversion_builds_sequential_pick_and_place_task():

    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    result = run_function_with_persistent_simulation_app(
        _test_arena_env_graph_conversion_builds_sequential_pick_and_place_task
    )
    assert result


def _test_composite_task_episode_length_sums_subtasks(simulation_app):
    spec = ArenaEnvGraphSpec.from_yaml(TEST_DATA_DIR / "two_subtask_episode_length_env_graph.yaml")
    arena_env = spec.to_arena_env()

    assert [subtask.get_episode_length_s() for subtask in arena_env.task.subtasks] == [20.0, 30.0]
    assert arena_env.task.get_episode_length_s() == 50.0

    return True


def test_composite_task_episode_length_sums_subtasks():
    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    result = run_function_with_persistent_simulation_app(_test_composite_task_episode_length_sums_subtasks)
    assert result


def _test_get_arena_builder_from_cli_builds_env_from_graph_yaml(simulation_app):
    import argparse
    import sys

    from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser

    yaml_path = str(TEST_DATA_DIR / "pick_and_place_maple_table_env_graph.yaml")

    # --env_spec with no example-environment subcommand: parses (subcommand is
    # optional) and the runner builds the env from the graph spec instead of the registry.
    sys.argv = ["policy_runner.py", "--env_spec", yaml_path]
    args = get_isaaclab_arena_environments_cli_parser().parse_args()

    builder = get_arena_builder_from_cli(args)
    assert builder.arena_env.name == "pick_and_place_maple_table_default"

    # The flags the YAML declares under `cli_override_specs` are registered dynamically by the
    # environments parser (not hardcoded). Confirm --object parses through that real parser
    # path and that apply_cli_override_args swaps the declared target asset's registry_name.
    sys.argv = ["policy_runner.py", "--env_spec", yaml_path, "--object", "dex_cube"]
    args = get_isaaclab_arena_environments_cli_parser().parse_args()
    assert args.object == "dex_cube"
    spec = ArenaEnvGraphSpec.from_yaml(yaml_path)
    spec.apply_cli_override_args(args)
    cube = next(obj for obj in spec.objects if obj.id == "rubiks_cube_hot3d_robolab")
    assert cube.registry_name == "dex_cube"

    # A non-existent --env_spec fails with a clear "not found" assertion from the YAML
    # loader, not an opaque FileNotFoundError. The parser hits it while building, when it reads the
    # graph's declared override flags.
    sys.argv = ["policy_runner.py", "--env_spec", "/no/such/env_graph.yaml"]
    with pytest.raises(AssertionError, match="not found"):
        get_isaaclab_arena_environments_cli_parser()

    # Neither source, or both at once, is rejected by the exactly-one-source assert.
    for bad in (
        argparse.Namespace(env_spec=None, example_environment=None),
        argparse.Namespace(env_spec=yaml_path, example_environment="lift_object"),
    ):
        with pytest.raises(AssertionError):
            get_arena_builder_from_cli(bad)

    return True


def test_get_arena_builder_from_cli_builds_env_from_graph_yaml():

    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    result = run_function_with_persistent_simulation_app(_test_get_arena_builder_from_cli_builds_env_from_graph_yaml)
    assert result


def _test_arena_env_graph_conversion_builds_object_set_node(simulation_app):
    from isaaclab_arena.assets.object_set import RigidObjectSet

    spec = ArenaEnvGraphSpec.from_yaml(TEST_DATA_DIR / "object_set_maple_table_env_graph.yaml")
    arena_env = spec.to_arena_env()

    object_set = arena_env.scene.assets["pick_up_object_set"]
    assert isinstance(object_set, RigidObjectSet)
    assert len(object_set.objects) == 2
    assert len(object_set.member_usd_paths) == 2
    assert object_set.random_choice

    # The set is a single node: the task manipulates it, and each env gets one of its members.
    assert arena_env.task.pick_up_object is object_set
    object_set.assign_variants(num_envs=4)
    assert len(object_set.object_usd_paths) == 4

    return True


def test_arena_env_graph_conversion_builds_object_set_node():
    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    result = run_function_with_persistent_simulation_app(_test_arena_env_graph_conversion_builds_object_set_node)
    assert result


def _minimal_scene_spec(*, objects: list[AssetSpec]) -> ArenaEnvGraphSpec:
    return ArenaEnvGraphSpec(
        env_name="lighting_test",
        embodiment=AssetSpec(id="robot", registry_name="droid_abs_joint_pos"),
        background=AssetSpec(id="background", registry_name="maple_table_robolab"),
        objects=objects,
        task=CompositeTaskSpec(
            composition=TaskCompositionType.ATOMIC,
            description="noop task",
            subtasks=[
                TaskSpec(
                    kind="PickAndPlaceTask",
                    params={
                        "pick_up_object": objects[0].id,
                        "destination_location": objects[0].id,
                        "background_scene": "background",
                    },
                )
            ],
        ),
    )


def _lights_of_type(arena_env, light_cls) -> list:
    return [asset for asset in arena_env.scene.assets.values() if isinstance(asset, light_cls)]


def _test_default_light_is_injected_when_scene_has_none(simulation_app):
    from isaaclab_arena.assets.object_library import DirectionalLight, DomeLight

    # A single YCB object with no light asset and no light baked into its USD: the converter
    # must inject a default light so the env does not render black.
    spec = _minimal_scene_spec(objects=[AssetSpec(id="mug", registry_name="mug_ycb_robolab")])
    arena_env = spec.to_arena_env()

    assert len(_lights_of_type(arena_env, DomeLight)) == 1

    # The directional light comes along so lighting variations have a target, but is off until one
    # of its variations activates it.
    directional_lights = _lights_of_type(arena_env, DirectionalLight)
    assert len(directional_lights) == 1
    assert directional_lights[0].spawner_cfg.intensity == 0.0

    # An explicit light suppresses injection — no double-lighting, and no directional light either.
    explicit = _minimal_scene_spec(
        objects=[
            AssetSpec(id="mug", registry_name="mug_ycb_robolab"),
            AssetSpec(id="my_light", registry_name="light"),
        ]
    )
    explicit_env = explicit.to_arena_env()
    assert len(_lights_of_type(explicit_env, DomeLight)) == 1
    assert len(_lights_of_type(explicit_env, DirectionalLight)) == 0

    # The injected lights own their spawner cfgs: turning the directional light off above must not
    # have darkened the shared class default for later builds.
    assert DirectionalLight.default_spawner_cfg.intensity == DirectionalLight.default_intensity

    return True


def test_default_light_is_injected_when_scene_has_none():
    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    result = run_function_with_persistent_simulation_app(_test_default_light_is_injected_when_scene_has_none)
    assert result


def _test_direction_variation_lights_injected_directional_light(simulation_app):
    from isaaclab_arena.assets.object_library import DirectionalLight, DomeLight

    spec = _minimal_scene_spec(objects=[AssetSpec(id="mug", registry_name="mug_ycb_robolab")])
    arena_env = spec.to_arena_env()
    dome_light = _lights_of_type(arena_env, DomeLight)[0]
    directional_light = _lights_of_type(arena_env, DirectionalLight)[0]

    direction_variation = directional_light.get_variation("direction")
    direction_variation.enable()
    direction_variation.configure_at_build_time()

    # Enabling the variation lights the sun and dims the dome so the sun's shadows are visible.
    assert directional_light.spawner_cfg.intensity == DirectionalLight.default_intensity
    assert dome_light.spawner_cfg.intensity == direction_variation.cfg.dome_intensity_when_active

    return True


def test_direction_variation_lights_injected_directional_light():
    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    result = run_function_with_persistent_simulation_app(_test_direction_variation_lights_injected_directional_light)
    assert result

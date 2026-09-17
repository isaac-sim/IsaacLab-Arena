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
    from isaaclab_arena.environment_spec.arena_env_graph_conversion_utils import build_arena_env_from_graph_spec
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase

    spec = ArenaEnvGraphSpec.from_yaml(TEST_DATA_DIR / "pick_and_place_maple_table_env_graph.yaml")
    arena_env = build_arena_env_from_graph_spec(spec)

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


def test_object_reference_uses_runtime_parent_name():
    from types import SimpleNamespace
    from unittest.mock import patch

    from isaaclab_arena.environment_spec.arena_env_graph_conversion_utils import _instantiate_object_reference
    from isaaclab_arena.environment_spec.arena_env_graph_types import ObjectReferenceSpec

    parent = SimpleNamespace(name="renamed_fixture")
    reference = ObjectReferenceSpec(id="floor", parent_id="fixture_node", prim_path="inside/floor", object_type="base")
    with patch("isaaclab_arena.environment_spec.arena_env_graph_conversion_utils.ObjectReference") as constructor:
        _instantiate_object_reference(reference, parent)
    assert constructor.call_args.kwargs["prim_path"] == "{ENV_REGEX_NS}/renamed_fixture/inside/floor"
    assert constructor.call_args.kwargs["parent_asset"] is parent


def test_initial_pose_validation_and_reset_contract():
    from isaaclab_arena.environment_spec.arena_env_graph_conversion_utils import _get_pose_from_dict
    from isaaclab_arena.tests.dummy_object import DummyObject
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    asset = DummyObject("cube", AxisAlignedBoundingBox((0, 0, 0), (1, 1, 1)))
    asset.maybe_set_initial_pose(_get_pose_from_dict({"position_xyz": [1, 2, 3], "rotation_xyzw": [0, 0, 0, 1]}))
    assert asset.get_initial_pose().position_xyz == (1.0, 2.0, 3.0)
    event = asset._pose_event_cfg
    assert asset.has_pose_reset_event()
    asset.maybe_set_initial_pose(None)
    assert asset.get_initial_pose().position_xyz == (1.0, 2.0, 3.0)
    assert asset._pose_event_cfg is event
    for bad in (
        {"position_xyz": [float("nan"), 0, 0]},
        {"position_xyz": [True, 0, 0]},
        {"rotation_xyzw": [0, 0, 0, 0]},
        {"rotation_xyzw": [0, 0, 1]},
        {"unknown": 1},
    ):
        with pytest.raises(AssertionError):
            _get_pose_from_dict(bad)


def test_partial_initial_pose_preserves_authored_components():
    from isaaclab_arena.environment_spec.arena_env_graph_conversion_utils import _get_pose_from_dict
    from isaaclab_arena.tests.dummy_object import DummyObject
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose, PosePerEnv

    asset = DummyObject(
        "cube", AxisAlignedBoundingBox((0, 0, 0), (1, 1, 1)), Pose((1.0, 2.0, 3.0), (1.0, 0.0, 0.0, 0.0))
    )
    asset.maybe_set_initial_pose(_get_pose_from_dict({"position_xyz": [4, 5, 6]}, asset.get_initial_pose()))
    assert asset.get_initial_pose().rotation_xyzw == (1.0, 0.0, 0.0, 0.0)
    asset.maybe_set_initial_pose(_get_pose_from_dict({"rotation_xyzw": [0, 0, 0, 1]}, asset.get_initial_pose()))
    assert asset.get_initial_pose().position_xyz == (4.0, 5.0, 6.0)
    assert asset.has_pose_reset_event()
    with pytest.raises(AssertionError, match="fixed default pose"):
        _get_pose_from_dict({"position_xyz": [4, 5, 6]}, PosePerEnv([Pose()]))


def test_no_task_accepts_base_constructor_parameters():
    from isaaclab_arena.tasks.no_task import NoTask

    task = NoTask(episode_length_s=12, task_description="inspect scene")
    assert task.episode_length_s == 12
    assert task.task_description == "inspect scene"


def _test_companion_layout_paths_and_graph_ids(simulation_app):
    import tempfile
    import yaml

    from isaaclab_arena.relations.placement_layouts import PlacementLayouts
    from isaaclab_arena.utils.pose import Pose

    source = Path(__file__).parents[2] / "isaaclab_arena_examples/relations/clutter/clutter_scene.yaml"
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        data = yaml.safe_load(source.read_text())
        data["embodiment"]["id"] = "arm"
        data["placement_layouts"] = "poses.yaml"
        path = directory / "env.yaml"
        path.write_text(yaml.safe_dump(data))
        poses = {f"cube_{i}": [Pose((i, 0, 1))] for i in range(4)}
        poses["arm"] = [Pose((0, 0, 0))]
        PlacementLayouts(poses).write_yaml(directory / "poses.yaml")
        spec = ArenaEnvGraphSpec.from_yaml(path)
        restored = ArenaEnvGraphSpec.from_dict(spec.to_dict())
        with pytest.raises(AssertionError, match="Relative placement_layouts requires a source YAML"):
            restored.to_arena_env()
        assert restored.to_arena_env(placement_layouts=directory / "poses.yaml").placement_layouts is not None
        loaded = spec.to_arena_env().placement_layouts
        assert loaded.poses["robot"] == poses["arm"]
        assert "arm" not in loaded.poses
        override = directory / "override.yaml"
        poses["cube_0"] = [Pose((2, 3, 4))]
        PlacementLayouts(poses).write_yaml(override)
        assert spec.to_arena_env(placement_layouts=override).placement_layouts.poses["cube_0"] == poses["cube_0"]
        for invalid, message in (({"unknown": [Pose()]}, "Unknown cached"), ({"cube_0": [Pose()]}, "missing placed")):
            invalid_path = directory / f"{message.split()[0]}.yaml"
            PlacementLayouts(invalid).write_yaml(invalid_path)
            with pytest.raises(AssertionError, match=message):
                spec.to_arena_env(placement_layouts=invalid_path)
    return True


def test_companion_layout_paths_and_graph_ids():
    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    assert run_function_with_persistent_simulation_app(_test_companion_layout_paths_and_graph_ids)


def _test_python_environment_loads_companion_layouts(simulation_app):
    import sys
    import tempfile
    from unittest.mock import patch

    from isaaclab_arena.relations.placement_layouts import PlacementLayouts
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "poses.yaml"
        with patch.object(sys, "argv", ["environment_runner.py", "droid_table_multi_object_placement"]):
            args = get_isaaclab_arena_environments_cli_parser().parse_args()
        original = get_arena_builder_from_cli(args).arena_env
        poses = {
            asset.get_scene_key(): [Pose((0.1 * i, 0, 1)), Pose((0.1 * i, 0.2, 1))]
            for i, asset in enumerate(original.scene.assets.values())
            if asset.get_spatial_relations() and not asset.is_anchor
        }
        assert poses
        PlacementLayouts(poses).write_yaml(path)
        with patch.object(
            sys,
            "argv",
            ["environment_runner.py", "--placement_layouts", str(path), "droid_table_multi_object_placement"],
        ):
            args = get_isaaclab_arena_environments_cli_parser().parse_args()
        loaded = get_arena_builder_from_cli(args).arena_env
        assert loaded.placement_layouts.poses == poses
        for invalid, message in (
            ({"unknown": [Pose()]}, "Unknown cached"),
            ({next(iter(poses)): [Pose()]}, "missing placed"),
        ):
            invalid_path = Path(directory) / f"{message.split()[0]}.yaml"
            PlacementLayouts(invalid).write_yaml(invalid_path)
            args.placement_layouts = str(invalid_path)
            with pytest.raises(AssertionError, match=message):
                get_arena_builder_from_cli(args)
    return True


def test_python_environment_loads_companion_layouts():
    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    assert run_function_with_persistent_simulation_app(_test_python_environment_loads_companion_layouts)


def _test_cached_graph_preserves_physics_settings(simulation_app, tmp_path, preset):
    import yaml

    from isaaclab_newton.physics import NewtonCfg
    from isaaclab_physx.physics import PhysxCfg

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.relations.placement_layouts import PlacementLayouts
    from isaaclab_arena.utils.pose import Pose

    source = Path(__file__).parents[2] / "isaaclab_arena_examples/relations/clutter/clutter_scene.yaml"
    data = yaml.safe_load(source.read_text())
    data["default_physics_backend"] = "newton"
    data["env_cfg_override"] = {"sim": {"dt": 0.007}}
    data["placement_layouts"] = "poses.yaml"
    path = tmp_path / "env.yaml"
    path.write_text(yaml.safe_dump(data))
    PlacementLayouts({f"cube_{i}": [Pose((i, 0, 1))] for i in range(4)}).write_yaml(tmp_path / "poses.yaml")
    arena_env = ArenaEnvGraphSpec.from_yaml(path).to_arena_env()
    builder = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(presets=preset))
    cfg, _ = builder.compose_manager_cfg()
    assert isinstance(cfg.sim.physics, NewtonCfg if preset is None else PhysxCfg)
    assert cfg.sim.dt == pytest.approx(0.007)
    assert cfg.scene.replicate_physics is (preset is None)
    assert cfg.events.cached_placement_reset.params["poses"].keys() == arena_env.placement_layouts.poses.keys()
    return True


@pytest.mark.parametrize("preset", [None, "physx"])
def test_cached_graph_preserves_physics_settings(tmp_path, preset):
    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    assert run_function_with_persistent_simulation_app(
        _test_cached_graph_preserves_physics_settings, tmp_path=tmp_path, preset=preset
    )

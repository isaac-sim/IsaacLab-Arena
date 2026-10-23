# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for AssetSelectionVariation, on its own and through ArenaEnvBuilder.build_env_cfg()."""

import json

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

HEADLESS = True

TABLE_TOP_Z = 0.1
ON_CLEARANCE_M = 0.01
NUM_ENVS = 2
DEFAULT_REGISTRY_NAME = "cracker_box"
SELECTION_KEY = "pickup.asset_selection"


def get_registered_usd_path(registry_name):
    """Return the USD path of the registered object ``registry_name``."""
    from isaaclab_arena.assets.registries import AssetRegistry

    return AssetRegistry().get_asset_by_name(registry_name).usd_path


def make_table():
    """Return a fixed 2 m by 2 m table, 0.1 m thick, with its top at TABLE_TOP_Z."""
    import isaaclab.sim as sim_utils

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.relations.relations import IsAnchor
    from isaaclab_arena.utils.pose import Pose

    return Object(
        name="table",
        object_type=ObjectType.BASE,
        spawner_cfg=sim_utils.CuboidCfg(size=(2.0, 2.0, 0.1), collision_props=sim_utils.CollisionPropertiesCfg()),
        initial_pose=Pose(position_xyz=(0.0, 0.0, 0.05), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
        relations=[IsAnchor()],
    )


def compose_pick_and_place_with_selection(
    *,
    selection_cfg,
    rebuild_index=0,
    solve_relations=False,
):
    """Compose a PickAndPlaceTask environment whose pickup, a cracker box on a table, has an asset selection.

    Args:
        selection_cfg: Settings applied to the pickup's asset selection.
        rebuild_index: Index of the build among the rebuilds of an Experiment Run.
        solve_relations: Whether to solve the pickup's placement on the table.

    Returns:
        The pickup Object, the composed env_cfg and the env_kwargs.
    """
    from types import SimpleNamespace

    import isaaclab.sim as sim_utils

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.relations.relations import On
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.utils.pose import Pose

    table = make_table()
    pickup = AssetRegistry().get_asset_by_name(DEFAULT_REGISTRY_NAME)(name="pickup")
    pickup.add_relation(On(table, clearance_m=ON_CLEARANCE_M))
    pickup.get_variation("asset_selection").apply_cfg(selection_cfg)
    plate_spawn_cfg = sim_utils.CuboidCfg(
        size=(0.3, 0.3, 0.02),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    plate_pose = Pose(position_xyz=(0.7, 0.7, 0.11), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
    plate = Object(name="plate", object_type=ObjectType.RIGID, spawner_cfg=plate_spawn_cfg, initial_pose=plate_pose)
    # The namespace stands in for a background; the task reads only its object_min_z.
    task = PickAndPlaceTask(pickup, plate, SimpleNamespace(object_min_z=-1.0))
    scene = Scene(assets=[table, pickup, plate])
    arena_env = IsaacLabArenaEnvironment(name="test_asset_selection_variation", scene=scene, task=task)
    # resolve_on_reset=False stores the solved pose per environment on the object. Only the config is composed.
    builder_cfg = ArenaEnvBuilderCfg(
        num_envs=NUM_ENVS,
        solve_relations=solve_relations,
        placement_seed=0,
        resolve_on_reset=False,
    )
    arena_builder = ArenaEnvBuilder(arena_env, builder_cfg, rebuild_index=rebuild_index)
    env_cfg, env_kwargs = arena_builder.build_env_cfg()
    return pickup, env_cfg, env_kwargs


def _test_selected_asset_is_spawned_placed_recorded_and_described(simulation_app):
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariationCfg

    # In order, rebuild 1 takes the second name.
    pickup, env_cfg, env_kwargs = compose_pick_and_place_with_selection(
        selection_cfg=AssetSelectionVariationCfg(enabled=True, registry_names=["mustard_bottle", "sugar_box"]),
        rebuild_index=1,
        solve_relations=True,
    )

    assert env_cfg.scene.pickup.spawn.usd_path == get_registered_usd_path("sugar_box")
    # The center of the sugar box rests 13 mm lower than the center of the cracker box would.
    sugar_box_bottom_z = AssetRegistry().get_asset_by_name("sugar_box")().get_bounding_box().min_point[0, 2].item()
    expected_z = TABLE_TOP_Z + ON_CLEARANCE_M - sugar_box_bottom_z
    placed_z = [pose.position_xyz[2] for pose in pickup.get_initial_pose().poses]
    assert placed_z == pytest.approx([expected_z] * NUM_ENVS, abs=1e-3)
    selection_record = env_kwargs["variation_recorder"][SELECTION_KEY]
    recorded_names = [selection_record.sample_for_episode(env_id, 0) for env_id in range(NUM_ENVS)]
    assert recorded_names == ["sugar_box"] * NUM_ENVS
    # The pickup keeps the name of its role, and its registry name and the description name the selected asset.
    assert pickup.name == "pickup"
    assert pickup.registry_name == "sugar_box"
    # The selection sets registry_name on the object only, so its class keeps the declared registry name.
    assert type(pickup).registry_name == DEFAULT_REGISTRY_NAME
    assert env_cfg.task_description == "Pick up the sugar_box, and place it into the plate"
    return True


def _test_asset_selection_rejects_invalid_configuration(simulation_app, registry_names, message_pattern):
    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariationCfg

    with pytest.raises(AssertionError, match=message_pattern):
        compose_pick_and_place_with_selection(
            selection_cfg=AssetSelectionVariationCfg(enabled=True, registry_names=registry_names)
        )
    return True


def _test_rigid_objects_have_disabled_asset_selection(simulation_app):
    import isaaclab.sim as sim_utils

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.assets.registries import AssetRegistry

    # A rigid Object, even one outside the object library, has a disabled selection.
    cube = Object(name="cube", object_type=ObjectType.RIGID, spawner_cfg=sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)))
    assert not cube.get_variation("asset_selection").enabled
    # A registered library object that is not rigid has none.
    ground_plane = AssetRegistry().get_asset_by_name("ground_plane")()
    assert "asset_selection" not in ground_plane.variations
    return True


def test_selected_asset_is_spawned_placed_recorded_and_described():
    assert run_function_with_persistent_simulation_app(
        _test_selected_asset_is_spawned_placed_recorded_and_described, headless=HEADLESS
    )


@pytest.mark.parametrize(
    "registry_names, message_pattern",
    [
        (["sugar_box", "banana_split"], "'banana_split' is not a registered asset"),
        (["ground_plane"], r"'ground_plane' .* is not a rigid object"),
        (["droid_differential_ik"], r"'droid_differential_ik' .* is not a rigid object"),
    ],
)
def test_asset_selection_rejects_invalid_configuration(registry_names, message_pattern):
    assert run_function_with_persistent_simulation_app(
        _test_asset_selection_rejects_invalid_configuration,
        headless=HEADLESS,
        registry_names=registry_names,
        message_pattern=message_pattern,
    )


def test_rigid_objects_have_disabled_asset_selection():
    assert run_function_with_persistent_simulation_app(
        _test_rigid_objects_have_disabled_asset_selection, headless=HEADLESS
    )


class _RecordingRigidObject:
    """Stands in for the Object whose asset a selection sets, and records each selected registry name."""

    def __init__(self):
        self.name = "pickup"
        self.bounding_box = None
        self.selected_registry_names: list[str] = []

    def set_spawned_asset_class(self, registry_name, spawn_cfg):
        self.selected_registry_names.append(registry_name)


@pytest.fixture
def placeholder_spawn_cfgs(monkeypatch):
    """Resolve registry names to placeholder spawn cfgs instead of building the registered assets.

    Building the registered assets needs a SimulationApp, so the tests that use this fixture run in plain Python.
    """
    from isaaclab_arena.variations import asset_selection_variation

    monkeypatch.setattr(
        asset_selection_variation, "_get_rigid_object_spawn_cfg", lambda registry_name: f"spawn cfg of {registry_name}"
    )


def test_random_choice_override_draws_seeded_names_and_swaps_the_sampler(placeholder_spawn_cfgs):
    import torch

    from isaaclab_arena.variations import variations_hydra
    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariation

    rigid_object = _RecordingRigidObject()
    variation = AssetSelectionVariation(rigid_object)
    variations = {"pickup": [variation]}

    def select_registry_name(seed, rebuild_index=0):
        variation.configure_at_build_time(seed=seed, variation_key=SELECTION_KEY, rebuild_index=rebuild_index)
        return rigid_object.selected_registry_names[-1]

    variations_hydra.apply_overrides(
        variations,
        [
            f"{SELECTION_KEY}.enabled=true",
            f"{SELECTION_KEY}.random_choice=true",
            f"{SELECTION_KEY}.registry_names=[mustard_bottle,sugar_box,tomato_soup_can]",
        ],
    )
    # A draw from torch's global random state would differ between these two builds.
    torch.manual_seed(0)
    first_registry_name = select_registry_name(seed=7)
    torch.manual_seed(2)
    assert select_registry_name(seed=7) == first_registry_name
    # In order, every build seed would take the first name.
    assert len({select_registry_name(seed) for seed in range(5)}) > 1

    variations_hydra.apply_overrides(variations, [f"{SELECTION_KEY}.random_choice=false"])
    # In order, rebuild 1 takes the second name.
    assert select_registry_name(seed=0, rebuild_index=1) == "sugar_box"

    # Set on the cfg directly, random_choice leaves the in-order sampler bound, so the build fails.
    variation.cfg.random_choice = True
    with pytest.raises(AssertionError, match="passing a cfg to apply_cfg"):
        select_registry_name(seed=0)


def test_in_order_selection_takes_repeated_names_in_list_order(placeholder_spawn_cfgs):
    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariation, AssetSelectionVariationCfg

    registry_names = ["mustard_bottle", "sugar_box", "mustard_bottle", "tomato_soup_can"]
    rigid_object = _RecordingRigidObject()
    variation = AssetSelectionVariation(
        rigid_object, AssetSelectionVariationCfg(enabled=True, registry_names=registry_names)
    )

    num_rebuilds = 6
    for rebuild_index in range(num_rebuilds):
        # Like _cfg_for_rebuild, give each rebuild another seed, which must not change the order.
        variation.configure_at_build_time(seed=rebuild_index, variation_key=SELECTION_KEY, rebuild_index=rebuild_index)

    # Rebuild k takes entry k % 4, so mustard_bottle spawns in rebuilds 0, 2 and 4.
    assert rigid_object.selected_registry_names == registry_names + registry_names[:2]


def test_asset_selection_rejects_empty_registry_names_and_bounding_box_override(placeholder_spawn_cfgs):
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariation, AssetSelectionVariationCfg

    selection_without_names = AssetSelectionVariation(
        _RecordingRigidObject(), AssetSelectionVariationCfg(enabled=True, registry_names=[])
    )
    with pytest.raises(AssertionError, match="'pickup.asset_selection' is enabled but its"):
        selection_without_names.configure_at_build_time(seed=0, variation_key=SELECTION_KEY)

    rigid_object_with_bounding_box = _RecordingRigidObject()
    rigid_object_with_bounding_box.bounding_box = AxisAlignedBoundingBox(
        min_point=(0.0, 0.0, 0.0), max_point=(0.1, 0.1, 0.1)
    )
    selection_with_bounding_box = AssetSelectionVariation(
        rigid_object_with_bounding_box, AssetSelectionVariationCfg(enabled=True, registry_names=["sugar_box"])
    )
    with pytest.raises(AssertionError, match="'pickup' sets a bounding_box override"):
        selection_with_bounding_box.configure_at_build_time(seed=0, variation_key=SELECTION_KEY)


def test_placement_replay_needs_recorded_asset_selection(placeholder_spawn_cfgs, tmp_path):
    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariation, AssetSelectionVariationCfg
    from isaaclab_arena.variations.recorded_variation_replay import configure_recorded_variation_replay

    def replay(record_row, record_name):
        """Return the names an in-order selection of mustard_bottle and sugar_box spawns while replaying ``record_row``."""
        rigid_object = _RecordingRigidObject()
        variation = AssetSelectionVariation(
            rigid_object, AssetSelectionVariationCfg(enabled=True, registry_names=["mustard_bottle", "sugar_box"])
        )
        record_path = tmp_path / f"{record_name}.jsonl"
        record_path.write_text(json.dumps(record_row) + "\n")
        # Like ArenaEnvBuilder.build_env_cfg, bind the record before the build-time variations are configured.
        configure_recorded_variation_replay(record_path, {"pickup": [variation]})
        variation.configure_at_build_time(seed=0, variation_key=SELECTION_KEY)
        return rigid_object.selected_registry_names

    # One layout row as write_placement_samples writes it: a pose for pickup and no variation values.
    pickup_pose = {"position_xyz": [0.0, 0.0, 0.05], "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]}
    layout_row = {"placement": {"layout_id": "layout_000000", "poses": {"pickup": pickup_pose}}}
    with pytest.raises(AssertionError, match=r"no 'pickup\.asset_selection' value"):
        replay(layout_row, "layout_without_selection")

    # In order, a live draw in the first rebuild would take "mustard_bottle".
    layout_row_with_selection = {**layout_row, "variations": {SELECTION_KEY: "sugar_box"}}
    assert replay(layout_row_with_selection, "layout_with_selection") == ["sugar_box"]

# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for AssetSelectionVariation, through ArenaEnvBuilder.compose_manager_cfg()."""

import json

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

HEADLESS = True

TABLE_TOP_Z = 0.1
ON_CLEARANCE_M = 0.01
NUM_ENVS = 2
SELECTION_KEY = "pickup.asset_selection"
ABC_CANDIDATE_HEIGHTS = {"a": 0.1, "b": 0.2, "c": 0.3}


def make_cuboid_candidate(candidate_name, height):
    """Return a ``(name, CuboidCfg)`` candidate for a 0.1 m by 0.1 m rigid cuboid of the given height."""
    import isaaclab.sim as sim_utils

    spawn_cfg = sim_utils.CuboidCfg(
        size=(0.1, 0.1, height),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        activate_contact_sensors=True,
    )
    return candidate_name, spawn_cfg


def make_abc_candidates():
    """Return the candidates a, b and c, with the heights in ABC_CANDIDATE_HEIGHTS."""
    return [make_cuboid_candidate(name, height) for name, height in ABC_CANDIDATE_HEIGHTS.items()]


def compose_pick_and_place_with_selection(
    selection, *, seed=42, default_asset=None, hydra_overrides=None, recorded_variation_samples_path=None
):
    """Compose a PickAndPlaceTask environment whose pickup Object, on a table, carries ``selection``.

    Returns:
        The pickup Object, the composed env_cfg and the env_kwargs.
    """
    from types import SimpleNamespace

    import isaaclab.sim as sim_utils

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.relations.relations import IsAnchor, On
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.utils.pose import Pose

    # A fixed 2 m by 2 m table, 0.1 m thick, with its top at TABLE_TOP_Z.
    table = Object(
        name="table",
        object_type=ObjectType.BASE,
        spawner_cfg=sim_utils.CuboidCfg(size=(2.0, 2.0, 0.1), collision_props=sim_utils.CollisionPropertiesCfg()),
        initial_pose=Pose(position_xyz=(0.0, 0.0, 0.05), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
        relations=[IsAnchor()],
    )
    # asset=None declares the pickup without an asset of its own.
    pickup = Object(name="pickup", asset=default_asset, relations=[On(table, clearance_m=ON_CLEARANCE_M)])
    pickup.add_variation(selection)
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
        seed=seed,
        placement_seed=0,
        resolve_on_reset=False,
        recorded_variation_samples_path=recorded_variation_samples_path,
    )
    env_cfg, env_kwargs = ArenaEnvBuilder(arena_env, builder_cfg, hydra_overrides=hydra_overrides).compose_manager_cfg()
    return pickup, env_cfg, env_kwargs


def _test_selected_asset_is_spawned_placed_and_recorded(simulation_app):
    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariation
    from isaaclab_arena.variations.variations_hydra import overrides_from_dict

    short_candidate = make_cuboid_candidate("short", 0.1)
    tall_candidate = make_cuboid_candidate("tall", 0.3)
    # Declared disabled next to a default asset; the build's overrides enable it and keep only "tall".
    selection = AssetSelectionVariation(asset_candidates=[short_candidate, tall_candidate])
    hydra_overrides = overrides_from_dict({f"{SELECTION_KEY}.enabled": True, f"{SELECTION_KEY}.asset_names": ["tall"]})

    pickup, env_cfg, env_kwargs = compose_pick_and_place_with_selection(
        selection, default_asset=short_candidate, hydra_overrides=hydra_overrides
    )

    assert env_cfg.scene.pickup.spawn.size[2] == pytest.approx(0.3)
    placed_z = [pose.position_xyz[2] for pose in pickup.get_initial_pose().poses]
    expected_z = TABLE_TOP_Z + ON_CLEARANCE_M + 0.3 / 2
    assert placed_z == pytest.approx([expected_z] * NUM_ENVS, abs=1e-3)
    selection_record = env_kwargs["variation_recorder"][SELECTION_KEY]
    assert [selection_record.sample_for_episode(env_id, 0) for env_id in range(NUM_ENVS)] == ["tall"] * NUM_ENVS
    return True


def _test_in_order_selection_follows_build_seed(simulation_app):
    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariation, AssetSelectionVariationCfg

    spawned_heights = []
    for seed in (42, 43, 44):
        selection = AssetSelectionVariation(
            asset_candidates=make_abc_candidates(),
            cfg=AssetSelectionVariationCfg(enabled=True, asset_names=["c", "a", "b"]),
        )
        _, env_cfg, _ = compose_pick_and_place_with_selection(selection, seed=seed)
        spawned_heights.append(env_cfg.scene.pickup.spawn.size[2])

    # 42 % 3 == 0, so seed 42 takes the first of asset_names and each next seed the next one.
    assert spawned_heights == pytest.approx([ABC_CANDIDATE_HEIGHTS[name] for name in ("c", "a", "b")])
    return True


def _test_random_selection_depends_only_on_seed_and_key(simulation_app):
    import torch

    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariation, AssetSelectionVariationCfg

    def compose_random_selection(seed):
        selection = AssetSelectionVariation(
            asset_candidates=make_abc_candidates(),
            cfg=AssetSelectionVariationCfg(enabled=True, random_choice=True),
        )
        _, env_cfg, _ = compose_pick_and_place_with_selection(selection, seed=seed)
        return env_cfg.scene.pickup.spawn.size[2]

    # A draw from torch's global random state would differ between these two builds.
    torch.manual_seed(0)
    first_height = compose_random_selection(seed=7)
    torch.manual_seed(2)
    second_height = compose_random_selection(seed=7)
    assert first_height == second_height

    heights_over_seeds = {compose_random_selection(seed) for seed in range(20)}
    assert len(heights_over_seeds) > 1
    return True


def _test_recorded_asset_name_is_replayed(simulation_app, tmp_path):
    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariation, AssetSelectionVariationCfg

    recorded_samples_path = tmp_path / "asset_selection_replay.jsonl"
    recorded_row = json.dumps({"variations": {SELECTION_KEY: "tall"}})
    recorded_samples_path.write_text(f"{recorded_row}\n" * NUM_ENVS)
    # In order with seed 42, a live draw over [short, tall] takes "short" (42 % 2 == 0).
    selection = AssetSelectionVariation(
        asset_candidates=[make_cuboid_candidate("short", 0.1), make_cuboid_candidate("tall", 0.3)],
        cfg=AssetSelectionVariationCfg(enabled=True),
    )

    _, env_cfg, env_kwargs = compose_pick_and_place_with_selection(
        selection, seed=42, recorded_variation_samples_path=str(recorded_samples_path)
    )

    assert env_cfg.scene.pickup.spawn.size[2] == pytest.approx(0.3)
    assert env_kwargs["variation_recorder"][SELECTION_KEY].sample_for_episode(0, 0) == "tall"
    return True


def _test_asset_selection_rejects_invalid_configuration(simulation_app, case_id):
    import isaaclab.sim as sim_utils

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.variations import variations_hydra
    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariation

    candidate = make_cuboid_candidate("short", 0.1)

    def add_to_rigid_object_set():
        members = [
            Object(name=name, object_type=ObjectType.RIGID, spawner_cfg=spawn_cfg)
            for name, spawn_cfg in make_abc_candidates()
        ]
        RigidObjectSet(name="pickup", objects=members).add_variation(AssetSelectionVariation([candidate]))

    def add_to_base_object():
        table = Object(name="table", object_type=ObjectType.BASE, spawner_cfg=sim_utils.CuboidCfg(size=(1.0, 1.0, 0.1)))
        table.add_variation(AssetSelectionVariation([candidate]))

    def add_to_object_with_bounding_box():
        pickup = Object(name="pickup")
        pickup.bounding_box = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.1, 0.1, 0.1))
        pickup.add_variation(AssetSelectionVariation([candidate]))

    def add_second_selection():
        pickup = Object(name="pickup")
        pickup.add_variation(AssetSelectionVariation([candidate]))
        pickup.add_variation(AssetSelectionVariation([candidate], name="second_asset_selection"))

    def add_one_selection_to_two_objects():
        selection = AssetSelectionVariation([candidate])
        Object(name="pickup").add_variation(selection)
        Object(name="other_pickup").add_variation(selection)

    def override_with_unknown_name():
        pickup = Object(name="pickup")
        pickup.add_variation(AssetSelectionVariation([candidate]))
        variations_hydra.apply_overrides({"pickup": pickup.get_variations()}, [f"{SELECTION_KEY}.asset_names=[banana]"])

    multi_asset_spawn_cfg = sim_utils.MultiAssetSpawnerCfg(assets_cfg=[candidate[1]])
    invalid_configurations = {
        "rigid_object_set": (add_to_rigid_object_set, "already spawns one member per environment"),
        "base_object": (add_to_base_object, "needs a rigid Object"),
        "bounding_box_override": (add_to_object_with_bounding_box, "bounding_box override"),
        "second_selection": (add_second_selection, "already has an AssetSelectionVariation"),
        "selection_on_two_objects": (add_one_selection_to_two_objects, "was already added to 'pickup'"),
        "no_candidates": (lambda: AssetSelectionVariation([]), "at least one"),
        "duplicate_names": (lambda: AssetSelectionVariation([candidate, candidate]), "Duplicate candidate name"),
        "multi_spawner_candidate": (
            lambda: AssetSelectionVariation([("several", multi_asset_spawn_cfg)]),
            (
                "AssetSelectionVariation 'asset_selection' candidate 0: 'several' is a MultiAssetSpawnerCfg; pass one"
                " spawn configuration"
            ),
        ),
        "unknown_name_in_override": (override_with_unknown_name, "unknown asset_names"),
    }
    build_invalid_configuration, message_fragment = invalid_configurations[case_id]
    with pytest.raises(AssertionError, match=message_fragment):
        build_invalid_configuration()
    return True


def _test_placement_replay_needs_recorded_asset_selection(simulation_app, tmp_path):
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariation, AssetSelectionVariationCfg

    pickup = Object(name="pickup")
    pickup.add_variation(
        AssetSelectionVariation([make_cuboid_candidate("cube", 0.1)], cfg=AssetSelectionVariationCfg(enabled=True))
    )
    # One layout row as write_placement_samples writes it: a pose for pickup and no variation values.
    pickup_pose = {"position_xyz": [0.0, 0.0, 0.05], "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]}
    layout_row = {"placement": {"layout_id": "layout_000000", "poses": {"pickup": pickup_pose}}}
    layouts_path = tmp_path / "layouts.jsonl"
    layouts_path.write_text(json.dumps(layout_row) + "\n")
    arena_env = IsaacLabArenaEnvironment(name="placement_replay_without_selection", scene=Scene(assets=[pickup]))
    builder_cfg = ArenaEnvBuilderCfg(num_envs=1, recorded_variation_samples_path=str(layouts_path))

    with pytest.raises(AssertionError, match=r"no 'pickup\.asset_selection' value"):
        ArenaEnvBuilder(arena_env, builder_cfg).compose_manager_cfg()
    assert pickup.object_cfg.spawn is None, "The selection drew a candidate before the replay check."
    return True


def test_selected_asset_is_spawned_placed_and_recorded():
    assert run_function_with_persistent_simulation_app(
        _test_selected_asset_is_spawned_placed_and_recorded, headless=HEADLESS
    )


def test_in_order_selection_follows_build_seed():
    assert run_function_with_persistent_simulation_app(_test_in_order_selection_follows_build_seed, headless=HEADLESS)


def test_random_selection_depends_only_on_seed_and_key():
    assert run_function_with_persistent_simulation_app(
        _test_random_selection_depends_only_on_seed_and_key, headless=HEADLESS
    )


def test_recorded_asset_name_is_replayed(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_recorded_asset_name_is_replayed, headless=HEADLESS, tmp_path=tmp_path
    )


@pytest.mark.parametrize(
    "case_id",
    [
        "rigid_object_set",
        "base_object",
        "bounding_box_override",
        "second_selection",
        "selection_on_two_objects",
        "no_candidates",
        "duplicate_names",
        "multi_spawner_candidate",
        "unknown_name_in_override",
    ],
)
def test_asset_selection_rejects_invalid_configuration(case_id):
    assert run_function_with_persistent_simulation_app(
        _test_asset_selection_rejects_invalid_configuration, headless=HEADLESS, case_id=case_id
    )


def test_placement_replay_needs_recorded_asset_selection(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_placement_replay_needs_recorded_asset_selection, headless=HEADLESS, tmp_path=tmp_path
    )

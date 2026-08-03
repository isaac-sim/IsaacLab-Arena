# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the generic SimReady USD object asset."""

from __future__ import annotations

from isaaclab_arena.assets.registries import AssetRegistry, ensure_assets_registered
from isaaclab_arena.environment_spec.arena_env_graph_types import AssetSpec

# SimReadyUsdObject is imported inside each test rather than here: it pulls in the asset base
# classes, which import pxr, and pxr loaded during collection breaks the simulation tests that
# start SimulationApp later in the same process.


def test_simready_usd_object_registered():
    from isaaclab_arena.assets.simready_object_library import SimReadyUsdObject

    ensure_assets_registered()
    cls = AssetRegistry().get_asset_by_name("simready_usd_object")
    assert cls is SimReadyUsdObject
    assert cls.name == "simready_usd_object"
    assert "sim-ready" in cls.tags


def test_asset_spec_accepts_simready_usd_path():
    ensure_assets_registered()
    spec = AssetSpec(
        id="hammer",
        registry_name="simready_usd_object",
        params={"usd_path": "https://example.com/hammer.usd"},
    )
    assert spec.registry_name == "simready_usd_object"
    assert spec.params["usd_path"] == "https://example.com/hammer.usd"


def test_asset_spec_drops_tags_copied_out_of_the_catalogue():
    from isaaclab_arena.assets.simready_object_library import SimReadyUsdObject

    ensure_assets_registered()
    # The catalogue lists every object as "name tags=[...]", so a generated spec often copies the
    # tags into params. The asset class passes its own tags to Asset.__init__, so a copy that
    # reached the constructor would be a duplicate keyword argument and fail the build.
    spec = AssetSpec(
        id="hammer",
        registry_name="simready_usd_object",
        params={"usd_path": "https://example.com/hammer.usd", "tags": ["sim-ready", "hammer"]},
    )
    assert "tags" not in spec.params
    assert spec.params["usd_path"] == "https://example.com/hammer.usd"
    assert "sim-ready" in SimReadyUsdObject.tags


def test_simready_usd_object_enables_physics_variant_by_default(tmp_path):
    from isaaclab_arena.assets.simready_object_library import SimReadyUsdObject
    from isaaclab_arena.tests.utils.usd_stages import add_body, new_stage

    ensure_assets_registered()
    # A real USD file, because the object reads it to work out its type.
    stage = new_stage()
    add_body(stage, "body_01")
    usd_path = str(tmp_path / "kettle.usd")
    assert stage.Export(usd_path)

    obj = SimReadyUsdObject(usd_path=usd_path, instance_name="kettle")
    assert obj.spawn_cfg_addon["variants"] == {"Physics": "physics"}
    spawn = obj._get_spawn_cfg(activate_contact_sensors=True)
    assert spawn.variants == {"Physics": "physics"}
    assert obj.usd_path == usd_path


def test_simready_search_registry_name_derives_an_identifier_from_the_phrase():
    from isaaclab_arena.assets.simready_object_library import simready_search_registry_name

    assert simready_search_registry_name("green trash can") == "simready_green_trash_can"
    assert simready_search_registry_name("  Chrome/Watering-Can  ") == "simready_chrome_watering_can"


def test_register_searched_simready_object_makes_it_a_catalogue_entry():
    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.assets.simready_object_library import register_searched_simready_object

    ensure_assets_registered()
    asset_cls = register_searched_simready_object(
        "green trash can", "s3://bucket/trash_can.usd", ("sim-ready", "green")
    )
    assert asset_cls.name == "simready_green_trash_can"
    assert AssetRegistry().get_asset_by_name("simready_green_trash_can") is asset_cls
    # Rigid, so the entry can also be an object_set member; the search only accepts one rigid body.
    assert asset_cls.object_type is ObjectType.RIGID
    assert "green" in asset_cls.tags


def test_a_searched_asset_resolves_its_usd_path_without_spec_params():
    from isaaclab_arena.assets.simready_object_library import register_searched_simready_object

    ensure_assets_registered()
    register_searched_simready_object("chrome watering can", "s3://bucket/watering_can.usd")
    # The path is baked into the registered class, so a spec naming it carries no params at all.
    spec = AssetSpec(id="watering_can", registry_name="simready_chrome_watering_can")
    assert spec.resolve_usd_path() == "s3://bucket/watering_can.usd"


def test_registering_the_same_phrase_twice_keeps_the_first_asset():
    from isaaclab_arena.assets.simready_object_library import register_searched_simready_object

    ensure_assets_registered()
    first = register_searched_simready_object("steel kettle", "s3://bucket/kettle_a.usd")
    second = register_searched_simready_object("steel kettle", "s3://bucket/kettle_b.usd")
    assert second is first

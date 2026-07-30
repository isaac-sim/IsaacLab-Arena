# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the generic SimReady USD object asset."""

from __future__ import annotations

from isaaclab_arena.assets.registries import AssetRegistry, ensure_assets_registered
from isaaclab_arena.assets.simready_object_library import SimReadyUsdObject
from isaaclab_arena.environment_spec.arena_env_graph_types import AssetSpec


def test_simready_usd_object_registered():
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
        params={"usd_path": "https://example.com/hammer.usd", "tags": ["sim-ready", "hammer"]},
    )
    assert spec.registry_name == "simready_usd_object"
    assert spec.params["usd_path"] == "https://example.com/hammer.usd"


def test_simready_usd_object_enables_physics_variant_by_default(tmp_path):
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

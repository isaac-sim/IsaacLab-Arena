# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for declaring an Object from an asset definition or without an asset."""

import re

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

HEADLESS = True


def _make_box_spawn_cfg(size: tuple[float, float, float]):
    import isaaclab.sim as sim_utils

    return sim_utils.CuboidCfg(
        size=size,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        activate_contact_sensors=True,
    )


def _test_get_asset_definition_copies_a_registered_rigid_object(simulation_app) -> bool:
    from isaaclab.sim import UsdFileCfg

    from isaaclab_arena.assets.registries import AssetRegistry

    asset_registry = AssetRegistry()
    asset_name, cracker_box_spawn_cfg = asset_registry.get_asset_definition("cracker_box")
    assert asset_name == "cracker_box"
    assert isinstance(cracker_box_spawn_cfg, UsdFileCfg)
    assert cracker_box_spawn_cfg.activate_contact_sensors is True

    _, sugar_box_spawn_cfg = asset_registry.get_asset_definition("sugar_box", scale=(1.5, 1.5, 1.5))
    assert sugar_box_spawn_cfg.scale == (1.5, 1.5, 1.5)

    with pytest.raises(AssertionError, match=re.escape("Asset 'ground_plane' is ObjectType.BASE, not rigid.")):
        asset_registry.get_asset_definition("ground_plane")
    return True


def test_get_asset_definition_copies_a_registered_rigid_object():
    result = run_function_with_persistent_simulation_app(
        _test_get_asset_definition_copies_a_registered_rigid_object, headless=HEADLESS
    )
    assert result, "Test failed"


def _test_object_declared_with_asset_definition_or_without_asset(simulation_app) -> bool:
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType

    object_with_asset = Object(name="pickup", asset=("box", _make_box_spawn_cfg(size=(0.1, 0.1, 0.3))))
    assert object_with_asset.object_type == ObjectType.RIGID
    _, object_cfg = object_with_asset.get_object_cfg()
    assert object_cfg.spawn.size == (0.1, 0.1, 0.3)

    object_without_asset = Object(name="pickup")
    assert object_without_asset.object_type == ObjectType.RIGID
    with pytest.raises(AssertionError, match="Object 'pickup' has no asset"):
        object_without_asset.get_object_cfg()
    with pytest.raises(AssertionError, match="Object 'pickup' has no asset"):
        object_without_asset.get_bounding_box()
    return True


def test_object_declared_with_asset_definition_or_without_asset():
    result = run_function_with_persistent_simulation_app(
        _test_object_declared_with_asset_definition_or_without_asset, headless=HEADLESS
    )
    assert result, "Test failed"


def _test_object_rejects_asset_with_other_spawn_inputs(simulation_app, case_id: str) -> bool:
    import isaaclab.sim as sim_utils

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType

    box_spawn_cfg = _make_box_spawn_cfg(size=(0.1, 0.1, 0.1))
    box_asset = ("box", box_spawn_cfg)
    multi_asset_spawn_cfg = sim_utils.MultiAssetSpawnerCfg(assets_cfg=[box_spawn_cfg])
    object_kwargs_and_message_by_case = {
        "asset_with_usd_path": (
            {"asset": box_asset, "usd_path": "box.usd"},
            "Object 'pickup': asset= cannot be combined with usd_path or spawner_cfg.",
        ),
        "asset_with_scale": (
            {"asset": box_asset, "scale": (2.0, 2.0, 2.0)},
            "Object 'pickup': set scale and spawn_cfg_addon in the asset's spawn configuration",
        ),
        "asset_with_multi_asset_spawner": (
            {"asset": ("several", multi_asset_spawn_cfg)},
            "Object 'pickup' asset=: 'several' is a MultiAssetSpawnerCfg; pass one spawn configuration",
        ),
        "no_asset_articulation_type": (
            {"object_type": ObjectType.ARTICULATION},
            "Object 'pickup' is declared with asset= or without an asset, so it must be rigid",
        ),
    }
    object_kwargs, expected_message = object_kwargs_and_message_by_case[case_id]
    with pytest.raises(AssertionError, match=re.escape(expected_message)):
        Object(name="pickup", **object_kwargs)
    return True


@pytest.mark.parametrize(
    "case_id",
    ["asset_with_usd_path", "asset_with_scale", "asset_with_multi_asset_spawner", "no_asset_articulation_type"],
)
def test_object_rejects_asset_with_other_spawn_inputs(case_id: str):
    result = run_function_with_persistent_simulation_app(
        _test_object_rejects_asset_with_other_spawn_inputs, headless=HEADLESS, case_id=case_id
    )
    assert result, "Test failed"

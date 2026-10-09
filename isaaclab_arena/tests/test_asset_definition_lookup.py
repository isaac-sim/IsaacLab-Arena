# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check registry definitions preserve native settings without scene state."""

from functools import wraps

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _with_simulation_app(test_function):
    @wraps(test_function)
    def run_test(*args, **kwargs):
        def run_in_app(simulation_app):
            test_function(*args, **kwargs)
            return True

        assert run_function_with_persistent_simulation_app(run_in_app)

    return run_test


def _make_object(**kwargs):
    from isaaclab.sim import CuboidCfg, MassPropertiesCfg, RigidBodyPropertiesCfg

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType

    return Object(
        name="constructor_instance",
        object_type=ObjectType.RIGID,
        spawn_cfg=CuboidCfg(
            size=(0.1, 0.2, 0.3),
            mass_props=MassPropertiesCfg(mass=0.4),
            rigid_props=RigidBodyPropertiesCfg(disable_gravity=True),
        ),
        **kwargs,
    )


def _registry_with_constructor(monkeypatch, constructor):
    from isaaclab_arena.assets.registries import AssetRegistry

    registry = AssetRegistry()
    monkeypatch.setitem(registry._components, "definition_lookup_test", constructor)
    return registry


@_with_simulation_app
def test_lookup_preserves_library_constructor_settings_and_registry_identity():
    from isaaclab.sim import UsdFileCfg

    from isaaclab_arena.assets.registries import AssetRegistry

    registry = AssetRegistry()
    definition_name, spawn_cfg = registry.get_asset_definition("mug", instance_name="pickup", scale=(0.7, 0.8, 0.9))

    assert definition_name == "mug"
    assert isinstance(spawn_cfg, UsdFileCfg)
    assert spawn_cfg.scale == (0.7, 0.8, 0.9)
    assert spawn_cfg.mass_props.mass == 0.25
    assert spawn_cfg.rigid_props.solver_position_iteration_count == 16
    assert spawn_cfg.rigid_props.disable_gravity is False


@_with_simulation_app
def test_lookup_snapshots_are_independent_of_the_source_and_each_other(monkeypatch):
    from isaaclab_arena.relations.relations import IsAnchor
    from isaaclab_arena.utils.pose import Pose

    source_object = _make_object(
        initial_pose=Pose(position_xyz=(1.0, 2.0, 3.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
        relations=[IsAnchor()],
    )
    source_object.get_variation("mass").enable()
    registry = _registry_with_constructor(monkeypatch, lambda: source_object)
    first_name, first_spawn_cfg = registry.get_asset_definition("definition_lookup_test")
    second_name, second_spawn_cfg = registry.get_asset_definition("definition_lookup_test")

    assert first_name == second_name == "definition_lookup_test"
    assert first_spawn_cfg is not source_object.spawn_cfg
    assert first_spawn_cfg is not second_spawn_cfg
    source_object.spawn_cfg.mass_props.mass = 10.0
    first_spawn_cfg.rigid_props.disable_gravity = False
    first_spawn_cfg.size = (2.0, 2.0, 2.0)
    assert first_spawn_cfg.mass_props.mass == 0.4
    assert second_spawn_cfg.mass_props.mass == 0.4
    assert second_spawn_cfg.rigid_props.disable_gravity is True
    assert second_spawn_cfg.size == (0.1, 0.2, 0.3)
    assert source_object.spawn_cfg.rigid_props.disable_gravity is True
    assert not hasattr(second_spawn_cfg, "initial_pose")
    assert not hasattr(second_spawn_cfg, "relations")
    assert not hasattr(second_spawn_cfg, "variations")


@pytest.mark.parametrize(
    "unsupported_kind",
    [
        "non_object",
        "base",
        "articulation",
        "object_set",
        "multi_asset",
        "multi_usd",
        "missing_spawn",
        "missing_object_cfg",
        "custom_bounds",
        "assigned",
    ],
)
@_with_simulation_app
def test_lookup_rejects_unsupported_definitions(monkeypatch, unsupported_kind):
    from isaaclab.sim import CuboidCfg, MultiAssetSpawnerCfg, MultiUsdFileCfg

    from isaaclab_arena.assets.asset import Asset
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    if unsupported_kind == "non_object":
        source_object = Asset(name="source")
    elif unsupported_kind in ("base", "articulation"):
        source_object = Object(
            name="source", object_type=ObjectType[unsupported_kind.upper()], spawn_cfg=CuboidCfg(size=(1.0,) * 3)
        )
    elif unsupported_kind == "object_set":
        source_object = RigidObjectSet(name="source", objects=[_make_object()])
    else:
        source_object = _make_object()
        if unsupported_kind == "multi_asset":
            source_object.object_cfg.spawn = MultiAssetSpawnerCfg(assets_cfg=[source_object.spawn_cfg])
        elif unsupported_kind == "multi_usd":
            source_object.object_cfg.spawn = MultiUsdFileCfg(usd_path=["unused.usd"])
        elif unsupported_kind == "missing_spawn":
            source_object.object_cfg.spawn = None
        elif unsupported_kind == "missing_object_cfg":
            source_object.object_cfg = None
        elif unsupported_kind == "custom_bounds":
            source_object.bounding_box = AxisAlignedBoundingBox((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        elif unsupported_kind == "assigned":
            source_object.bind_asset_assignment((0, 0))

    registry = _registry_with_constructor(monkeypatch, lambda: source_object)
    with pytest.raises(AssertionError, match="Asset 'definition_lookup_test'"):
        registry.get_asset_definition("definition_lookup_test")


@pytest.mark.parametrize("enabled", [False, True])
@_with_simulation_app
def test_lookup_rejects_attached_asset_selection(monkeypatch, enabled):
    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariation, AssetSelectionVariationCfg

    source_object = _make_object()
    source_object.add_variation(
        AssetSelectionVariation(
            asset_candidates=[("candidate", _make_object().spawn_cfg)],
            cfg=AssetSelectionVariationCfg(enabled=enabled),
        )
    )
    registry = _registry_with_constructor(monkeypatch, lambda: source_object)
    with pytest.raises(AssertionError, match="cannot have an asset selection variation"):
        registry.get_asset_definition("definition_lookup_test")

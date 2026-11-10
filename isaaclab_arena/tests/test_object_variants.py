# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for objects whose USD variant selections are given as a configclass."""

from __future__ import annotations

from pathlib import Path

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

HEADLESS = True


def _write_physics_variant_usd(path: Path) -> None:
    """Write a USD whose only rigid body, /Asset/Body, exists under the 'physics' variant of its 'Physics' set.

    Args:
        path: File to write.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    variant_set = root.GetPrim().GetVariantSets().AddVariantSet("Physics")
    variant_set.AddVariant("none")
    variant_set.AddVariant("physics")
    variant_set.SetVariantSelection("physics")
    with variant_set.GetVariantEditContext():
        body = UsdGeom.Xform.Define(stage, "/Asset/Body")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdGeom.Cube.Define(stage, "/Asset/Body/Cube")
    variant_set.SetVariantSelection("none")
    stage.GetRootLayer().Save()


def _test_object_accepts_configclass_variants(simulation_app, asset_dir: Path) -> bool:
    from isaaclab.utils.configclass import configclass

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType

    @configclass
    class PhysicsVariantsCfg:
        Physics: str = "physics"

    usd_path = asset_dir / "prop.usda"
    _write_physics_variant_usd(usd_path)

    # The rigid body exists only under the selected variant, so detecting RIGID shows the selection was applied.
    prop = Object(name="prop", usd_path=str(usd_path), spawn_cfg_addon={"variants": PhysicsVariantsCfg()})
    assert prop.object_type == ObjectType.RIGID

    # A contact sensor against a static object filters on the body nested under the selected variant.
    shelf = Object(
        name="shelf",
        usd_path=str(usd_path),
        object_type=ObjectType.BASE,
        spawn_cfg_addon={"variants": PhysicsVariantsCfg()},
    )
    contact_sensor_cfg = prop.get_contact_sensor_cfg(contact_against_object=shelf)
    assert contact_sensor_cfg.filter_prim_paths_expr == [shelf.get_prim_path() + "/Body"]
    return True


def test_object_accepts_configclass_variants(tmp_path: Path):
    result = run_function_with_persistent_simulation_app(
        _test_object_accepts_configclass_variants, headless=HEADLESS, asset_dir=tmp_path
    )
    assert result, "Test failed"

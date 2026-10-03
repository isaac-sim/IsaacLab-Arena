# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for rigid-body USD helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from isaaclab_arena.utils.usd.rigid_bodies import (
    find_shallowest_rigid_body,
    find_shallowest_rigid_body_from_stage,
    read_asset_rigid_body_paths,
)


def _write_physics_variant_usd(path: Path) -> None:
    """Write a minimal USD with Physics variants and two same-depth rigid bodies."""
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    variant_set = root.GetPrim().GetVariantSets().AddVariantSet("Physics")
    variant_set.AddVariant("none")
    variant_set.AddVariant("physics")
    variant_set.SetVariantSelection("none")

    variant_set.SetVariantSelection("physics")
    with variant_set.GetVariantEditContext():
        lid = UsdGeom.Xform.Define(stage, "/Root/Geometry/lid_obj")
        UsdPhysics.RigidBodyAPI.Apply(lid.GetPrim())
        body = UsdGeom.Xform.Define(stage, "/Root/Geometry/body_obj")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    variant_set.SetVariantSelection("none")
    stage.GetRootLayer().Save()


def test_read_asset_rigid_body_paths_counts_bodies_under_the_physics_variant(tmp_path: Path):
    from pxr import Usd

    usd_path = tmp_path / "prop.usda"
    _write_physics_variant_usd(usd_path)

    # A prop has no physics at all until its variant is selected, and two bodies once it is.
    assert read_asset_rigid_body_paths(str(usd_path)) == []
    assert len(read_asset_rigid_body_paths(str(usd_path), {"Physics": "physics"})) == 2

    # Reading the asset does not write the selection back into it.
    stage = Usd.Stage.Open(str(usd_path))
    variant_set = stage.GetDefaultPrim().GetVariantSets().GetVariantSet("Physics")
    assert variant_set.GetVariantSelection() == "none"


def test_find_shallowest_rigid_body_requires_physics_variant(tmp_path: Path):
    usd_path = tmp_path / "prop.usda"
    _write_physics_variant_usd(usd_path)

    # No physics until the variant is selected, and then two bodies tie for shallowest.
    assert find_shallowest_rigid_body(str(usd_path)) is None
    with pytest.raises(ValueError, match="Expected only one"):
        find_shallowest_rigid_body(str(usd_path), relative_to_root=True, variants={"Physics": "physics"})


def test_find_shallowest_rigid_body_from_stage_raises_on_a_tie(tmp_path: Path):
    from pxr import Usd

    usd_path = tmp_path / "prop.usda"
    _write_physics_variant_usd(usd_path)
    stage = Usd.Stage.Open(str(usd_path))
    stage.GetDefaultPrim().GetVariantSets().GetVariantSet("Physics").SetVariantSelection("physics")
    with pytest.raises(ValueError, match="Expected only one"):
        find_shallowest_rigid_body_from_stage(stage)


def _write_nested_rigid_bodies_usd(
    path: Path,
    body_specs: list[tuple[str, bool, bool]],
) -> None:
    """Write a root xform with rigid bodies; each spec is (prim_path, kinematic, enabled)."""
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    for prim_path, kinematic, enabled in body_specs:
        xform = UsdGeom.Xform.Define(stage, prim_path)
        body = UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
        body.CreateRigidBodyEnabledAttr(enabled)
        body.CreateKinematicEnabledAttr(kinematic)
    stage.GetRootLayer().Save()


def test_is_enabled_dynamic_rigid_body_classifies_reset_targets(tmp_path: Path):
    from pxr import Usd

    from isaaclab_arena.utils.usd.helpers import is_enabled_dynamic_rigid_body

    kinematic_path = tmp_path / "kinematic.usda"
    _write_nested_rigid_bodies_usd(
        kinematic_path,
        [
            ("/Root/kinematic_table", True, True),
            ("/Root/kinematic_table/leg", True, True),
        ],
    )
    kinematic_stage = Usd.Stage.Open(str(kinematic_path))
    table = kinematic_stage.GetPrimAtPath("/Root/kinematic_table")
    leg = kinematic_stage.GetPrimAtPath("/Root/kinematic_table/leg")
    assert not is_enabled_dynamic_rigid_body(table)
    assert not is_enabled_dynamic_rigid_body(leg)

    dynamic_path = tmp_path / "dynamic.usda"
    _write_nested_rigid_bodies_usd(dynamic_path, [("/Root/free_prop", False, True)])
    dynamic_stage = Usd.Stage.Open(str(dynamic_path))
    assert is_enabled_dynamic_rigid_body(dynamic_stage.GetPrimAtPath("/Root/free_prop"))

    mixed_path = tmp_path / "mixed.usda"
    _write_nested_rigid_bodies_usd(
        mixed_path,
        [
            ("/Root/kinematic_table", True, True),
            ("/Root/kinematic_table/prop", False, True),
        ],
    )
    mixed_stage = Usd.Stage.Open(str(mixed_path))
    mixed_table = mixed_stage.GetPrimAtPath("/Root/kinematic_table")
    mixed_prop = mixed_stage.GetPrimAtPath("/Root/kinematic_table/prop")
    assert not is_enabled_dynamic_rigid_body(mixed_table)
    assert is_enabled_dynamic_rigid_body(mixed_prop)

    disabled_dynamic_path = tmp_path / "disabled_dynamic.usda"
    _write_nested_rigid_bodies_usd(
        disabled_dynamic_path,
        [("/Root/off_prop", False, False)],
    )
    disabled_stage = Usd.Stage.Open(str(disabled_dynamic_path))
    assert not is_enabled_dynamic_rigid_body(disabled_stage.GetPrimAtPath("/Root/off_prop"))

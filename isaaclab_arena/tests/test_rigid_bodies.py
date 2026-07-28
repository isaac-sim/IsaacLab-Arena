# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for rigid-body USD helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from isaaclab_arena.utils.usd.rigid_bodies import find_shallowest_rigid_body, find_shallowest_rigid_body_from_stage


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


def test_find_shallowest_rigid_body_requires_physics_variant(tmp_path: Path):
    usd_path = tmp_path / "prop.usda"
    _write_physics_variant_usd(usd_path)

    assert find_shallowest_rigid_body(str(usd_path)) is None
    path = find_shallowest_rigid_body(
        str(usd_path),
        relative_to_root=True,
        variants={"Physics": "physics"},
        prefer_body_when_tied=True,
    )
    assert path == "/Geometry/body_obj"


def test_find_shallowest_rigid_body_from_stage_raises_without_preference(tmp_path: Path):
    from pxr import Usd

    usd_path = tmp_path / "prop.usda"
    _write_physics_variant_usd(usd_path)
    stage = Usd.Stage.Open(str(usd_path))
    stage.GetDefaultPrim().GetVariantSets().GetVariantSet("Physics").SetVariantSelection("physics")
    with pytest.raises(ValueError, match="Expected only one"):
        find_shallowest_rigid_body_from_stage(stage)

# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for rigid-body USD helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from isaaclab_arena.tests.utils.usd_stages import add_body, new_stage
from isaaclab_arena.utils.usd.rigid_bodies import (
    find_shallowest_rigid_body,
    find_shallowest_rigid_body_from_stage,
    freeze_loose_rigid_bodies,
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


def test_freeze_loose_rigid_bodies_preserves_jointed_bodies():
    from pxr import UsdPhysics

    stage = new_stage()
    loose_body_path = add_body(stage, "loose_prop")
    cabinet_body_path = add_body(stage, "cabinet")
    door_body_path = add_body(stage, "door")
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/Root/cabinet_door_joint")
    joint.CreateBody0Rel().SetTargets([cabinet_body_path])
    joint.CreateBody1Rel().SetTargets([f"{door_body_path}/door_mesh_00"])

    frozen_paths = freeze_loose_rigid_bodies(stage.GetDefaultPrim())

    assert frozen_paths == (loose_body_path,)
    assert UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(loose_body_path)).GetKinematicEnabledAttr().Get()
    assert UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(cabinet_body_path)).GetKinematicEnabledAttr().Get() is not True
    assert UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(door_body_path)).GetKinematicEnabledAttr().Get() is not True


def test_freeze_loose_rigid_bodies_preserves_world_joint_body():
    from pxr import UsdPhysics

    stage = new_stage()
    loose_body_path = add_body(stage, "loose_prop")
    door_body_path = add_body(stage, "door")
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/Root/world_door_joint")
    joint.CreateBody1Rel().SetTargets([f"{door_body_path}/door_mesh_00"])

    frozen_paths = freeze_loose_rigid_bodies(stage.GetDefaultPrim())

    assert frozen_paths == (loose_body_path,)
    assert UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(door_body_path)).GetKinematicEnabledAttr().Get() is not True


def test_freeze_loose_rigid_bodies_rejects_missing_joint_target():
    from pxr import UsdPhysics

    stage = new_stage()
    add_body(stage, "loose_prop")
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/Root/broken_joint")
    joint.CreateBody1Rel().SetTargets(["/Root/missing_body"])

    with pytest.raises(AssertionError, match="targets missing prim"):
        freeze_loose_rigid_bodies(stage.GetDefaultPrim())


def test_freeze_loose_rigid_bodies_does_not_follow_targets_outside_root():
    from pxr import UsdGeom, UsdPhysics

    stage = new_stage()
    loose_body_path = add_body(stage, "loose_prop")
    external_body = UsdGeom.Xform.Define(stage, "/ExternalBody")
    UsdPhysics.RigidBodyAPI.Apply(external_body.GetPrim())
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/Root/external_joint")
    joint.CreateBody1Rel().SetTargets(["/ExternalBody"])

    frozen_paths = freeze_loose_rigid_bodies(stage.GetDefaultPrim())

    assert frozen_paths == (loose_body_path,)


def test_freeze_loose_rigid_bodies_accepts_empty_root():
    stage = new_stage()
    frozen_paths = freeze_loose_rigid_bodies(stage.GetDefaultPrim())

    assert frozen_paths == ()

# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Asset-independent USD derivation, recipe, and package relocation checks."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from isaaclab_arena.assets.sources.derived import DerivedUsdSource

PIPELINE = "isaaclab_arena.assets.derivation.pipeline:derive_asset"


@pytest.fixture
def source_package(tmp_path):
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    package = tmp_path / "input"
    entry = package / "usd/item.usda"
    entry.parent.mkdir(parents=True)
    texture = package / "textures/color.png"
    texture.parent.mkdir()
    texture.write_bytes(b"texture")
    stage = Usd.Stage.CreateNew(str(entry))
    root = UsdGeom.Xform.Define(stage, "/Gadget").GetPrim()
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    variants = root.GetVariantSets().AddVariantSet("Quality")
    variants.AddVariant("Low")
    variants.AddVariant("High")
    variants.SetVariantSelection("High")
    with variants.GetVariantEditContext():
        root.CreateAttribute("quality", Sdf.ValueTypeNames.Token).Set("high")
    variants.SetVariantSelection("Low")
    body = UsdGeom.Xform.Define(stage, "/Gadget/Geometry/Body").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    collider = UsdGeom.Cube.Define(stage, body.GetPath().AppendChild("Cube")).GetPrim()
    UsdPhysics.CollisionAPI.Apply(collider)
    UsdPhysics.PrismaticJoint.Define(stage, "/Gadget/Joints/Slide")
    UsdPhysics.PrismaticJoint.Define(stage, "/Gadget/Joints/Other")
    shader = UsdShade.Shader.Define(stage, "/Gadget/Texture")
    shader.CreateIdAttr("UsdUVTexture")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("../textures/color.png"))
    stage.GetRootLayer().Save()
    return entry


def _write_recipe(tmp_path, source_path, **options):
    recipe = {
        "source_identity": str(source_path),
        "source_entry": "usd/item.usda",
        "output_entry": "usd/derived.usda",
        "operation": PIPELINE,
        "expected_source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        **options,
    }
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe))
    return path


@pytest.mark.parametrize("flatten", [False, True])
@pytest.mark.parametrize("scaled", [False, True])
def test_derivation_edits_a_non_workbench_asset_and_keeps_packages_relocatable(
    tmp_path,
    monkeypatch,
    source_package,
    flatten,
    scaled,
):
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    monkeypatch.setattr("isaaclab_arena.assets.derivation.cache.get_arena_asset_cache_dir", lambda: tmp_path / "cache")
    prefix = "BakedScale/" if scaled else ""
    body = prefix + "Geometry/Body"
    slide = prefix + "Joints/Slide"
    other = prefix + "Joints/Other"
    steps = [
        {
            "operation": "edit_prims",
            "paths": [body],
            "attributes": {
                "physics:mass": {"type": "Float", "value": 0.02, "custom": False},
                "calibrationPoints": {"type": "Float3Array", "value": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]},
            },
            "apply_api_schemas": ["PhysicsMassAPI"],
        },
        {
            "operation": "edit_prims",
            "paths": [body],
            "descendants": True,
            "has_api": "PhysicsCollisionAPI",
            "attributes": {"physics:collisionEnabled": {"type": "Bool", "value": False, "custom": False}},
        },
        {
            "operation": "edit_prims",
            "descendants": True,
            "is_a": "PhysicsJoint",
            "exclude_paths": [slide],
            "type_name": "PhysicsFixedJoint",
        },
        {
            "operation": "joint_drive",
            "path": slide,
            "instance": "linear",
            "settings": {"stiffness": 20.0, "damping": 0.2, "target_position": 0.0},
        },
        {"operation": "fixed_joint", "path": "FixedBase", "body1": body},
        {
            "operation": "preview_surface",
            "path": "Materials/Plastic",
            "bind_to": [body],
            "stronger_than_descendants": True,
            "inputs": {
                "diffuseColor": {"type": "Color3f", "value": [0.1, 0.2, 0.3]},
                "roughness": {"type": "Float", "value": 0.4},
            },
        },
    ]
    options = {
        "root_prim": "/Derived",
        "variants": {"Quality": "High"},
        "steps": steps,
        "flatten": flatten,
        "required_prims": [body, slide, "Materials/Plastic"],
    }
    if scaled:
        options["scale"] = [0.5, 0.5, 0.5]
    recipe = _write_recipe(tmp_path, source_package, **options)
    source_bytes = source_package.read_bytes()
    output = Path(DerivedUsdSource(str(source_package), str(recipe)).resolve())
    package = output.parents[1]
    relocated = tmp_path / "relocated"
    shutil.copytree(package, relocated)
    stage = Usd.Stage.Open(str(relocated / "usd/derived.usda"))
    assert not stage.GetCompositionErrors()
    assert UsdGeom.GetStageMetersPerUnit(stage) == 0.01
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y
    owner = stage.GetPrimAtPath("/Derived/BakedScale" if scaled else "/Derived")
    assert owner.GetAttribute("quality").Get() == "high"
    if scaled:
        assert tuple(owner.GetAttribute("xformOp:scale").Get()) == (0.5, 0.5, 0.5)
    body_prim = stage.GetPrimAtPath("/Derived/" + body)
    assert UsdPhysics.MassAPI(body_prim).GetMassAttr().Get() == pytest.approx(0.02)
    assert not body_prim.GetAttribute("physics:mass").IsCustom()
    assert [tuple(point) for point in body_prim.GetAttribute("calibrationPoints").Get()] == [(1, 2, 3), (4, 5, 6)]
    collider = stage.GetPrimAtPath(body_prim.GetPath().AppendChild("Cube"))
    assert not UsdPhysics.CollisionAPI(collider).GetCollisionEnabledAttr().Get()
    assert stage.GetPrimAtPath("/Derived/" + other).IsA(UsdPhysics.FixedJoint)
    drive = UsdPhysics.DriveAPI(stage.GetPrimAtPath("/Derived/" + slide), "linear")
    assert drive.GetStiffnessAttr().Get() == 20.0
    fixed = UsdPhysics.FixedJoint(stage.GetPrimAtPath("/Derived/FixedBase"))
    assert str(fixed.GetBody1Rel().GetTargets()[0]) == "/Derived/" + body
    material, _ = UsdShade.MaterialBindingAPI(body_prim).ComputeBoundMaterial()
    assert str(material.GetPath()) == "/Derived/Materials/Plastic"
    color = stage.GetPrimAtPath("/Derived/Materials/Plastic/Surface").GetAttribute("inputs:diffuseColor").Get()
    assert tuple(color) == pytest.approx((0.1, 0.2, 0.3))
    texture = stage.GetPrimAtPath("/Derived/" + prefix + "Texture").GetAttribute("inputs:file").Get()
    assert Path(texture.resolvedPath) == relocated / "textures/color.png"
    assert source_package.read_bytes() == source_bytes
    assert DerivedUsdSource(str(source_package), str(recipe)).resolve() == str(output)


@pytest.mark.parametrize(
    "step",
    [
        {"operation": "missing"},
        {"operation": "edit_prims", "paths": ["Missing"]},
        {"operation": "edit_prims", "paths": ["/OutsideAsset"]},
        {"operation": "edit_prims", "attributes": {"bad": {"type": "Unknown", "value": 1}}},
    ],
)
def test_invalid_steps_do_not_publish_a_derived_package(tmp_path, monkeypatch, source_package, step):
    cache = tmp_path / "cache"
    monkeypatch.setattr("isaaclab_arena.assets.derivation.cache.get_arena_asset_cache_dir", lambda: cache)
    recipe = _write_recipe(tmp_path, source_package, steps=[step])
    with pytest.raises(AssertionError):
        DerivedUsdSource(str(source_package), str(recipe)).resolve()
    assert not list((cache / "derived").glob("*/manifest.json"))
    assert not [path for path in (cache / "derived").iterdir() if path.is_dir()]


def test_recipe_can_partition_unrelated_mesh_components(tmp_path, monkeypatch):
    from pxr import Usd, UsdGeom

    from isaaclab_arena.utils.usd.mesh_partition import inspect_mesh_components

    source = tmp_path / "input/usd/item.usda"
    source.parent.mkdir(parents=True)
    stage = Usd.Stage.CreateNew(str(source))
    root = UsdGeom.Xform.Define(stage, "/Shape").GetPrim()
    stage.SetDefaultPrim(root)
    mesh = UsdGeom.Mesh.Define(stage, "/Shape/Mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0), (2, 0, 0), (3, 0, 0), (2, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3, 4, 5])
    stage.GetRootLayer().Save()
    digest = inspect_mesh_components(stage, "Mesh")["geometry_sha256"]
    recipe = _write_recipe(
        tmp_path,
        source,
        steps=[{
            "operation": "partition_mesh",
            "prim": "Mesh",
            "groups": {"Left": [0]},
            "remainder": "Right",
            "expected_geometry_sha256": digest,
        }],
        required_prims=["Mesh/Left", "Mesh/Right"],
    )
    monkeypatch.setattr("isaaclab_arena.assets.derivation.cache.get_arena_asset_cache_dir", lambda: tmp_path / "cache")
    derived = Usd.Stage.Open(DerivedUsdSource(str(source), str(recipe)).resolve())
    assert derived.GetPrimAtPath("/Asset/Mesh").IsA(UsdGeom.Xform)
    left = UsdGeom.Mesh(derived.GetPrimAtPath("/Asset/Mesh/Left"))
    right = UsdGeom.Mesh(derived.GetPrimAtPath("/Asset/Mesh/Right"))
    assert list(left.GetFaceVertexCountsAttr().Get()) == [3]
    assert list(right.GetFaceVertexCountsAttr().Get()) == [3]
    assert max(point[0] for point in left.GetPointsAttr().Get()) < min(
        point[0] for point in right.GetPointsAttr().Get()
    )


def test_derivation_imports_do_not_require_usd_kit_or_environment_packages():
    import subprocess
    import sys

    code = """
import sys
from isaaclab_arena.assets.derivation import cache, operations, pipeline
from isaaclab_arena.assets.sources.derived import DerivedUsdSource
assert not any(name.startswith(('pxr', 'omni', 'isaacsim', 'isaaclab_arena_environments')) for name in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr

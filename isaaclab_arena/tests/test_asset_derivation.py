# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Asset-independent USD derivation, recipe, and package relocation checks."""

from __future__ import annotations

import hashlib
import json

import pytest

from isaaclab_arena.assets.sources.derived import DerivedUsdSource


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
        "expected_source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        **options,
    }
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe))
    return path


@pytest.mark.parametrize(
    "step",
    [
        {"operation": "missing"},
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

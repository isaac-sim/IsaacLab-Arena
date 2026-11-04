# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""USD-only regression tests for mesh component partitioning."""

from __future__ import annotations

import numpy as np

import pytest

from isaaclab_arena_environments.workbench_cleanup.scripts.mesh_partition import (
    inspect_mesh_components,
    open_mesh_partition_stage,
    partition_mesh,
)


@pytest.fixture
def mesh_source(tmp_path):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    path = tmp_path / "source.usda"
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    body = UsdGeom.Xform.Define(stage, "/Asset/Body")
    body.AddTranslateOp().Set(Gf.Vec3d(2, 0, 0))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Asset/Body/Mesh")
    mesh.AddTranslateOp().Set(Gf.Vec3d(0, 3, 0))
    mesh.AddScaleOp().Set(Gf.Vec3f(2, 1, 1))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreatePointsAttr([
        (0, 0, 0.5),
        (1, 0, 0.5),
        (1, 1, 0.5),
        (0, 1, 0.5),
        (0, 0, 1.5),
        (1, 0, 1.5),
        (0, 1, 1.5),
        (999, 999, 999),
    ])
    mesh.CreateFaceVertexCountsAttr([4, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3, 4, 5, 6])
    mesh.CreateNormalsAttr([(0, 0, 1)] * 4 + [(0, 0, -1)] * 3)
    mesh.SetNormalsInterpolation("faceVarying")
    mesh.CreateHoleIndicesAttr([1])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
    primvars = UsdGeom.PrimvarsAPI(mesh)
    uv = primvars.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying")
    uv.Set([(0, 0), (1, 0), (1, 1), (0, 1)])
    uv.SetIndices([0, 1, 2, 3, 0, 1, 2])
    primvars.CreatePrimvar("vertex_id", Sdf.ValueTypeNames.FloatArray, "vertex").Set(list(range(8)))
    primvars.CreatePrimvar("face_id", Sdf.ValueTypeNames.IntArray, "uniform").Set([10, 20])
    paired = primvars.CreatePrimvar("paired", Sdf.ValueTypeNames.FloatArray, "vertex", elementSize=2)
    paired.Set(list(range(16)))
    primvars.CreatePrimvar("label", Sdf.ValueTypeNames.String, "constant").Set("unchanged")
    for name, face in (("Wood", 0), ("Metal", 1)):
        material = UsdShade.Material.Define(stage, f"/Asset/Looks/{name}")
        subset = UsdGeom.Subset.CreateGeomSubset(mesh, name, "face", [face], "materialBind", "partition")
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()
    return path


def test_component_bounds_use_asset_frame_and_ignore_unused_points(mesh_source):
    stage = open_mesh_partition_stage(str(mesh_source))
    report = inspect_mesh_components(stage, "Body/Mesh")

    assert [part["face_count"] for part in report["components"]] == [1, 1]
    assert report["components"][0]["min_root_m"] == pytest.approx([2, 3, 0.5])
    assert report["components"][0]["max_root_m"] == pytest.approx([4, 4, 0.5])
    assert report["components"][1]["material_subsets"] == ["Metal"]
    assert report["collision"]


def test_split_preserves_geometry_uvs_normals_materials_and_rigid_owner(mesh_source, tmp_path):
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    source_bytes = mesh_source.read_bytes()
    stage = open_mesh_partition_stage(str(mesh_source))
    report = partition_mesh(stage, "Body/Mesh", {"Tabletop": [0]}, remainder="Structure")
    output = tmp_path / "split.usda"
    stage.GetRootLayer().Export(str(output))
    stage = Usd.Stage.Open(str(output))

    assert mesh_source.read_bytes() == source_bytes
    container = stage.GetPrimAtPath("/Asset/Body/Mesh")
    assert container.IsA(UsdGeom.Xform)
    assert not container.HasAPI(UsdPhysics.CollisionAPI)
    assert stage.GetPrimAtPath("/Asset/Body").HasAPI(UsdPhysics.RigidBodyAPI)
    assert [p["face_count"] for p in report["parts"]] == [1, 1]
    assert report["parts"][0]["max_root_m"] == pytest.approx([4, 4, 0.5])
    for name, vertex_ids, corner_count, material_name, face_id in (
        ("Tabletop", list(range(4)), 4, "Wood", 10),
        ("Structure", list(range(4, 7)), 3, "Metal", 20),
    ):
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f"/Asset/Body/Mesh/{name}"))
        assert len(mesh.GetPointsAttr().Get()) == len(vertex_ids)
        assert list(mesh.GetFaceVertexCountsAttr().Get()) == [corner_count]
        assert mesh.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
        assert not mesh.GetPrim().HasAPI(UsdPhysics.RigidBodyAPI)
        assert UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Get() == "none"
        primvars = UsdGeom.PrimvarsAPI(mesh)
        assert list(primvars.GetPrimvar("vertex_id").Get()) == vertex_ids
        assert list(primvars.GetPrimvar("face_id").Get()) == [face_id]
        assert list(primvars.GetPrimvar("paired").Get()) == list(range(2 * vertex_ids[0], 2 * (vertex_ids[-1] + 1)))
        assert primvars.GetPrimvar("label").Get() == "unchanged"
        assert len(primvars.GetPrimvar("st").GetIndices()) == corner_count
        assert primvars.GetPrimvar("st").IsIndexed()
        np.testing.assert_array_equal(
            primvars.GetPrimvar("st").ComputeFlattened(), [(0, 0), (1, 0), (1, 1), (0, 1)][:corner_count]
        )
        assert len(mesh.GetNormalsAttr().Get()) == corner_count
        assert mesh.GetNormalsInterpolation() == "faceVarying"
        assert list(mesh.GetHoleIndicesAttr().Get()) == ([0] if name == "Structure" else [])
        subsets = UsdGeom.Subset.GetAllGeomSubsets(mesh)
        assert len(subsets) == 1
        assert list(subsets[0].GetIndicesAttr().Get()) == [0]
        material, _ = UsdShade.MaterialBindingAPI(subsets[0]).ComputeBoundMaterial()
        assert str(material.GetPath()) == f"/Asset/Looks/{material_name}"


def test_split_edits_instance_in_wrapper_only(mesh_source, tmp_path):
    from pxr import Usd, UsdGeom

    path = tmp_path / "instance.usda"
    source = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(source, "/Root").GetPrim()
    source.SetDefaultPrim(root)
    instance = source.DefinePrim("/Root/Instance")
    instance.GetReferences().AddReference(str(mesh_source))
    instance.SetInstanceable(True)
    source.GetRootLayer().Save()
    before = path.read_bytes()
    stage = open_mesh_partition_stage(str(path))
    assert stage.GetPrimAtPath("/Root/Instance/Body/Mesh").IsInstanceProxy()

    partition_mesh(stage, "Instance/Body/Mesh", {"Tabletop": [0]})

    assert not stage.GetPrimAtPath("/Root/Instance").IsInstance()
    assert stage.GetPrimAtPath("/Root/Instance/Body/Mesh/Tabletop").IsA(UsdGeom.Mesh)
    assert path.read_bytes() == before


@pytest.mark.parametrize("groups", [{"Top": [0], "Other": [0]}, {"Top": [2]}, {"Top": []}, {"bad/name": [0]}])
def test_split_rejects_invalid_component_groups(mesh_source, groups):
    stage = open_mesh_partition_stage(str(mesh_source))
    with pytest.raises(AssertionError):
        partition_mesh(stage, "Body/Mesh", groups)


def test_split_rejects_animated_geometry(mesh_source):
    from pxr import UsdGeom

    stage = open_mesh_partition_stage(str(mesh_source))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Asset/Body/Mesh"))
    mesh.GetPointsAttr().Set(mesh.GetPointsAttr().Get(), 1.0)
    with pytest.raises(AssertionError, match="Animated mesh"):
        partition_mesh(stage, "Body/Mesh", {"Top": [0]})

# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate Blender exports independently of an Isaac Sim process."""

from __future__ import annotations

import json
import math
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from .fits import collision_features, validate_assembly_fits
from .uv import UV_DOUBLED_AREA_TOLERANCE, surface_area_tolerance


def _texture_uv_names(material: UsdShade.Material) -> set[str]:
    """Resolve the primvar readers used by an exported image-texture material."""
    textured = False
    names = set()
    for prim in Usd.PrimRange(material.GetPrim()):
        shader = UsdShade.Shader(prim)
        if not shader or shader.GetIdAttr().Get() != "UsdUVTexture":
            continue
        textured = True
        coordinates = shader.GetInput("st")
        assert coordinates and coordinates.HasConnectedSource(), f"Image texture lacks UV input: {prim.GetPath()}"
        source, _, _ = coordinates.GetConnectedSource()
        reader = UsdShade.Shader(source.GetPrim())
        assert (
            reader and reader.GetIdAttr().Get() == "UsdPrimvarReader_float2"
        ), f"Image texture requires a direct UV primvar reader: {prim.GetPath()}"
        varname = reader.GetInput("varname")
        name = varname.Get()
        if varname.HasConnectedSource():
            source, attribute, _ = varname.GetConnectedSource()
            name = source.GetInput(attribute).Get()
        assert isinstance(name, str) and name, f"Unresolved texture UV reader: {prim.GetPath()}"
        names.add(name)
    assert not textured or names, f"Image material lacks a UV reader: {material.GetPath()}"
    return names if textured else set()


def _validate_texture_coordinates(mesh: UsdGeom.Mesh) -> tuple[bool, int, float]:
    """Validate textured faces and return texture presence, excluded count, and excluded area."""
    material, _ = UsdShade.MaterialBindingAPI(mesh.GetPrim()).ComputeBoundMaterial()
    assert material, f"Visual mesh lacks a material: {mesh.GetPath()}"
    names = _texture_uv_names(material)
    points = mesh.GetPointsAttr().Get()
    area_tolerance = surface_area_tolerance(points)
    excluded_faces = {}
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    for name in names:
        primvar = UsdGeom.PrimvarsAPI(mesh).FindPrimvarWithInheritance(name)
        assert primvar, f"Textured mesh lacks {name} UVs: {mesh.GetPath()}"
        values = primvar.ComputeFlattened()
        assert values, f"Empty {name} UVs: {mesh.GetPath()}"
        for value in values:
            assert len(value) == 2 and all(
                math.isfinite(component) for component in value
            ), f"Invalid {name} UV coordinate: {mesh.GetPath()}"
        interpolation = primvar.GetInterpolation()
        if interpolation == UsdGeom.Tokens.faceVarying:
            assert len(values) == len(indices), f"UV corner count mismatch: {mesh.GetPath()}"
            corners = values
        else:
            assert interpolation in (
                UsdGeom.Tokens.vertex,
                UsdGeom.Tokens.varying,
            ), f"Unsupported UV interpolation {interpolation}: {mesh.GetPath()}"
            assert len(values) == len(points), f"UV vertex count mismatch: {mesh.GetPath()}"
            corners = [values[index] for index in indices]
        offset = 0
        for face_index, count in enumerate(counts):
            anchor = corners[offset]
            point = points[indices[offset]]
            has_uv_area = False
            surface_area = 0.0
            for index in range(1, count - 1):
                a = corners[offset + index] - anchor
                b = corners[offset + index + 1] - anchor
                has_uv_area |= abs(a[0] * b[1] - a[1] * b[0]) > UV_DOUBLED_AREA_TOLERANCE
                edge_a = points[indices[offset + index]] - point
                edge_b = points[indices[offset + index + 1]] - point
                surface_area += Gf.Cross(edge_a, edge_b).GetLength() / 2
            if not has_uv_area and surface_area <= area_tolerance:
                excluded_faces[face_index] = surface_area
            assert (
                has_uv_area or surface_area <= area_tolerance
            ), f"Collapsed {name} UVs on face {face_index}: {mesh.GetPath()}"
            offset += count
    return bool(names), len(excluded_faces), sum(excluded_faces.values())


def validate_exports(output_dir: str | Path) -> dict:
    """Verify geometry, collision, texture and instrument-state export contracts.

    Args:
        output_dir: Asset bundle containing manifest.json and exported USD files.

    Returns:
        Counts and per-asset findings, also written to validation_report.json.
    """
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    report = {"assets": {}, "texture_files": set(), "asset_count": len(manifest["assets"])}
    features = {}
    for name, entry in manifest["assets"].items():
        stage = Usd.Stage.Open(str(output_dir / entry["file"]))
        assert stage is not None, f"Cannot open {name}"
        assert stage.GetDefaultPrim().GetPath() == entry["root_prim"], f"Default prim mismatch: {name}"
        assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z, f"Wrong up axis: {name}"
        assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0, f"Wrong units: {name}"
        features[name] = collision_features(stage)
        meshes = 0
        collision_count = 0
        face_count = 0
        textured_meshes = 0
        uv_degenerate_faces = 0
        uv_degenerate_area = 0.0
        for prim in stage.Traverse():
            assert not prim.HasAPI(UsdPhysics.RigidBodyAPI), f"Runtime owns rigid bodies: {prim.GetPath()}"
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_count += 1
                assert UsdGeom.Imageable(prim).GetVisibilityAttr().Get() == UsdGeom.Tokens.invisible
            if prim.IsA(UsdGeom.Mesh):
                meshes += 1
                mesh = UsdGeom.Mesh(prim)
                points = mesh.GetPointsAttr().Get()
                counts = mesh.GetFaceVertexCountsAttr().Get()
                indices = mesh.GetFaceVertexIndicesAttr().Get()
                assert len(points) > 0 and len(counts) > 0, f"Empty mesh: {prim.GetPath()}"
                assert len(indices) == sum(counts), f"Invalid mesh topology: {prim.GetPath()}"
                for point in points:
                    assert all(math.isfinite(value) for value in point), f"Non-finite vertex: {prim.GetPath()}"
                face_count += len(counts)
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    textured, excluded_faces, excluded_area = _validate_texture_coordinates(mesh)
                    textured_meshes += textured
                    uv_degenerate_faces += excluded_faces
                    uv_degenerate_area += excluded_area
            for attribute in prim.GetAttributes():
                if attribute.GetTypeName() != Sdf.ValueTypeNames.Asset:
                    continue
                value = attribute.Get()
                if value and value.path:
                    texture_path = Path(value.resolvedPath or output_dir / value.path)
                    assert texture_path.is_file(), f"Missing texture: {texture_path}"
                    report["texture_files"].add(str(texture_path.relative_to(output_dir)))
        if name != "lighting":
            assert meshes > 0, f"No visual geometry: {name}"
            assert collision_count == entry["collision_count"] > 0, f"Collision export mismatch: {name}"
            assert entry["mass_kg"] > 0, f"Invalid mass: {name}"
        for status, path in entry["affordances"].get("status_paths", {}).items():
            prim = stage.GetPrimAtPath(path)
            assert prim.IsValid(), f"Missing instrument status mesh: {name}/{status}"
            visibility = UsdGeom.Imageable(prim).ComputeVisibility()
            assert (visibility != UsdGeom.Tokens.invisible) == (status == "idle"), f"Wrong status visibility: {name}"
        for reading, path in entry["affordances"].get("reading_paths", {}).items():
            prim = stage.GetPrimAtPath(path)
            assert prim.IsValid(), f"Missing numeric reading mesh: {name}/{reading}"
            assert UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible
        report["assets"][name] = {
            "mesh_count": meshes,
            "face_count": face_count,
            "collision_count": collision_count,
            "textured_mesh_count": textured_meshes,
            "uv_degenerate_face_count": uv_degenerate_faces,
            "uv_degenerate_area_m2": uv_degenerate_area,
        }
    report["assembly_fits"] = validate_assembly_fits(manifest["assets"], features)
    report["texture_files"] = sorted(report["texture_files"])
    report["texture_count"] = len(report["texture_files"])
    (output_dir / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report

# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Partition disconnected USD mesh components into named collider subtrees."""

from __future__ import annotations

import hashlib
import numpy as np
from pathlib import Path
from typing import Any


def _mesh(stage, prim_path: str):
    from pxr import Sdf, UsdGeom

    root = stage.GetDefaultPrim()
    assert root, "Mesh partitioning requires a default prim."
    path = Sdf.Path(prim_path)
    if not path.IsAbsolutePath():
        path = root.GetPath().AppendPath(path)
    assert path.IsPrimPath() and path.HasPrefix(root.GetPath()), "Mesh must be beneath the default prim."
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(path))
    assert mesh, f"Expected a Mesh prim at {path}."
    return mesh


def _topology(mesh):
    points_M = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    assert points_M.ndim == 2 and points_M.shape[1] == 3 and np.isfinite(points_M).all(), "Invalid mesh points."
    assert counts.ndim == 1 and len(counts) > 0 and (counts >= 3).all(), "Expected polygon faces."
    assert indices.ndim == 1 and counts.sum() == len(indices), "Invalid face topology."
    assert (indices >= 0).all() and (indices < len(points_M)).all(), "Face index outside the points array."
    return points_M, counts, indices


def _component_labels(point_count: int, counts: np.ndarray, indices: np.ndarray):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    ends = np.cumsum(counts)
    starts = np.concatenate(([0], ends[:-1]))
    following = np.roll(indices, -1)
    following[ends - 1] = indices[starts]
    adjacency = coo_matrix(
        (np.ones(len(indices), dtype=bool), (indices, following)), shape=(point_count, point_count)
    ).tocsr()
    _, point_labels = connected_components(adjacency, directed=False)
    face_labels = point_labels[indices[starts]]
    # Number components by first face, independently of sparse-graph traversal order.
    _, first_faces = np.unique(face_labels, return_index=True)
    labels_in_order = face_labels[np.sort(first_faces)]
    component_ids = np.full(point_labels.max() + 1, -1, dtype=np.int64)
    component_ids[labels_in_order] = np.arange(len(labels_in_order))
    return component_ids[point_labels], component_ids[face_labels]


def _points_in_root_m(mesh, points_M: np.ndarray) -> np.ndarray:
    from pxr import UsdGeom

    stage = mesh.GetPrim().GetStage()
    cache = UsdGeom.XformCache()
    # USD matrices multiply row vectors; transpose to active target-source notation.
    T_R_M = np.asarray(
        cache.GetLocalToWorldTransform(mesh.GetPrim())
        * cache.GetLocalToWorldTransform(stage.GetDefaultPrim()).GetInverse()
    ).T
    return (points_M @ T_R_M[:3, :3].T + T_R_M[:3, 3]) * UsdGeom.GetStageMetersPerUnit(stage)


def inspect_mesh_components(stage, prim_path: str) -> dict[str, Any]:
    """Report connected components, material subsets, and bounds in asset-root meters.

    Args:
        stage: Open USD stage, including desired variant selections.
        prim_path: Absolute or default-prim-relative mesh path.

    Returns:
        JSON-serializable report with stable IDs ordered by first source face.
    """
    from pxr import UsdGeom, UsdPhysics

    mesh = _mesh(stage, prim_path)
    points_M, counts, indices = _topology(mesh)
    point_labels, face_labels = _component_labels(len(points_M), counts, indices)
    points_R = _points_in_root_m(mesh, points_M)
    component_count = int(face_labels.max()) + 1
    minimum_R = np.full((component_count, 3), np.inf)
    maximum_R = np.full((component_count, 3), -np.inf)
    used = point_labels >= 0
    np.minimum.at(minimum_R, point_labels[used], points_R[used])
    np.maximum.at(maximum_R, point_labels[used], points_R[used])
    face_counts = np.bincount(face_labels)
    subset_names = [[] for _ in range(component_count)]
    for subset in UsdGeom.Subset.GetAllGeomSubsets(mesh):
        if subset.GetElementTypeAttr().Get() != UsdGeom.Tokens.face:
            continue
        faces = np.asarray(subset.GetIndicesAttr().Get(), dtype=np.int64)
        for component_id in np.unique(face_labels[faces]):
            subset_names[component_id].append(subset.GetPrim().GetName())
    components = []
    for component_id in range(component_count):
        components.append({
            "id": component_id,
            "face_count": int(face_counts[component_id]),
            "min_root_m": minimum_R[component_id].tolist(),
            "max_root_m": maximum_R[component_id].tolist(),
            "material_subsets": sorted(subset_names[component_id]),
        })
    digest = hashlib.sha256(points_M.tobytes() + counts.tobytes() + indices.tobytes()).hexdigest()
    return {
        "mesh": str(mesh.GetPath()),
        "geometry_sha256": digest,
        "face_count": len(counts),
        "point_count": len(points_M),
        "collision": mesh.GetPrim().HasAPI(UsdPhysics.CollisionAPI),
        "components": components,
    }


def _selected_values(value, selection: np.ndarray, element_size: int = 1):
    array = np.asarray(value)
    assert len(array) % element_size == 0, "Primvar length does not match elementSize."
    grouped = array.reshape((-1, element_size) + array.shape[1:])
    assert len(selection) == 0 or selection.max() < len(grouped), "Attribute does not match mesh topology."
    selected = grouped[selection].reshape((-1,) + array.shape[1:])
    return type(value)(selected.tolist())


def _select_interpolation(interpolation, faces, corners, points):
    if interpolation == "uniform":
        return faces
    if interpolation == "faceVarying":
        return corners
    if interpolation in ("vertex", "varying"):
        return points
    assert interpolation == "constant", f"Unsupported interpolation: {interpolation}."
    return None


def _validate_static_mesh(mesh) -> None:
    from pxr import UsdGeom

    prim = mesh.GetPrim()
    assert not any(schema.startswith("Skel") for schema in prim.GetAppliedSchemas()), "Skinned meshes are unsupported."
    for attr in prim.GetAttributes():
        if attr.GetName().startswith("xformOp:") or attr.GetName() == "xformOpOrder":
            continue
        assert not attr.GetTimeSamples(), f"Animated mesh attribute is unsupported: {attr.GetPath()}."
    # Subdivision tags need topology-specific remapping; do not silently corrupt them.
    for name in ("creaseIndices", "cornerIndices"):
        value = prim.GetAttribute(name).Get()
        assert value is None or len(value) == 0, f"Meshes with {name} are unsupported."
    for child in prim.GetChildren():
        assert child.IsA(UsdGeom.Subset), f"Unsupported mesh child: {child.GetPath()}."
        subset = UsdGeom.Subset(child)
        assert subset.GetElementTypeAttr().Get() in ("face", "point"), "Only face and point subsets are supported."
        assert not subset.GetIndicesAttr().GetTimeSamples(), "Animated subsets are unsupported."


def _write_piece(stage, snapshot, source_path, destination_path, faces, points_M, counts, indices):
    from pxr import Sdf, UsdGeom, Vt

    stage.OverridePrim(destination_path)
    assert Sdf.CopySpec(snapshot, source_path, stage.GetEditTarget().GetLayer(), destination_path)
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(destination_path))
    prim = mesh.GetPrim()
    # The original prim becomes the transform/rigid-body owner of these pieces.
    for attr in prim.GetAttributes():
        if attr.GetName().startswith("xformOp:") or attr.GetName() == "xformOpOrder":
            prim.RemoveProperty(attr.GetName())
    for schema in prim.GetAppliedSchemas():
        if "RigidBody" in schema or "Articulation" in schema or schema == "PhysicsMassAPI":
            prim.RemoveAppliedSchema(schema)

    face_mask = np.zeros(len(counts), dtype=bool)
    face_mask[faces] = True
    corners = np.flatnonzero(np.repeat(face_mask, counts))
    point_ids, piece_indices = np.unique(indices[corners], return_inverse=True)
    piece_points = points_M[point_ids]
    mesh.GetPointsAttr().Set(Vt.Vec3fArray(piece_points.tolist()))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(counts[faces].tolist()))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(piece_indices.tolist()))
    mesh.CreateExtentAttr().Set(Vt.Vec3fArray([piece_points.min(0).tolist(), piece_points.max(0).tolist()]))
    for name in ("velocities", "accelerations"):
        attr = prim.GetAttribute(name)
        value = attr.Get()
        if value is not None and len(value):
            attr.Set(_selected_values(value, point_ids))
    normals = mesh.GetNormalsAttr().Get()
    if normals is not None and len(normals):
        selection = _select_interpolation(mesh.GetNormalsInterpolation(), faces, corners, point_ids)
        if selection is not None:
            mesh.GetNormalsAttr().Set(_selected_values(normals, selection))
    for primvar in UsdGeom.PrimvarsAPI(prim).GetPrimvars():
        value = primvar.Get()
        if value is None:
            continue
        selection = _select_interpolation(primvar.GetInterpolation(), faces, corners, point_ids)
        if selection is None:
            continue
        if primvar.IsIndexed():
            selected_indices = np.asarray(primvar.GetIndices())[selection]
            assert (selected_indices >= 0).all(), f"Invalid primvar indices: {primvar.GetName()}."
            used_values, new_indices = np.unique(selected_indices, return_inverse=True)
            unauthored_index = primvar.GetUnauthoredValuesIndex()
            primvar.Set(_selected_values(value, used_values, primvar.GetElementSize()))
            primvar.SetIndices(Vt.IntArray(new_indices.tolist()))
            if unauthored_index >= 0:
                matching = np.flatnonzero(used_values == unauthored_index)
                primvar.SetUnauthoredValuesIndex(int(matching[0]) if len(matching) else -1)
        else:
            primvar.Set(_selected_values(value, selection, primvar.GetElementSize()))

    face_remap = np.full(len(counts), -1, dtype=np.int64)
    face_remap[faces] = np.arange(len(faces))
    point_remap = np.full(len(points_M), -1, dtype=np.int64)
    point_remap[point_ids] = np.arange(len(point_ids))
    holes = np.asarray(mesh.GetHoleIndicesAttr().Get(), dtype=np.int64)
    if len(holes):
        mapped = face_remap[holes]
        mesh.GetHoleIndicesAttr().Set(Vt.IntArray(mapped[mapped >= 0].tolist()))
    for subset in UsdGeom.Subset.GetAllGeomSubsets(mesh):
        remap = face_remap if subset.GetElementTypeAttr().Get() == "face" else point_remap
        mapped = remap[np.asarray(subset.GetIndicesAttr().Get(), dtype=np.int64)]
        mapped = mapped[mapped >= 0]
        if len(mapped):
            subset.GetIndicesAttr().Set(Vt.IntArray(mapped.tolist()))
        else:
            stage.RemovePrim(subset.GetPath())
    return mesh


def partition_mesh(stage, prim_path: str, groups: dict[str, list[int]], remainder: str = "Remaining") -> dict[str, Any]:
    """Replace one Mesh with an Xform containing named meshes of complete connected components.

    Args:
        stage: Writable derived USD stage; source layers are never edited.
        prim_path: Absolute or default-prim-relative mesh path.
        groups: Child prim names mapped to IDs from inspect_mesh_components().
        remainder: Child name receiving every unselected component.

    Returns:
        Report of output prims and face counts. Every input face occurs exactly once.
    """
    from pxr import Tf

    mesh = _mesh(stage, prim_path)
    # A referenced instance must become editable before adding child mesh prims.
    ancestors = []
    ancestor = mesh.GetPrim()
    while ancestor and not ancestor.IsPseudoRoot():
        ancestors.append(ancestor.GetPath())
        ancestor = ancestor.GetParent()
    for path in reversed(ancestors):
        prim = stage.GetPrimAtPath(path)
        if prim.IsInstanceable():
            prim.SetInstanceable(False)
    mesh = _mesh(stage, prim_path)
    _validate_static_mesh(mesh)
    points_M, counts, indices = _topology(mesh)
    _, face_labels = _component_labels(len(points_M), counts, indices)
    component_count = int(face_labels.max()) + 1
    assert isinstance(groups, dict) and groups, "At least one named component group is required."
    selections = {}
    selected_components = set()
    original_children = {child.GetName() for child in mesh.GetPrim().GetChildren()}
    for name, component_ids in groups.items():
        assert isinstance(name, str) and Tf.IsValidIdentifier(name), f"Invalid child prim name: {name!r}."
        assert name not in original_children, f"Child prim already exists: {name}."
        assert isinstance(component_ids, list) and component_ids, f"Empty component group: {name}."
        for component_id in component_ids:
            assert (
                type(component_id) is int and 0 <= component_id < component_count
            ), f"Invalid component ID: {component_id}."
            assert component_id not in selected_components, f"Component {component_id} selected more than once."
            selected_components.add(component_id)
        selections[name] = np.flatnonzero(np.isin(face_labels, component_ids))
    remaining_faces = np.flatnonzero(~np.isin(face_labels, list(selected_components)))
    if len(remaining_faces):
        assert isinstance(remainder, str) and Tf.IsValidIdentifier(remainder), "Invalid remainder prim name."
        assert remainder not in groups and remainder not in original_children, "Remainder prim name already used."
        selections[remainder] = remaining_faces

    source_path = mesh.GetPath()
    snapshot = stage.Flatten()
    parts = []
    for name, faces in selections.items():
        path = source_path.AppendChild(name)
        piece = _write_piece(stage, snapshot, source_path, path, faces, points_M, counts, indices)
        piece_R = _points_in_root_m(piece, np.asarray(piece.GetPointsAttr().Get()))
        parts.append({
            "path": str(path),
            "face_count": len(faces),
            "min_root_m": piece_R.min(0).tolist(),
            "max_root_m": piece_R.max(0).tolist(),
        })

    source = stage.GetPrimAtPath(source_path)
    source.SetTypeName("Xform")
    for schema in source.GetAppliedSchemas():
        if "Collision" in schema:
            source.RemoveAppliedSchema(schema)
    for name in ("points", "faceVertexCounts", "faceVertexIndices", "normals", "extent", "holeIndices"):
        attr = source.GetAttribute(name)
        if attr:
            attr.Block()
    for name in original_children:
        stage.OverridePrim(source_path.AppendChild(name)).SetActive(False)
    assert sum(part["face_count"] for part in parts) == len(counts), "Partition lost mesh faces."
    return {"source_mesh": str(source_path), "source_face_count": len(counts), "parts": parts}


def open_mesh_partition_stage(usd_path: str, variants: dict[str, str] | None = None):
    """Open an editable reference wrapper for a local or revision-pinned Central USD."""
    from pxr import Usd, UsdGeom

    from isaaclab_arena.assets.simready_central_asset_loader import download_central_asset_package, is_central_asset_url
    from isaaclab_arena.utils.usd.rigid_bodies import apply_usd_variant_selections

    if is_central_asset_url(usd_path):
        usd_path = download_central_asset_package(usd_path)
    source_path = Path(usd_path).expanduser().absolute()
    assert source_path.is_file(), f"USD source file does not exist: {source_path}."
    source = Usd.Stage.Open(str(source_path))
    assert source and source.GetDefaultPrim(), "USD source requires a default prim."
    stage = Usd.Stage.CreateInMemory()
    root = stage.DefinePrim(source.GetDefaultPrim().GetPath())
    root.GetReferences().AddReference(str(source_path))
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(source))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(source))
    apply_usd_variant_selections(stage, variants or {"Physics": "physics", "PhysX": "Enabled"})
    assert not stage.GetCompositionErrors(), "USD source has unresolved composition errors."
    return stage

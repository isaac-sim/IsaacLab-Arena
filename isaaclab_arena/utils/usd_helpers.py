# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import numpy as np
import trimesh
from collections.abc import Mapping, Sequence
from contextlib import contextmanager

from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.usd_articulation import (
    articulation_joint_prims,
    compute_posed_prim_world_deltas,
    resolve_joint_pos_patterns,
    resolve_prim_world_delta,
)

_POSED_GEOMETRY_CACHE_SIZE = 16
"""Distinct (USD, joint positions, scale) combinations to keep posed geometry for.

An environment poses a handful of articulations, so a small cache spares repeated USD opens
without pinning many multi-megabyte meshes.
"""


class NoCollisionMeshError(ValueError):
    """No extractable collision mesh exists at the requested USD location."""


class UnsupportedCollisionGeometryError(NoCollisionMeshError):
    """USD geometry exists but cannot be represented as a collision mesh."""


def get_all_prims(
    stage: Usd.Stage, prim: Usd.Prim | None = None, prims_list: list[Usd.Prim] | None = None
) -> list[Usd.Prim]:
    """Get all prims in the stage.

    Performs a Depth First Search (DFS) through the prims in a stage
    and returns all the prims.

    Args:
        stage: The stage to get the prims from.
        prim: The prim to start the search from. Defaults to the pseudo-root.
        prims_list: The list to store the prims in. Defaults to an empty list.

    Returns:
        A list of prims in the stage.
    """
    if prims_list is None:
        prims_list = []
    if prim is None:
        prim = stage.GetPseudoRoot()
    for child in prim.GetAllChildren():
        prims_list.append(child)
        get_all_prims(stage, child, prims_list)
    return prims_list


def has_light(stage: Usd.Stage) -> bool:
    """Check if the stage has a light"""
    LIGHT_TYPES = (
        UsdLux.SphereLight,
        UsdLux.RectLight,
        UsdLux.DomeLight,
        UsdLux.DistantLight,
        UsdLux.DiskLight,
    )
    has_light = False
    all_prims = get_all_prims(stage)
    for prim in all_prims:
        if any(prim.IsA(t) for t in LIGHT_TYPES):
            has_light = True
            break
    return has_light


def is_articulation_root(prim: Usd.Prim) -> bool:
    """Check if prim is articulation root"""
    return prim.HasAPI(UsdPhysics.ArticulationRootAPI)


def is_rigid_body(prim: Usd.Prim) -> bool:
    """Check if prim is rigidbody"""
    return prim.HasAPI(UsdPhysics.RigidBodyAPI)


def has_physics_or_collision(prim: Usd.Prim) -> bool:
    """Return True when prim participates in physics simulation or collision."""
    if is_articulation_root(prim) or is_rigid_body(prim):
        return True
    return prim.HasAPI(UsdPhysics.CollisionAPI)


def object_type_for_prim(prim: Usd.Prim) -> ObjectType:
    """Classify a prim for object-reference resolution."""
    if is_articulation_root(prim):
        return ObjectType.ARTICULATION
    if is_rigid_body(prim):
        return ObjectType.RIGID
    return ObjectType.BASE


def relative_path_from_default_prim(stage: Usd.Stage, prim_path: str) -> str:
    """Return the prim path suffix relative to the stage default prim."""
    default_prim = stage.GetDefaultPrim()
    assert default_prim, f"USD stage has no default prim: {stage.GetRootLayer().identifier}"
    default_prefix = str(default_prim.GetPath())
    if default_prefix == "/":
        return prim_path.lstrip("/")
    if prim_path == default_prefix:
        return ""
    prefix = default_prefix if default_prefix.endswith("/") else default_prefix + "/"
    if prim_path.startswith(prefix):
        return prim_path[len(prefix) :]
    return prim_path.lstrip("/")


def articulation_joint_names(articulation_prim: Usd.Prim) -> tuple[str, ...]:
    """Return sorted movable joint names under an articulation root."""
    joint_names: list[str] = []
    for desc in Usd.PrimRange(articulation_prim):
        if desc.IsA(UsdPhysics.RevoluteJoint) or desc.IsA(UsdPhysics.PrismaticJoint):
            joint_names.append(desc.GetName())
    return tuple(sorted(set(joint_names)))


def get_prim_depth(prim: Usd.Prim) -> int:
    """Get the depth of a prim"""
    return len(str(prim.GetPath()).split("/")) - 2


@contextmanager
def open_stage(path):
    """Open a stage and ensure it is closed after use."""
    stage = Usd.Stage.Open(path)
    try:
        yield stage
    finally:
        # Drop the local reference; Garbage Collection will reclaim once no prim/attr handles remain
        del stage


def get_asset_usd_path_from_prim_path(prim_path: str, stage: Usd.Stage) -> str | None:
    """Get the USD path from a prim path, that is referring to an asset."""
    # Note (xinjieyao, 2025.12.12): preferred way to get the composed asset path is to ask the Usd.Prim object itself,
    # which handles the entire composition stack. Here it achieved this goal thru root layer due to the USD API limitations.
    # It only finds references authored on the root layer.
    # If the asset was referenced in an intermediate sublayer, this method would fail to find the asset path.
    root_layer = stage.GetRootLayer()
    prim_spec = root_layer.GetPrimAtPath(prim_path)
    if not prim_spec:
        return None

    try:
        reference_list = prim_spec.referenceList.GetAddedOrExplicitItems()
    except Exception as e:
        print(f"Failed to get reference list for prim {prim_path}: {e}")
        return None
    if len(reference_list) > 0:
        for reference_spec in reference_list:
            if reference_spec.assetPath:
                return reference_spec.assetPath

    return None


def _read_default_prim_scale(prim: Usd.Prim) -> tuple[float, float, float]:
    """Return the default prim's root ``xformOp:scale``, or identity if absent."""
    if not prim.IsA(UsdGeom.Xformable):
        return (1.0, 1.0, 1.0)
    for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            value = op.Get()
            if value is not None:
                return (float(value[0]), float(value[1]), float(value[2]))
    return (1.0, 1.0, 1.0)


def compute_local_bounding_box_from_usd(
    usd_path: str,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    prim_path: str | None = None,
) -> AxisAlignedBoundingBox:
    """Compute the local bounding box matching Isaac Lab ``UsdFileCfg`` spawn size.

    Opening a USD directly includes the default prim's root ``xformOp:scale``
    in ``ComputeWorldBound``, but Isaac Lab's spawner ignores it and only
    Object.scale on the spawn wrapper applies.
    This helper unbakes the default prim's root scale from the USD, then
    applies ``Object.scale`` once so relation-solver bboxes match what is
    actually spawned.

    Args:
        usd_path: Path to the USD file.
        scale: Spawn-time scale passed to ``UsdFileCfg`` / ``Object.scale``.
        prim_path: Optional sub-prim to bound. When set, returns that prim's AABB
            expressed in the default prim's frame (root-relative). When None,
            bounds the default prim itself.

    Returns:
        AxisAlignedBoundingBox containing local min and max points.
    """
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise ValueError(f"Failed to open USD file: {usd_path}")

    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        default_prim = stage.GetPseudoRoot()

    if prim_path is None:
        bbox = compute_local_bounding_box_from_prim(stage, default_prim.GetPath().pathString)
    else:
        # Sub-prim bounds expressed in the default-prim (root) frame.
        sub_prim = stage.GetPrimAtPath(prim_path)
        assert sub_prim, f"No prim found at path {prim_path}"
        bbox = compute_local_bounding_box_from_prim(stage, prim_path)
        time_code = Usd.TimeCode.Default()
        sub_world = UsdGeom.Xformable(sub_prim).ComputeLocalToWorldTransform(time_code).ExtractTranslation()
        root_world = UsdGeom.Xformable(default_prim).ComputeLocalToWorldTransform(time_code).ExtractTranslation()
        bbox = bbox.translated((
            float(sub_world[0] - root_world[0]),
            float(sub_world[1] - root_world[1]),
            float(sub_world[2] - root_world[2]),
        ))

    usd_scale = _read_default_prim_scale(default_prim)
    assert not any(
        s == 0.0 for s in usd_scale
    ), f"Default prim {default_prim.GetPath().pathString} has scale {usd_scale}"
    composed_scale = (scale[0] / usd_scale[0], scale[1] / usd_scale[1], scale[2] / usd_scale[2])
    bbox = bbox.scaled(composed_scale)
    return bbox


def compute_local_bounding_box_from_prim(
    stage: Usd.Stage,
    prim_path: str,
) -> AxisAlignedBoundingBox:
    """Compute the local bounding box of a specific prim (relative to prim's transform origin).

    Args:
        stage: The USD stage containing the prim.
        prim_path: Path to the prim to compute the bounding box for.

    Returns:
        AxisAlignedBoundingBox containing the local min and max points relative to the
        prim's own origin.

    Raises:
        ValueError: If the prim is not found at the given path.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"No prim found at path {prim_path}")

    # Compute the world-space bounding box of the prim
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(prim)
    bbox_range = bbox.ComputeAlignedBox()

    # Get world-space min/max
    world_min = bbox_range.GetMin()
    world_max = bbox_range.GetMax()

    # Get the target prim's world position to compute local bounding box
    prim_xformable = UsdGeom.Xformable(prim)
    prim_world_transform = prim_xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    prim_world_pos = prim_world_transform.ExtractTranslation()

    # Compute local bounding box by subtracting the prim's own world position
    local_min = Gf.Vec3d(
        world_min[0] - prim_world_pos[0],
        world_min[1] - prim_world_pos[1],
        world_min[2] - prim_world_pos[2],
    )
    local_max = Gf.Vec3d(
        world_max[0] - prim_world_pos[0],
        world_max[1] - prim_world_pos[1],
        world_max[2] - prim_world_pos[2],
    )

    return AxisAlignedBoundingBox(
        min_point=(local_min[0], local_min[1], local_min[2]),
        max_point=(local_max[0], local_max[1], local_max[2]),
    )


def extract_trimesh_from_usd(
    usd_path: str,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    excluded_prim_paths: Sequence[str] = (),
) -> trimesh.Trimesh | None:
    """Extract all UsdGeom.Mesh prims from a USD into a single trimesh.

    Scale is applied per-vertex in local frame before the prim-to-world transform.
    All scale components must be positive (negative flips winding/SDF sign).
    Other Gprim geometry is rejected, not silently dropped.

    Args:
        usd_path: Path to the .usd/.usda/.usdc file.
        scale: (sx, sy, sz) per-axis scale factors applied in local frame.
        excluded_prim_paths: Absolute USD prim paths whose complete subtrees are omitted.

    Returns:
        Combined trimesh with per-prim world transforms baked in, or ``None`` when exclusions
        remove every mesh.
    """
    assert all(
        s > 0 for s in scale
    ), f"All scale components must be positive (negative scale flips winding/SDF sign), got {scale}"
    excluded_paths = tuple(path.rstrip("/") for path in excluded_prim_paths)
    assert all(
        path.startswith("/") for path in excluded_paths
    ), f"excluded_prim_paths must be absolute USD paths, got {excluded_paths}"

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise ValueError(f"Failed to open USD: {usd_path}")

    all_verts: list[np.ndarray] = []
    all_faces: list[list[int]] = []
    skipped_gprims: list[str] = []
    excluded_mesh_count = 0
    included_mesh_count = 0
    offset = 0

    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if any(prim_path == path or prim_path.startswith(f"{path}/") for path in excluded_paths):
            excluded_mesh_count += int(prim.IsA(UsdGeom.Mesh))
            continue
        if not prim.IsA(UsdGeom.Mesh):
            if prim.IsA(UsdGeom.Gprim):
                skipped_gprims.append(str(prim.GetPath()))
            continue
        included_mesh_count += 1
        mesh_prim = UsdGeom.Mesh(prim)
        points = mesh_prim.GetPointsAttr().Get()
        face_vertex_counts = mesh_prim.GetFaceVertexCountsAttr().Get()
        face_vertex_indices = mesh_prim.GetFaceVertexIndicesAttr().Get()
        if points is None or face_vertex_counts is None or face_vertex_indices is None:
            continue

        xform = UsdGeom.Xformable(prim)
        world_tf = np.array(xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default()))

        verts = np.asarray(points, dtype=np.float64)
        verts_scaled = verts * np.array(scale, dtype=np.float64)
        verts_h = np.hstack([verts_scaled, np.ones((len(verts_scaled), 1))])
        verts_world = (verts_h @ world_tf)[:, :3]

        # Fan-triangulate faces
        idx = 0
        for count in face_vertex_counts:
            for k in range(1, count - 1):
                all_faces.append([
                    face_vertex_indices[idx] + offset,
                    face_vertex_indices[idx + k] + offset,
                    face_vertex_indices[idx + k + 1] + offset,
                ])
            idx += count

        all_verts.append(verts_world)
        offset += len(verts_world)

    if all_verts:
        if skipped_gprims:
            print(f"Unsupported non-mesh geometry in {usd_path}: {', '.join(skipped_gprims)}")
        return trimesh.Trimesh(vertices=np.vstack(all_verts), faces=np.array(all_faces, dtype=np.int32))
    if skipped_gprims:
        raise UnsupportedCollisionGeometryError(
            f"Unsupported non-mesh geometry in {usd_path}: {', '.join(skipped_gprims)}"
        )
    if excluded_mesh_count and not included_mesh_count:
        return None
    raise NoCollisionMeshError(f"No mesh geometry found in {usd_path}")


def extract_trimesh_from_prim(
    stage: Usd.Stage,
    prim_path: str,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    prim_world_deltas: Mapping[str, np.ndarray] | None = None,
) -> trimesh.Trimesh:
    """Extract UsdGeom.Mesh geometry under a prim into the prim's local frame.

    Other Gprim geometry is rejected, not silently dropped.

    Args:
        stage: Stage containing the prim.
        prim_path: Root prim to gather mesh geometry under.
        scale: Per-axis scale applied in the root frame.
        prim_world_deltas: Optional world-space transform per prim path, applied before the root
            frame conversion so callers can relocate geometry (e.g. posing an articulation). A mesh
            inherits the delta of its nearest ancestor present in the mapping.
    """
    assert all(
        s > 0 for s in scale
    ), f"All scale components must be positive (negative scale flips winding/SDF sign), got {scale}"

    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim:
        raise ValueError(f"No prim found at path {prim_path}")
    if not root_prim.IsA(UsdGeom.Xformable):
        raise ValueError(f"Prim at path {prim_path} is not Xformable")

    root_world_tf = np.array(UsdGeom.Xformable(root_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    root_world_tf_inv = np.linalg.inv(root_world_tf)
    scale_np = np.asarray(scale, dtype=np.float64)

    all_verts: list[np.ndarray] = []
    all_faces: list[list[int]] = []
    skipped_gprims: list[str] = []
    offset = 0

    # Instance proxies must be traversed explicitly, or an instanceable stand contributes no vertices.
    for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            if prim.IsA(UsdGeom.Gprim):
                skipped_gprims.append(str(prim.GetPath()))
            continue
        mesh_prim = UsdGeom.Mesh(prim)
        points = mesh_prim.GetPointsAttr().Get()
        face_vertex_counts = mesh_prim.GetFaceVertexCountsAttr().Get()
        face_vertex_indices = mesh_prim.GetFaceVertexIndicesAttr().Get()
        if points is None or face_vertex_counts is None or face_vertex_indices is None:
            continue

        prim_world_tf = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        if prim_world_deltas is not None:
            delta = resolve_prim_world_delta(str(prim.GetPath()), prim_world_deltas)
            if delta is not None:
                prim_world_tf = prim_world_tf @ delta
        prim_to_root_tf = prim_world_tf @ root_world_tf_inv
        verts = np.asarray(points, dtype=np.float64)
        verts_h = np.hstack([verts, np.ones((len(verts), 1))])
        verts_root = (verts_h @ prim_to_root_tf)[:, :3] * scale_np

        idx = 0
        for count in face_vertex_counts:
            for k in range(1, count - 1):
                all_faces.append([
                    face_vertex_indices[idx] + offset,
                    face_vertex_indices[idx + k] + offset,
                    face_vertex_indices[idx + k + 1] + offset,
                ])
            idx += count

        all_verts.append(verts_root)
        offset += len(verts_root)

    if all_verts:
        if skipped_gprims:
            print(f"Unsupported non-mesh geometry under {prim_path}: {', '.join(skipped_gprims)}")
        return trimesh.Trimesh(vertices=np.vstack(all_verts), faces=np.array(all_faces, dtype=np.int32))
    if skipped_gprims:
        raise UnsupportedCollisionGeometryError(
            f"Unsupported non-mesh geometry under {prim_path}: {', '.join(skipped_gprims)}"
        )
    raise NoCollisionMeshError(f"No mesh geometry found under {prim_path}")


def extract_trimesh_from_usd_path(
    usd_path: str,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> trimesh.Trimesh:
    """Extract the mesh under a USD file's default prim into that prim's local frame.

    Scoping extraction to the default prim excludes sibling scene props (ground planes, stray
    objects) baked into some flattened articulation USDs.
    """
    stage = Usd.Stage.Open(usd_path)
    assert stage is not None, f"could not open USD: {usd_path}"
    default_prim = stage.GetDefaultPrim() or stage.GetPseudoRoot()
    return extract_trimesh_from_prim(stage, default_prim.GetPath().pathString, scale)


# -----------------------------------------------------------------------------
# Joint-posed articulation geometry helpers
# -----------------------------------------------------------------------------


def extract_trimesh_from_usd_at_joint_pos(
    usd_path: str,
    joint_pos: Mapping[str, float],
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> trimesh.Trimesh:
    """Return one mesh containing all of an articulation's posed link boxes."""
    mesh = _extract_trimesh_from_usd_at_joint_pos(usd_path, tuple(sorted(joint_pos.items())), tuple(scale))
    return mesh.copy()


# NOTE(zihaox, 2026-07-28): Cache here rather than on the asset. Isaac Lab reaches assets through
# EventTermCfg params, and configclass's validation walk tracks no visited set, so a trimesh held by
# an asset sends it recursing through trimesh's internal back-references until the stack overflows.
@functools.lru_cache(maxsize=_POSED_GEOMETRY_CACHE_SIZE)
def _extract_trimesh_from_usd_at_joint_pos(
    usd_path: str,
    joint_pos_items: tuple[tuple[str, float], ...],
    scale: tuple[float, float, float],
) -> trimesh.Trimesh:
    """Return a cached mesh containing all of an articulation's link boxes."""
    assert all(
        component > 0 for component in scale
    ), f"All scale components must be positive (negative scale flips winding/SDF sign), got {scale}"
    joint_pos = dict(joint_pos_items)
    stage = Usd.Stage.Open(usd_path)
    assert stage is not None, f"could not open USD: {usd_path}"
    default_prim = stage.GetDefaultPrim() or stage.GetPseudoRoot()
    root_path = default_prim.GetPath().pathString
    resolved = resolve_joint_pos_patterns(articulation_joint_prims(default_prim), joint_pos)
    deltas = compute_posed_prim_world_deltas(stage, root_path, resolved)
    return trimesh.util.concatenate(_posed_link_bbox_meshes(stage, default_prim, deltas, scale))


def _nearest_rigid_body_ancestor(prim: Usd.Prim, root_prim: Usd.Prim) -> Usd.Prim | None:
    """Return the nearest rigid-body ancestor at or below root_prim."""
    candidate = prim
    root_path = root_prim.GetPath()
    while candidate and candidate.IsValid() and candidate.GetPath().HasPrefix(root_path):
        if candidate.HasAPI(UsdPhysics.RigidBodyAPI):
            return candidate
        if candidate == root_prim:
            break
        candidate = candidate.GetParent()
    return None


def _untransformed_gprim_corners(
    prim: Usd.Prim,
    bbox_cache: UsdGeom.BBoxCache,
) -> np.ndarray | None:
    """Return a Gprim's local homogeneous bound corners, or None for an empty bound."""
    local_range = bbox_cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
    if local_range.IsEmpty():
        return None
    low, high = local_range.GetMin(), local_range.GetMax()
    return np.array(
        [[x, y, z, 1.0] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])],
        dtype=np.float64,
    )


def _posed_link_bbox_meshes(
    stage: Usd.Stage,
    default_prim: Usd.Prim,
    body_deltas: Mapping[str, np.ndarray],
    scale: tuple[float, float, float],
) -> tuple[trimesh.Trimesh, ...]:
    """Build and pose one local bounding box per rigid body or static root."""
    time = Usd.TimeCode.Default()
    bbox_cache = UsdGeom.BBoxCache(time, includedPurposes=[UsdGeom.Tokens.default_])
    root_world_tf = np.array(UsdGeom.Xformable(default_prim).ComputeLocalToWorldTransform(time))
    root_world_tf_inv = np.linalg.inv(root_world_tf)
    scale_np = np.asarray(scale, dtype=np.float64)

    frames: dict[str, Usd.Prim] = {}
    corners_by_frame: dict[str, list[np.ndarray]] = {}
    for prim in Usd.PrimRange(default_prim, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Gprim):
            continue
        local_corners = _untransformed_gprim_corners(prim, bbox_cache)
        if local_corners is None:
            continue
        body_prim = _nearest_rigid_body_ancestor(prim, default_prim)
        frame_prim = body_prim or default_prim
        frame_path = frame_prim.GetPath().pathString
        frames.setdefault(frame_path, frame_prim)
        frame_world_tf = np.array(UsdGeom.Xformable(frame_prim).ComputeLocalToWorldTransform(time))
        prim_world_tf = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time))
        corners_by_frame.setdefault(frame_path, []).append(
            local_corners @ (prim_world_tf @ np.linalg.inv(frame_world_tf))
        )

    meshes: list[trimesh.Trimesh] = []
    for frame_path, local_corners in corners_by_frame.items():
        stacked = np.vstack(local_corners)[:, :3]
        low, high = stacked.min(axis=0), stacked.max(axis=0)
        extents = high - low
        assert np.all(extents > 0.0), f"degenerate geometry under {frame_path}: extents={extents}"
        box = trimesh.creation.box(extents=extents)
        box.apply_translation((low + high) / 2.0)

        frame_prim = frames[frame_path]
        posed_frame_world_tf = np.array(UsdGeom.Xformable(frame_prim).ComputeLocalToWorldTransform(time))
        delta = body_deltas.get(frame_path)
        if delta is not None:
            posed_frame_world_tf = posed_frame_world_tf @ delta
        frame_to_root = posed_frame_world_tf @ root_world_tf_inv
        vertices_h = np.column_stack([box.vertices, np.ones(len(box.vertices))])
        vertices = (vertices_h @ frame_to_root)[:, :3] * scale_np
        meshes.append(trimesh.Trimesh(vertices=vertices, faces=box.faces, process=False))

    assert meshes, f"no bounded geometry found under {default_prim.GetPath()}"
    return tuple(meshes)


def compute_local_bounding_box_from_usd_at_joint_pos(
    usd_path: str,
    joint_pos: Mapping[str, float],
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    prim_path: str | None = None,
) -> AxisAlignedBoundingBox:
    """Compute posed bounds under a prim in the articulation's default-prim-local frame.

    Every ``UsdGeom.Gprim`` contributes, matching the geometry the unposed
    ``compute_local_bounding_box_from_usd`` covers. Bounds must not come from the posed mesh alone:
    mesh extraction drops analytic gprims (Droid's ``gripper_adapter``), which would understate the
    robot's footprint by centimetres.

    The result is cached on the arguments and copied before return. Relation losses ask for bounds
    every optimisation step, which would otherwise reopen the USD and redo the posing per step.

    Args:
        usd_path: Path to the articulation's .usd/.usda/.usdc file.
        joint_pos: Joint positions keyed by exact joint name or Isaac Lab regex, revolute in radians.
        scale: Spawn-time scale passed to ``UsdFileCfg``.
        prim_path: Optional sub-prim to bound. When None, bounds the full default prim.

    Returns:
        AxisAlignedBoundingBox containing the posed local bounds.
    """
    bbox = _compute_local_bounding_box_from_usd_at_joint_pos(
        usd_path, tuple(sorted(joint_pos.items())), tuple(scale), prim_path
    )
    return AxisAlignedBoundingBox(bbox.min_point.clone(), bbox.max_point.clone())


@functools.lru_cache(maxsize=_POSED_GEOMETRY_CACHE_SIZE)
def _compute_local_bounding_box_from_usd_at_joint_pos(
    usd_path: str,
    joint_pos_items: tuple[tuple[str, float], ...],
    scale: tuple[float, float, float],
    prim_path: str | None,
) -> AxisAlignedBoundingBox:
    """Cacheable body of ``compute_local_bounding_box_from_usd_at_joint_pos``, keyed by hashable args."""
    joint_pos = dict(joint_pos_items)
    stage = Usd.Stage.Open(usd_path)
    assert stage is not None, f"could not open USD: {usd_path}"
    default_prim = stage.GetDefaultPrim() or stage.GetPseudoRoot()
    root_path = default_prim.GetPath().pathString
    bound_prim = stage.GetPrimAtPath(prim_path) if prim_path is not None else default_prim
    assert bound_prim.IsValid(), f"prim not found: {prim_path} in {usd_path}"
    resolved = resolve_joint_pos_patterns(articulation_joint_prims(default_prim), joint_pos)
    deltas = compute_posed_prim_world_deltas(stage, root_path, resolved)

    # Expressing corners relative to the root cancels its own transform, including the root scale
    # Isaac Lab's spawner ignores, so only the caller's spawn scale applies.
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
    root_world_tf_inv = np.linalg.inv(
        np.array(UsdGeom.Xformable(default_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    )
    scale_np = np.asarray(scale, dtype=np.float64)

    corners: list[np.ndarray] = []
    # Instance proxies must be traversed explicitly: robot-on-stand assemblies mark the stand (and for
    # Franka, every visual) instanceable, so a default traversal stops at the instance and reports bounds
    # covering the arm alone.
    for prim in Usd.PrimRange(bound_prim, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Gprim):
            continue
        local_range = bbox_cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
        if local_range.IsEmpty():
            continue
        low, high = local_range.GetMin(), local_range.GetMax()
        box_corners = np.array(
            [[x, y, z, 1.0] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])]
        )
        prim_world_tf = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        delta = resolve_prim_world_delta(str(prim.GetPath()), deltas)
        if delta is not None:
            prim_world_tf = prim_world_tf @ delta
        corners.append((box_corners @ (prim_world_tf @ root_world_tf_inv))[:, :3] * scale_np)

    assert corners, f"no bounded geometry found under {root_path} in {usd_path}"
    stacked = np.vstack(corners)
    return AxisAlignedBoundingBox(
        min_point=tuple(stacked.min(axis=0)),
        max_point=tuple(stacked.max(axis=0)),
    )

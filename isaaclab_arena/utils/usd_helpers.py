# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import trimesh
from contextlib import contextmanager

from pxr import Usd, UsdGeom, UsdLux, UsdPhysics

from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.utils.bounding_box import OrientedBoundingBox


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


def compute_local_bounding_box_from_usd(
    usd_path: str,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> OrientedBoundingBox:
    """Compute default-prim-local bounds matching an Isaac Lab USD spawn.

    The default prim's own transform is excluded because Isaac Lab's spawner
    ignores it. ``scale`` is applied once to match the spawn wrapper.

    Args:
        usd_path: Path to the USD file.
        scale: Spawn-time scale passed to ``UsdFileCfg`` / ``Object.scale``.

    Returns:
        OrientedBoundingBox containing local bounds.
    """
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise ValueError(f"Failed to open USD file: {usd_path}")

    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        default_prim = stage.GetPseudoRoot()

    bbox = compute_local_bounding_box_from_prim(stage, default_prim.GetPath().pathString)

    scale_array = np.asarray(scale, dtype=np.float32)
    return OrientedBoundingBox(
        center=bbox.center * bbox.center.new_tensor(scale_array),
        half_extents=bbox.half_extents * bbox.half_extents.new_tensor(np.abs(scale_array)),
        rotation_xyzw=bbox.rotation_xyzw,
    )


def compute_local_bounding_box_from_prim(
    stage: Usd.Stage,
    prim_path: str,
) -> OrientedBoundingBox:
    """Compute axis-aligned geometry in the prim's local frame.

    Args:
        stage: The USD stage containing the prim.
        prim_path: Path to the prim to compute the bounding box for.

    Returns:
        Axis-aligned bounds in the prim's local frame.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"No prim found at path {prim_path}")

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeUntransformedBound(prim)
    bbox_range = bbox.ComputeAlignedBox()
    local_min = bbox_range.GetMin()
    local_max = bbox_range.GetMax()

    return OrientedBoundingBox.from_min_max(
        min_point=(local_min[0], local_min[1], local_min[2]),
        max_point=(local_max[0], local_max[1], local_max[2]),
    )


def extract_trimesh_from_usd(
    usd_path: str,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> trimesh.Trimesh:
    """Extract mesh geometry under a USD's default prim into a single trimesh.

    This is the public alias for ``extract_trimesh_from_usd_path``.
    Unlike the legacy whole-stage extractor, it returns default-prim-local geometry
    and applies scale after child transforms so mesh and bounding-box frames agree.
    All scale components must be positive (negative flips winding/SDF sign).
    Other Gprim geometry under the default prim is rejected, not silently dropped.

    Args:
        usd_path: Path to the .usd/.usda/.usdc file.
        scale: (sx, sy, sz) per-axis scale factors applied in the default-prim frame.

    Returns:
        Combined trimesh in the scaled default-prim frame.
    """
    return extract_trimesh_from_usd_path(usd_path, scale)


def extract_trimesh_from_prim(
    stage: Usd.Stage,
    prim_path: str,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> trimesh.Trimesh:
    """Extract UsdGeom.Mesh geometry under a prim into the prim's local frame.

    Other Gprim geometry is rejected, not silently dropped.
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

    for prim in Usd.PrimRange(root_prim):
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

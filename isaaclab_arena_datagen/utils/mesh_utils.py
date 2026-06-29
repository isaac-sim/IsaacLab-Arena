# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""USD mesh extraction and reconstruction helpers for the data-generation pipeline."""

from __future__ import annotations

import numpy as np
import trimesh
import trimesh.creation
from typing import Any

from pxr import UsdGeom


def triangulate_usd_faces(prim: Any) -> np.ndarray:
    """Convert a USD Mesh prim into triangulated face indices ``(F, 3)``."""
    mesh = UsdGeom.Mesh(prim)
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    faces_n3: list = []
    it = iter(indices)
    for face_vertex_count in counts:
        poly = [next(it) for _ in range(face_vertex_count)]
        for k in range(1, face_vertex_count - 1):
            faces_n3.append([poly[0], poly[k], poly[k + 1]])
    return np.asarray(faces_n3, dtype=np.int64)


def create_primitive_mesh(prim: Any) -> Any:
    """Create a trimesh mesh from a USD geometric primitive."""
    prim_type = prim.GetTypeName()
    if prim_type == "Cube":
        size_m = UsdGeom.Cube(prim).GetSizeAttr().Get()
        return trimesh.creation.box(extents=(size_m, size_m, size_m))
    if prim_type == "Sphere":
        radius_m = UsdGeom.Sphere(prim).GetRadiusAttr().Get()
        return trimesh.creation.icosphere(subdivisions=3, radius=radius_m)
    if prim_type == "Cylinder":
        c = UsdGeom.Cylinder(prim)
        return trimesh.creation.cylinder(
            radius=c.GetRadiusAttr().Get(),
            height=c.GetHeightAttr().Get(),
        )
    if prim_type == "Capsule":
        c = UsdGeom.Capsule(prim)
        return trimesh.creation.capsule(
            radius=c.GetRadiusAttr().Get(),
            height=c.GetHeightAttr().Get(),
        )
    if prim_type == "Cone":
        c = UsdGeom.Cone(prim)
        return trimesh.creation.cone(
            radius=c.GetRadiusAttr().Get(),
            height=c.GetHeightAttr().Get(),
        )
    raise KeyError(f"{prim_type} is not a valid primitive mesh type")


def reconstruct_mesh_points_at_step(
    mesh_samples: Any,
    T_W_from_localbody_arrays: dict[str, np.ndarray],
    step_idx: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Reconstruct world-space positions and normals of sampled mesh points.

    For each object/link, applies the SE(3) pose at *step_idx* to the stored
    relative SE(3) per point to recover world-space positions and normals.

    Args:
        mesh_samples: The :class:`MeshSamplesResult` from
            :meth:`DynamicObjectTracker.sample_dynamic_object_meshes`.
        T_W_from_localbody_arrays: Dict of ``(num_steps, 3, 4)`` float32 arrays,
            loaded from ``dynamic_objects_poses.npz``.
        step_idx: The simulation step to reconstruct at.

    Returns:
        Dict mapping each pose-array key to a tuple
        ``(points_W_n3, normals_W_n3)`` where each is ``(N, 3)`` float32.
    """
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for key, se3_localbody_from_point_n34 in mesh_samples.se3_localbody_from_point_arrays.items():
        if key not in T_W_from_localbody_arrays:
            continue
        num_points = se3_localbody_from_point_n34.shape[0]

        T_W_from_localbody_34 = T_W_from_localbody_arrays[key][step_idx]  # (3, 4)
        T_W_from_localbody = np.eye(4, dtype=np.float64)
        T_W_from_localbody[:3, :] = T_W_from_localbody_34.astype(np.float64)

        T_localbody_from_point_n = np.zeros((num_points, 4, 4), dtype=np.float64)
        T_localbody_from_point_n[:, :3, :] = se3_localbody_from_point_n34.astype(np.float64)
        T_localbody_from_point_n[:, 3, 3] = 1.0

        T_W_from_point_n = T_W_from_localbody[np.newaxis, :, :] @ T_localbody_from_point_n

        points_W_n3 = T_W_from_point_n[:, :3, 3].astype(np.float32)
        normals_W_n3 = T_W_from_point_n[:, :3, 2].astype(np.float32)

        result[key] = (points_W_n3, normals_W_n3)

    return result

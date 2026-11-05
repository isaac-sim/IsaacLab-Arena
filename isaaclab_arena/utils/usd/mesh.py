# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Read USD polygon meshes without choosing geometry or applying collision processing."""

import numpy as np
from collections.abc import Sequence

from pxr import Usd, UsdGeom

MeshArrays = tuple[np.ndarray, np.ndarray, np.ndarray]
"""Vertices, polygon vertex counts, and flattened polygon vertex indices."""


def read_mesh_arrays(mesh: UsdGeom.Mesh, T_F_M: np.ndarray) -> MeshArrays | None:
    """Read a mesh in a caller-selected frame, preserving its polygon topology.

    Args:
        mesh: USD mesh to read at the default time. Selection and validation belong to the caller.
        T_F_M: USD row-vector transform from mesh frame M to output frame F, including scale.

    Returns:
        Float64 vertices, integer polygon counts and indices, or None if an attribute is missing.
    """
    time_code = Usd.TimeCode.Default()
    points = mesh.GetPointsAttr().Get(time_code)
    face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get(time_code)
    face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get(time_code)
    if points is None or face_vertex_counts is None or face_vertex_indices is None:
        return None

    vertices_M = np.asarray(points, dtype=np.float64)
    vertices_F = vertices_M @ T_F_M[:3, :3] + T_F_M[3, :3]
    return (
        vertices_F,
        np.asarray(face_vertex_counts, dtype=np.int64),
        np.asarray(face_vertex_indices, dtype=np.int64),
    )


def concatenate_mesh_arrays(meshes: Sequence[MeshArrays]) -> MeshArrays:
    """Combine meshes already expressed in the same frame, offsetting their vertex indices.

    Args:
        meshes: Nonempty sequence of vertex, polygon-count and polygon-index arrays.

    Returns:
        Combined arrays in input order, without triangulating or merging vertices.
    """
    assert meshes, "At least one mesh is required for concatenation."
    vertices_by_mesh = []
    face_vertex_counts_by_mesh = []
    face_vertex_indices_by_mesh = []
    vertex_offset = 0
    for vertices, face_vertex_counts, face_vertex_indices in meshes:
        vertices_by_mesh.append(vertices)
        face_vertex_counts_by_mesh.append(face_vertex_counts)
        face_vertex_indices_by_mesh.append(face_vertex_indices + vertex_offset)
        vertex_offset += len(vertices)
    return (
        np.concatenate(vertices_by_mesh),
        np.concatenate(face_vertex_counts_by_mesh),
        np.concatenate(face_vertex_indices_by_mesh),
    )

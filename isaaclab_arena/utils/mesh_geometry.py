# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Local mesh geometry and convex-hull containment checks."""

import numpy as np
import torch
from dataclasses import dataclass


@dataclass(frozen=True, eq=False)
class LocalMeshGeometry:
    """Mesh vertices and original polygon topology in one asset pose frame."""

    vertices: np.ndarray
    """Finite float64 vertex coordinates with shape (P, 3)."""

    face_vertex_counts: np.ndarray
    """Integer number of vertices in each polygon, with shape (F,)."""

    face_vertex_indices: np.ndarray
    """Flattened integer vertex indices, preserving polygon order."""

    def __post_init__(self):
        _validate_vertices(self.vertices)
        assert self.vertices.dtype == np.float64, "Local mesh vertices must use float64 coordinates"
        assert self.face_vertex_counts.ndim == 1, "Face vertex counts must have shape (F,)"
        assert np.issubdtype(self.face_vertex_counts.dtype, np.integer), "Face vertex counts must be integers"
        assert self.face_vertex_counts.size > 0, "Local mesh geometry must contain polygons"
        assert np.all(self.face_vertex_counts >= 3), "Each mesh polygon must contain at least three vertices"
        assert self.face_vertex_indices.ndim == 1, "Face vertex indices must be a flattened array"
        assert np.issubdtype(self.face_vertex_indices.dtype, np.integer), "Face vertex indices must be integers"
        assert (
            self.face_vertex_counts.sum() == self.face_vertex_indices.size
        ), "Face vertex counts must match the number of face vertex indices"
        assert np.all(self.face_vertex_indices >= 0), "Face vertex indices must be non-negative"
        assert np.all(self.face_vertex_indices < len(self.vertices)), "Face vertex indices exceed the vertex array"


@dataclass(frozen=True, eq=False)
class ConvexHullGeometry:
    """Convex-hull vertices and plane equations in the mesh's local frame."""

    vertices: torch.Tensor
    """Float32 hull vertices with shape (V, 3)."""

    vertex_centroid: torch.Tensor
    """Arithmetic mean of the hull vertices, with shape (3,); not a mass or volume centroid."""

    planes: torch.Tensor
    """Outward half-space equations (nx, ny, nz, d), with shape (F, 4)."""

    open_top_planes: torch.Tensor
    """Plane equations remaining after removing normals with local z >= 0.7."""

    def __post_init__(self):
        assert self.vertices.ndim == 2 and self.vertices.shape[1] == 3, "Hull vertices must have shape (V, 3)"
        assert self.vertices.shape[0] >= 4, "A three-dimensional hull needs at least four vertices"
        assert self.vertex_centroid.shape == (3,), "Hull vertex centroid must have shape (3,)"
        for plane_equations in (self.planes, self.open_top_planes):
            assert plane_equations.ndim == 2 and plane_equations.shape[1] == 4, "Hull planes must have shape (F, 4)"
            assert plane_equations.shape[0] > 0, "Hull plane sets must not be empty"
        for geometry_tensor in (self.vertices, self.vertex_centroid, self.planes, self.open_top_planes):
            assert geometry_tensor.dtype == torch.float32, "Hull geometry must use float32 tensors"
            assert geometry_tensor.device == self.vertices.device, "Hull geometry must share one device"
            assert torch.isfinite(geometry_tensor).all(), "Hull geometry must be finite"


def _validate_vertices(vertices: np.ndarray) -> None:
    """Validate finite three-dimensional mesh coordinates."""
    assert vertices.ndim == 2 and vertices.shape[1] == 3, "Vertices must have shape (P, 3)"
    assert vertices.shape[0] > 0, "Mesh geometry must contain vertices"
    assert np.issubdtype(vertices.dtype, np.floating) or np.issubdtype(
        vertices.dtype, np.integer
    ), "Vertices must contain real numeric coordinates"
    assert np.isfinite(vertices).all(), "Vertices must be finite"


def build_convex_hull(vertices: np.ndarray, device: torch.device | str = "cpu") -> ConvexHullGeometry:
    """Build a convex hull with closed and open-top plane sets.

    Faces with local normal z >= 0.7 are removed for the open-top region.
    Retained sloped walls can still constrain points above the original top.
    Degenerate geometry is rejected by SciPy without perturbing the vertices.

    Args:
        vertices: Finite mesh coordinates with shape (P, 3), including any asset scale.
        device: Device on which the returned float32 tensors are stored.

    Returns:
        Local hull geometry with both complete and open-top plane equations.
    """
    from scipy.spatial import ConvexHull

    _validate_vertices(vertices)
    assert vertices.shape[0] >= 4, "A three-dimensional hull needs at least four input vertices"
    mesh_vertices = np.asarray(vertices, dtype=np.float64)
    convex_hull = ConvexHull(mesh_vertices)
    hull_vertices = torch.tensor(mesh_vertices[convex_hull.vertices], dtype=torch.float32, device=device)
    hull_planes = torch.tensor(convex_hull.equations, dtype=torch.float32, device=device)
    return ConvexHullGeometry(
        vertices=hull_vertices,
        vertex_centroid=hull_vertices.mean(dim=0),
        planes=hull_planes,
        open_top_planes=hull_planes[hull_planes[:, 2] < 0.7],
    )


def points_inside_convex_hull(points: torch.Tensor, planes: torch.Tensor) -> torch.Tensor:
    """Test points against every hull half-space, including its exact boundary.

    Args:
        points: Coordinates with shape (N, 3), in the plane equations' frame.
        planes: Finite outward plane equations with shape (F, 4), as returned by
            build_convex_hull. Must share the points' floating dtype and device.

    Returns:
        Boolean tensor with shape (N,), true when every equation is <= 0.
    """
    assert points.ndim == 2 and points.shape[1] == 3, "Points must have shape (N, 3)"
    assert planes.ndim == 2 and planes.shape[1] == 4, "Planes must have shape (F, 4)"
    assert planes.shape[0] > 0, "Hull plane sets must not be empty"
    assert points.is_floating_point() and planes.dtype == points.dtype, "Points and planes must share a floating dtype"
    assert planes.device == points.device, "Points and planes must share a device"
    signed_plane_distances = points @ planes[:, :3].T + planes[:, 3]
    return signed_plane_distances.amax(dim=-1) <= 0

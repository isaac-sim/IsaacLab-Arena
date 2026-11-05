# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for local mesh topology and convex-hull containment semantics."""

import numpy as np
import torch
from scipy.spatial import QhullError

import pytest

from isaaclab_arena.utils.mesh_geometry import LocalMeshGeometry, build_convex_hull, points_inside_convex_hull


def _box_vertices(half_extent: float = 1.0) -> np.ndarray:
    return np.array(
        [
            [-half_extent, -half_extent, -half_extent],
            [half_extent, -half_extent, -half_extent],
            [-half_extent, half_extent, -half_extent],
            [half_extent, half_extent, -half_extent],
            [-half_extent, -half_extent, half_extent],
            [half_extent, -half_extent, half_extent],
            [-half_extent, half_extent, half_extent],
            [half_extent, half_extent, half_extent],
        ],
        dtype=np.float64,
    )


def test_local_mesh_preserves_mixed_polygon_topology():
    vertices = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [0, 0, 2]], dtype=np.float64)
    face_vertex_counts = np.array([4, 3, 3, 3, 3], dtype=np.int64)
    face_vertex_indices = np.array([0, 3, 2, 1, 0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4], dtype=np.int64)

    mesh = LocalMeshGeometry(vertices, face_vertex_counts, face_vertex_indices)

    np.testing.assert_array_equal(mesh.vertices, vertices)
    np.testing.assert_array_equal(mesh.face_vertex_counts, face_vertex_counts)
    np.testing.assert_array_equal(mesh.face_vertex_indices, face_vertex_indices)


@pytest.mark.parametrize(
    "face_vertex_counts, face_vertex_indices",
    [
        ([], []),
        ([2], [0, 1]),
        ([3], [0, 1]),
        ([3], [0, 1, -1]),
        ([3], [0, 1, 8]),
    ],
)
def test_local_mesh_rejects_invalid_topology(face_vertex_counts, face_vertex_indices):
    with pytest.raises(AssertionError):
        LocalMeshGeometry(
            _box_vertices(),
            np.array(face_vertex_counts, dtype=np.int64),
            np.array(face_vertex_indices, dtype=np.int64),
        )


def test_convex_hull_includes_boundaries_and_rejects_every_outside_direction():
    hull = build_convex_hull(_box_vertices())
    points = torch.tensor(
        [
            [0, 0, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
            [1, 1, 1],
            [1.01, 0, 0],
            [-1.01, 0, 0],
            [0, 1.01, 0],
            [0, -1.01, 0],
            [0, 0, 1.01],
            [0, 0, -1.01],
        ],
        dtype=torch.float32,
    )

    inside = points_inside_convex_hull(points, hull.planes)

    assert inside.shape == (14,)
    assert inside.dtype == torch.bool
    assert inside.tolist() == [True] * 8 + [False] * 6


def test_open_top_keeps_lateral_and_bottom_bounds():
    hull = build_convex_hull(_box_vertices())
    points = torch.tensor([[0, 0, 100], [1, 0, 100], [1.01, 0, 100], [0, 0, -1.01]], dtype=torch.float32)

    assert points_inside_convex_hull(points, hull.planes).tolist() == [False, False, False, False]
    assert points_inside_convex_hull(points, hull.open_top_planes).tolist() == [True, True, False, False]


def test_centroid_containment_does_not_require_the_whole_object_inside():
    container = build_convex_hull(_box_vertices())
    object_vertices = _box_vertices(half_extent=0.25)
    object_vertices[:, 0] *= 8
    object_hull = build_convex_hull(object_vertices)

    centroid_inside = points_inside_convex_hull(object_hull.vertex_centroid.unsqueeze(0), container.planes)
    vertices_inside = points_inside_convex_hull(object_hull.vertices, container.planes)

    assert centroid_inside.tolist() == [True]
    assert not vertices_inside.any()


def test_vertex_centroid_differs_from_aabb_center_and_ignores_interior_samples():
    vertices = np.array([[0, 0, 0], [4, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=np.float64)
    interior_vertices = np.array([[0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [0.1, 0.2, 0.1]], dtype=np.float64)

    hull = build_convex_hull(np.concatenate([vertices, interior_vertices]))

    torch.testing.assert_close(hull.vertex_centroid, torch.tensor([1.0, 0.5, 0.25]))
    assert not torch.allclose(hull.vertex_centroid, torch.tensor([2.0, 1.0, 0.5]))


def test_convex_hull_rejects_aabb_false_positives():
    vertices = np.array([[0, 0, 0], [4, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=np.float64)
    hull = build_convex_hull(vertices)
    points = torch.tensor([[0.5, 0.25, 0.125], [2.0, 1.0, 0.5]])

    # Both points pass the bounding box, but the second crosses the tetrahedron's sloped face.
    assert ((points >= 0) & (points <= torch.tensor([4.0, 2.0, 1.0]))).all()
    assert points_inside_convex_hull(points, hull.planes).tolist() == [True, False]


def test_convex_hull_fills_a_concave_mesh_notch():
    # An extruded L occupies x <= 1 or y <= 1 within [0, 2] x [0, 2].
    vertices = np.array(
        [
            [0, 0, 0],
            [2, 0, 0],
            [2, 1, 0],
            [1, 1, 0],
            [1, 2, 0],
            [0, 2, 0],
            [0, 0, 1],
            [2, 0, 1],
            [2, 1, 1],
            [1, 1, 1],
            [1, 2, 1],
            [0, 2, 1],
        ],
        dtype=np.float64,
    )
    hull = build_convex_hull(vertices)
    points = torch.tensor([[1.4, 1.4, 0.5], [1.8, 1.8, 0.5]])

    assert points_inside_convex_hull(points, hull.planes).tolist() == [True, False]


def test_tapered_walls_can_bound_the_open_top_region():
    # These walls satisfy |x| + z / 4 <= 1 and |y| + z / 4 <= 1.
    # Their normals have z < 0.7, so removing the top preserves these bounds.
    vertices = np.array(
        [
            [-1, -1, 0],
            [1, -1, 0],
            [-1, 1, 0],
            [1, 1, 0],
            [-0.5, -0.5, 2],
            [0.5, -0.5, 2],
            [-0.5, 0.5, 2],
            [0.5, 0.5, 2],
        ],
        dtype=np.float64,
    )
    hull = build_convex_hull(vertices)
    points = torch.tensor([[0, 0, 3], [0.5, 0, 3], [0, 0, 5]], dtype=torch.float32)

    assert points_inside_convex_hull(points, hull.planes).tolist() == [False, False, False]
    assert points_inside_convex_hull(points, hull.open_top_planes).tolist() == [True, False, False]


@pytest.mark.parametrize(
    "vertices",
    [
        np.empty((0, 3)),
        np.zeros((3,)),
        np.zeros((4, 2)),
        np.zeros((3, 3)),
        np.full((4, 3), np.nan),
        np.full((4, 3), np.inf),
    ],
)
def test_convex_hull_rejects_missing_or_invalid_vertices(vertices):
    with pytest.raises(AssertionError):
        build_convex_hull(vertices)


@pytest.mark.parametrize(
    "vertices",
    [
        np.zeros((4, 3)),
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64),
        np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np.float64),
    ],
)
def test_convex_hull_rejects_degenerate_geometry_without_perturbing_it(vertices):
    with pytest.raises(QhullError):
        build_convex_hull(vertices)


@pytest.mark.parametrize("point_shape, plane_shape", [((3,), (6, 4)), ((2, 4), (6, 4)), ((2, 3), (6, 3))])
def test_point_query_rejects_ambiguous_shapes(point_shape, plane_shape):
    with pytest.raises(AssertionError):
        points_inside_convex_hull(torch.zeros(point_shape), torch.zeros(plane_shape))


def test_point_query_rejects_empty_plane_set():
    with pytest.raises(AssertionError, match="must not be empty"):
        points_inside_convex_hull(torch.zeros((1, 3)), torch.empty((0, 4)))


def test_point_query_accepts_empty_point_batch():
    hull = build_convex_hull(_box_vertices())

    result = points_inside_convex_hull(torch.empty((0, 3)), hull.planes)

    assert result.shape == (0,)
    assert result.dtype == torch.bool

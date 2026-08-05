# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for placement geometry drawing helpers (no Kit required)."""

from __future__ import annotations

import numpy as np

import pytest

from isaaclab_arena.visualization.isaac_sim_debug_draw import (
    oriented_bbox_corners,
    oriented_bbox_edge_segments,
    rotate_points_xyzw,
    transform_trimesh_vertices,
    trimesh_edge_segments,
)
from isaaclab_arena.visualization.placement_geometry_draw import KIND_COLORS, classify_entity_kind


def test_oriented_bbox_corners_identity_pose():
    corners = oriented_bbox_corners(
        min_point=(-1.0, -2.0, -3.0),
        max_point=(1.0, 2.0, 3.0),
        position_xyz=(0.0, 0.0, 0.0),
        rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    assert corners.shape == (8, 3)
    assert np.allclose(corners.min(axis=0), [-1.0, -2.0, -3.0])
    assert np.allclose(corners.max(axis=0), [1.0, 2.0, 3.0])


def test_oriented_bbox_corners_translation():
    corners = oriented_bbox_corners(
        min_point=(0.0, 0.0, 0.0),
        max_point=(1.0, 1.0, 1.0),
        position_xyz=(10.0, 20.0, 30.0),
        rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    assert np.allclose(corners.min(axis=0), [10.0, 20.0, 30.0])
    assert np.allclose(corners.max(axis=0), [11.0, 21.0, 31.0])


def test_oriented_bbox_edge_segments_count():
    starts, ends = oriented_bbox_edge_segments(
        min_point=(0.0, 0.0, 0.0),
        max_point=(1.0, 1.0, 1.0),
        position_xyz=(0.0, 0.0, 0.0),
        rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    assert len(starts) == 12
    assert len(ends) == 12


def test_rotate_points_xyzw_180_about_z():
    # 180 deg about Z: (x,y,z) -> (-x,-y,z); quat xyzw = (0,0,1,0)
    pts = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float64)
    out = rotate_points_xyzw(pts, (0.0, 0.0, 1.0, 0.0))
    assert np.allclose(out, [[-1.0, 0.0, 0.0], [0.0, -2.0, 0.0]], atol=1e-6)


def test_trimesh_edge_segments_unique_and_decimated():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int32)
    starts, ends = trimesh_edge_segments(vertices, faces, max_edges=100)
    assert len(starts) == len(ends)
    # Two triangles sharing an edge => 5 unique edges.
    assert len(starts) == 5

    starts_cap, ends_cap = trimesh_edge_segments(vertices, faces, max_edges=2)
    assert len(starts_cap) == 2
    assert len(ends_cap) == 2

    starts_full, ends_full = trimesh_edge_segments(vertices, faces, max_edges=None)
    assert starts_full == starts
    assert ends_full == ends


def test_transform_trimesh_vertices_identity():
    verts = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    out = transform_trimesh_vertices(verts, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    assert np.allclose(out, verts)


def test_kind_colors_cover_expected_kinds():
    for kind in ("background", "embodiment", "object", "object_reference", "other"):
        assert kind in KIND_COLORS
        assert len(KIND_COLORS[kind]) == 4


def test_classify_entity_kind_fixed_collision_object():
    import trimesh

    from isaaclab_arena.relations.background_collision_object import FixedCollisionObject

    mesh = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    fixed = FixedCollisionObject(mesh, name="fixed_collision_mesh")
    assert classify_entity_kind(fixed) == "background"


def test_live_pose_fallback_collapses_per_env_and_ranged_poses():
    from types import SimpleNamespace

    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose, PosePerEnv, PoseRange
    from isaaclab_arena.visualization.placement_geometry_draw import _live_pose_for_asset

    class _Placeable(PlaceableAsset):
        def get_bounding_box(self):
            return AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 1.0))

    env = SimpleNamespace(unwrapped=SimpleNamespace(scene={}))
    asset = _Placeable(name="missing_from_scene")
    env_zero_pose = Pose(position_xyz=(1.0, 2.0, 3.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))

    asset.initial_pose = PosePerEnv([env_zero_pose])
    assert _live_pose_for_asset(env, asset) == env_zero_pose

    asset.initial_pose = PoseRange(position_xyz_min=(0.0, 2.0, 4.0), position_xyz_max=(2.0, 4.0, 6.0))
    assert _live_pose_for_asset(env, asset).position_xyz == pytest.approx((1.0, 3.0, 5.0))


def test_redraw_reports_persistent_aabb_failure_once(monkeypatch, capsys):
    from types import SimpleNamespace

    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena.visualization import placement_geometry_draw

    asset = SimpleNamespace(name="broken")
    entry = placement_geometry_draw._OverlayEntry(
        name="broken",
        kind="other",
        color=(1.0, 1.0, 0.0, 1.0),
        asset=asset,
        mesh=None,
    )
    draw = placement_geometry_draw.PlacementGeometryDraw.__new__(placement_geometry_draw.PlacementGeometryDraw)
    draw._entries = [entry]
    draw._draw = SimpleNamespace(clear=lambda: None)
    draw._static_meshes_drawn = True
    draw._reported_aabb_failures = set()

    monkeypatch.setattr(
        placement_geometry_draw,
        "_live_pose_for_asset",
        lambda env, received_asset: Pose.identity(),
    )

    def fail_aabb(received_asset, pose):
        raise ValueError("invalid bounds")

    monkeypatch.setattr(placement_geometry_draw, "_solver_world_aabb", fail_aabb)

    draw.redraw(SimpleNamespace())
    draw.redraw(SimpleNamespace())

    assert capsys.readouterr().out.count("skip AABB for 'broken'") == 1


def test_solver_world_aabb_fixed_collision_object_is_world_baked():
    import trimesh

    from isaaclab_arena.relations.background_collision_object import FixedCollisionObject
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena.visualization.placement_geometry_draw import _solver_world_aabb

    mesh = trimesh.creation.box(extents=(0.2, 0.4, 0.6))
    mesh.apply_translation([1.0, 2.0, 3.0])
    fixed = FixedCollisionObject(mesh, name="fixed")
    # Pose must be ignored: mesh / AABB are already in world frame.
    pose = Pose(position_xyz=(9.0, 9.0, 9.0), rotation_xyzw=(0.0, 0.0, 0.70710678, 0.70710678))
    world = _solver_world_aabb(fixed, pose)
    expected = fixed.get_world_bounding_box()
    assert np.allclose(world.min_point.numpy(), expected.min_point.numpy())
    assert np.allclose(world.max_point.numpy(), expected.max_point.numpy())


def test_solver_world_aabb_live_placeable_matches_rotated_then_translated():
    """Non-anchor placeables: solver refits AABB under quat, then translates (not OBB draw)."""
    import math

    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena.visualization.placement_geometry_draw import _solver_world_aabb

    class _FakePlaceable:
        name = "fake"
        is_anchor = False

        def get_bounding_box(self):
            return AxisAlignedBoundingBox(min_point=(-2.0, -0.5, 0.0), max_point=(2.0, 0.5, 1.0))

        def get_world_bounding_box(self):
            raise AssertionError("live non-anchors must not use get_world_bounding_box in overlay")

        def get_relations(self):
            return []

    yaw = math.pi / 2
    # xyzw for +90 deg about Z
    quat = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    pose = Pose(position_xyz=(10.0, 20.0, 30.0), rotation_xyzw=quat)
    asset = _FakePlaceable()
    world = _solver_world_aabb(asset, pose)
    expected = asset.get_bounding_box().rotated_by_quat(quat).translated(pose.position_xyz)
    assert np.allclose(world.min_point.numpy(), expected.min_point.numpy())
    assert np.allclose(world.max_point.numpy(), expected.max_point.numpy())
    # Must remain axis-aligned in world (solver style), not an OBB at the live pose.
    # After 90° Z, the long axis (was X) becomes Y: extents ~1 on X, ~4 on Y.
    size = (world.max_point - world.min_point)[0].tolist()
    assert size[0] == pytest.approx(1.0, abs=1e-5)
    assert size[1] == pytest.approx(4.0, abs=1e-5)

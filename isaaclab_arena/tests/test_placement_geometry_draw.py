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
from isaaclab_arena.visualization.placement_geometry_draw import _color_for_asset


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


def test_color_for_fixed_collision_object_uses_background_color():
    import trimesh

    from isaaclab_arena.relations.background_collision_object import FixedCollisionObject

    mesh = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    fixed = FixedCollisionObject(mesh, name="fixed_collision_mesh")
    assert _color_for_asset(fixed) == (1.0, 0.55, 0.0, 1.0)


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


def test_live_pose_converts_warp_scene_buffers_to_torch(monkeypatch):
    import torch
    from types import SimpleNamespace

    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.visualization import placement_geometry_draw

    class _Placeable(PlaceableAsset):
        def get_bounding_box(self):
            return AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 1.0, 1.0))

    position_buffer = object()
    quaternion_buffer = object()
    scene_asset = SimpleNamespace(
        data=SimpleNamespace(root_pos_w=position_buffer, root_quat_w=quaternion_buffer),
    )
    env = SimpleNamespace(unwrapped=SimpleNamespace(scene={"asset": scene_asset}))
    asset = _Placeable(name="asset")
    monkeypatch.setattr(asset, "get_scene_key", lambda: "asset")

    converted = {
        position_buffer: torch.tensor([[1.0, 2.0, 3.0]]),
        quaternion_buffer: torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
    }
    monkeypatch.setattr(placement_geometry_draw.wp, "to_torch", converted.__getitem__)

    pose = placement_geometry_draw._live_pose_for_asset(env, asset)

    assert pose is not None
    assert pose.position_xyz == pytest.approx((1.0, 2.0, 3.0))
    assert pose.rotation_xyzw == pytest.approx((0.0, 0.0, 0.0, 1.0))


def test_redraw_reports_persistent_aabb_failure_once(monkeypatch, capsys):
    from types import SimpleNamespace

    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena.visualization import placement_geometry_draw

    class _BrokenAsset:
        name = "broken"

        def get_bounding_box(self):
            raise ValueError("invalid bounds")

    asset = _BrokenAsset()
    entry = placement_geometry_draw._OverlayEntry(
        name="broken",
        color=(1.0, 1.0, 0.0, 1.0),
        asset=asset,
        mesh=None,
        is_static=False,
    )
    draw = placement_geometry_draw.PlacementGeometryDraw.__new__(placement_geometry_draw.PlacementGeometryDraw)
    draw._entries = [entry]
    draw._draw = SimpleNamespace(clear=lambda: None)
    draw._static_geometry_drawn = True
    draw._reported_aabb_failures = set()

    monkeypatch.setattr(
        placement_geometry_draw,
        "_live_pose_for_asset",
        lambda env, received_asset: Pose.identity(),
    )

    draw.redraw(SimpleNamespace())
    draw.redraw(SimpleNamespace())

    assert capsys.readouterr().out.count("skip AABB for 'broken'") == 1


def test_redraw_skips_static_entries_and_transforms_live_aabb(monkeypatch):
    import math
    from types import SimpleNamespace

    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena.visualization import placement_geometry_draw

    class _Asset:
        def __init__(self, name):
            self.name = name

        def get_bounding_box(self):
            return AxisAlignedBoundingBox(min_point=(-2.0, -0.5, 0.0), max_point=(2.0, 0.5, 1.0))

    yaw = math.pi / 2
    quat = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    pose = Pose(position_xyz=(10.0, 20.0, 30.0), rotation_xyzw=quat)
    static_asset = _Asset("static")
    live_asset = _Asset("live")
    entries = [
        placement_geometry_draw._OverlayEntry(
            name="static",
            color=(1.0, 1.0, 0.0, 1.0),
            asset=static_asset,
            mesh=None,
            is_static=True,
        ),
        placement_geometry_draw._OverlayEntry(
            name="live",
            color=(1.0, 1.0, 0.0, 1.0),
            asset=live_asset,
            mesh=None,
            is_static=False,
        ),
    ]
    drawn_bboxes = []
    fake_debug_draw = SimpleNamespace(
        clear=lambda: None,
        draw_oriented_bbox=lambda min_pt, max_pt, *args, **kwargs: drawn_bboxes.append((min_pt, max_pt)),
    )
    draw = placement_geometry_draw.PlacementGeometryDraw.__new__(placement_geometry_draw.PlacementGeometryDraw)
    draw._entries = entries
    draw._draw = fake_debug_draw
    draw._static_geometry_drawn = True
    draw._reported_aabb_failures = set()

    pose_requests = []

    def live_pose(env, asset):
        pose_requests.append(asset)
        return pose

    monkeypatch.setattr(placement_geometry_draw, "_live_pose_for_asset", live_pose)
    draw.redraw(SimpleNamespace())

    assert pose_requests == [live_asset]
    assert len(drawn_bboxes) == 1
    min_pt, max_pt = drawn_bboxes[0]
    assert np.subtract(max_pt, min_pt) == pytest.approx((1.0, 4.0, 1.0), abs=1e-5)

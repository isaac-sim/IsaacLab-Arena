# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Wireframe overlay of relation-solver / placement-validator geometry in Kit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import warp as wp

from isaaclab_arena.relations.collision_mode import CollisionMode, object_uses_mesh_collision
from isaaclab_arena.relations.placement_events import get_placement_pool
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.visualization.isaac_sim_debug_draw import (
    IsaacSimDebugDraw,
    oriented_bbox_edge_segments,
    transform_trimesh_vertices,
    trimesh_edge_segments,
)

if TYPE_CHECKING:
    import gymnasium as gym
    import trimesh

    from isaaclab_arena.relations.collision_object import CollisionObject
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer

IDENTITY_XYZW = (0.0, 0.0, 0.0, 1.0)
IDENTITY_POS = (0.0, 0.0, 0.0)

MAX_MESH_EDGES = 8000
STATIC_GEOMETRY_ROOT = "/World/IsaacLabArenaPlacementStaticGeometry"


@dataclass(frozen=True)
class _PlacementGeometry:
    """Geometry and display metadata for one placement collision asset."""

    name: str
    color: tuple[float, float, float, float]
    asset: CollisionObject
    mesh: trimesh.Trimesh | None
    is_static: bool


class PlacementGeometryDraw:
    """Draw placement AABBs and MESH-mode collision meshes as Kit wireframe overlays."""

    def __init__(self, pool: PooledObjectPlacer, max_mesh_edges: int = MAX_MESH_EDGES):
        self._max_mesh_edges = max_mesh_edges
        self._draw = IsaacSimDebugDraw()
        self._default_mode: CollisionMode = pool.default_collision_mode
        self._geometries = _build_placement_geometries(pool)
        self._legend_printed = False
        self._static_geometry_drawn = False
        self._reported_aabb_failures: set[int] = set()

    @classmethod
    def from_env(cls, env: gym.Env) -> PlacementGeometryDraw | None:
        """Build an overlay from the env placement pool, or ``None`` when absent."""
        pool = get_placement_pool(env)
        if pool is None:
            return None
        return cls(pool)

    def print_legend(self) -> None:
        """Print color and geometry kind for each overlay entity once."""
        if self._legend_printed:
            return
        self._legend_printed = True
        print(
            f"[placement_debug] default collision_mode={self._default_mode.value}; "
            f"drawing {len(self._geometries)} entit(y/ies)",
            flush=True,
        )
        for geometry in self._geometries:
            geom_label = "AABB+mesh" if geometry.mesh is not None else "AABB"
            print(
                f"[placement_debug] '{geometry.name}' {geom_label} color={geometry.color}",
                flush=True,
            )

    def redraw(self, env: gym.Env) -> None:
        """Draw static geometry once, then refresh overlays for live assets."""
        self._draw_static_geometry_once(env)
        self._draw.clear()
        for geometry in self._geometries:
            if geometry.is_static:
                continue
            pose = _live_pose_for_asset(env, geometry.asset)
            if pose is None:
                continue
            try:
                # Draw the solver's world AABB (axis-aligned), not a full-pose OBB.
                local_bbox = geometry.asset.get_bounding_box()
                world_bbox = local_bbox.rotated_by_quat(pose.rotation_xyzw).translated(pose.position_xyz)
            except Exception as exc:  # noqa: BLE001 — visualizer must not crash the runner
                asset_id = id(geometry.asset)
                if asset_id not in self._reported_aabb_failures:
                    self._reported_aabb_failures.add(asset_id)
                    print(f"[placement_debug] skip AABB for '{geometry.name}': {exc}", flush=True)
            else:
                self._reported_aabb_failures.discard(id(geometry.asset))
                min_pt = tuple(float(v) for v in world_bbox.min_point[0].tolist())
                max_pt = tuple(float(v) for v in world_bbox.max_point[0].tolist())
                self._draw.draw_oriented_bbox(
                    min_pt,
                    max_pt,
                    IDENTITY_POS,
                    IDENTITY_XYZW,
                    color=geometry.color,
                    thickness=3.0,
                )
            if geometry.mesh is not None:
                # Collision meshes are in a true local frame; apply the live pose.
                self._draw.draw_trimesh_wireframe(
                    geometry.mesh,
                    color=geometry.color,
                    thickness=1.5,
                    max_edges=self._max_mesh_edges,
                    position_xyz=pose.position_xyz,
                    rotation_xyzw=pose.rotation_xyzw,
                )

    def close(self) -> None:
        """Remove persistent geometry and clear transient lines."""
        import omni.usd

        self._draw.clear()
        stage = omni.usd.get_context().get_stage()
        if stage is not None:
            stage.RemovePrim(STATIC_GEOMETRY_ROOT)
        self._static_geometry_drawn = False

    def _draw_static_geometry_once(self, env: gym.Env) -> None:
        """Create persistent USD curves for static AABBs and collision meshes."""
        if self._static_geometry_drawn:
            return

        import omni.usd
        from pxr import Tf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        self._static_geometry_drawn = True

        UsdGeom.Scope.Define(stage, STATIC_GEOMETRY_ROOT)
        for index, geometry in enumerate(self._geometries):
            if not geometry.is_static:
                continue

            prim_name = Tf.MakeValidIdentifier(f"{index}_{geometry.name}")
            try:
                world_bbox = geometry.asset.get_world_bounding_box()
                min_pt = tuple(float(v) for v in world_bbox.min_point[0].tolist())
                max_pt = tuple(float(v) for v in world_bbox.max_point[0].tolist())
                starts, ends = oriented_bbox_edge_segments(min_pt, max_pt, IDENTITY_POS, IDENTITY_XYZW)
                self._define_curve(stage, f"{STATIC_GEOMETRY_ROOT}/{prim_name}_bbox", starts, ends, geometry, 0.01)
            except Exception as exc:  # noqa: BLE001 — visualizer must not crash the runner
                print(f"[placement_debug] skip static AABB for '{geometry.name}': {exc}", flush=True)

            if geometry.mesh is None:
                continue
            pose = _live_pose_for_asset(env, geometry.asset)
            if pose is None:
                continue
            vertices = transform_trimesh_vertices(geometry.mesh.vertices, pose.position_xyz, pose.rotation_xyzw)
            starts, ends = trimesh_edge_segments(vertices, geometry.mesh.faces, max_edges=None)
            self._define_curve(stage, f"{STATIC_GEOMETRY_ROOT}/{prim_name}_mesh", starts, ends, geometry, 0.006)
            print(f"[placement_debug] drew static '{geometry.name}' once with {len(starts)} full edges", flush=True)

    @staticmethod
    def _define_curve(stage, path: str, starts, ends, geometry: _PlacementGeometry, width: float) -> None:
        """Define persistent linear curves for line segments."""
        from pxr import UsdGeom

        points = [point for segment in zip(starts, ends, strict=True) for point in segment]
        curves = UsdGeom.BasisCurves.Define(stage, path)
        curves.CreateTypeAttr(UsdGeom.Tokens.linear)
        curves.CreateCurveVertexCountsAttr([2] * len(starts))
        curves.CreatePointsAttr(points)
        curves.CreateWidthsAttr([width])
        curves.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        curves.CreateDisplayColorAttr([geometry.color[:3]])
        curves.CreateDisplayOpacityAttr([geometry.color[3]])


def _build_placement_geometries(pool: PooledObjectPlacer) -> list[_PlacementGeometry]:
    """Build deduplicated draw geometry from placed and fixed collision assets."""
    assets: list[tuple[CollisionObject, bool]] = [(asset, asset.is_anchor) for asset in pool.objects]
    assets.extend((asset, True) for asset in pool.collision_objects)
    geometries: list[_PlacementGeometry] = []
    seen: set[int] = set()
    for asset, is_static in assets:
        asset_id = id(asset)
        if asset_id in seen:
            continue
        seen.add(asset_id)
        mesh = asset.get_collision_mesh() if object_uses_mesh_collision(asset, pool.default_collision_mode) else None
        geometries.append(
            _PlacementGeometry(
                name=asset.name,
                color=_color_for_asset(asset),
                asset=asset,
                mesh=mesh,
                is_static=is_static,
            )
        )
    return geometries


def _color_for_asset(asset: CollisionObject) -> tuple[float, float, float, float]:
    """Return the debug color for a collision asset's entity kind."""
    from isaaclab_arena.assets.background import Background
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
    from isaaclab_arena.relations.background_collision_object import FixedCollisionObject

    if isinstance(asset, (FixedCollisionObject, Background)):
        return (1.0, 0.55, 0.0, 1.0)  # orange
    if isinstance(asset, EmbodimentBase):
        return (0.2, 0.6, 1.0, 1.0)  # blue
    if isinstance(asset, ObjectReference):
        return (0.95, 0.35, 0.9, 1.0)  # magenta
    if isinstance(asset, Object):
        return (0.2, 0.9, 0.3, 1.0)  # green
    return (0.85, 0.85, 0.2, 1.0)  # yellow


def _live_pose_for_asset(env, asset: CollisionObject) -> Pose | None:
    """Return the world pose used to place mesh overlay geometry for ``asset``."""
    from isaaclab_arena.relations.background_collision_object import FixedCollisionObject
    from isaaclab_arena.relations.placement_asset import PlaceableAsset

    if isinstance(asset, FixedCollisionObject):
        # Mesh vertices are already baked in world coordinates.
        return Pose.identity()

    if not isinstance(asset, PlaceableAsset):
        return None

    scene = env.unwrapped.scene
    scene_key = asset.get_scene_key()
    scene_asset = scene[scene_key] if scene_key in scene.keys() else None
    if scene_asset is not None and hasattr(scene_asset, "data") and hasattr(scene_asset.data, "root_pos_w"):
        pos = wp.to_torch(scene_asset.data.root_pos_w)[0].detach().cpu().tolist()
        # Isaac Lab root_quat_w is (x, y, z, w) — same as Arena Pose.rotation_xyzw.
        quat_xyzw = wp.to_torch(scene_asset.data.root_quat_w)[0].detach().cpu().tolist()
        return Pose(
            position_xyz=tuple(float(v) for v in pos),
            rotation_xyzw=tuple(float(v) for v in quat_xyzw),
        )

    initial_pose = asset.get_initial_pose()
    return initial_pose if isinstance(initial_pose, Pose) else None

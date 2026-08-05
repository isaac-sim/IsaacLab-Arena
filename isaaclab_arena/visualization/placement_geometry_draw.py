# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Wireframe overlay of relation-solver / placement-validator geometry in Kit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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

# One RGBA color per scene-entity kind.
KIND_COLORS: dict[str, tuple[float, float, float, float]] = {
    "background": (1.0, 0.55, 0.0, 1.0),  # orange
    "embodiment": (0.2, 0.6, 1.0, 1.0),  # blue
    "object": (0.2, 0.9, 0.3, 1.0),  # green
    "object_reference": (0.95, 0.35, 0.9, 1.0),  # magenta
    "other": (0.85, 0.85, 0.2, 1.0),  # yellow
}

MAX_MESH_EDGES = 8000
STATIC_GEOMETRY_ROOT = "/World/IsaacLabArenaPlacementStaticGeometry"


@dataclass(frozen=True)
class _OverlayEntry:
    """One placement geometry entity scheduled for drawing."""

    name: str
    kind: str
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
        self._entries = self._build_entries(pool)
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
            f"drawing {len(self._entries)} entit(y/ies)",
            flush=True,
        )
        for entry in self._entries:
            geom_label = "AABB+mesh" if entry.mesh is not None else "AABB"
            print(
                f"[placement_debug] {entry.kind} '{entry.name}' {geom_label} color={entry.color}",
                flush=True,
            )

    def redraw(self, env: gym.Env) -> None:
        """Draw static geometry once, then refresh overlays for live assets."""
        self._draw_static_geometry_once(env)
        self._draw.clear()
        for entry in self._entries:
            if entry.is_static:
                continue
            pose = _live_pose_for_asset(env, entry.asset)
            if pose is None:
                continue
            try:
                # Draw the solver's world AABB (axis-aligned), not a full-pose OBB.
                local_bbox = entry.asset.get_bounding_box()
                world_bbox = local_bbox.rotated_by_quat(pose.rotation_xyzw).translated(pose.position_xyz)
            except Exception as exc:  # noqa: BLE001 — visualizer must not crash the runner
                asset_id = id(entry.asset)
                if asset_id not in self._reported_aabb_failures:
                    self._reported_aabb_failures.add(asset_id)
                    print(f"[placement_debug] skip AABB for '{entry.name}': {exc}", flush=True)
            else:
                self._reported_aabb_failures.discard(id(entry.asset))
                min_pt = tuple(float(v) for v in world_bbox.min_point[0].tolist())
                max_pt = tuple(float(v) for v in world_bbox.max_point[0].tolist())
                self._draw.draw_oriented_bbox(
                    min_pt,
                    max_pt,
                    IDENTITY_POS,
                    IDENTITY_XYZW,
                    color=entry.color,
                    thickness=3.0,
                )
            if entry.mesh is not None:
                # Collision meshes are in a true local frame; apply the live pose.
                self._draw.draw_trimesh_wireframe(
                    entry.mesh,
                    color=entry.color,
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

    def _build_entries(self, pool: PooledObjectPlacer) -> list[_OverlayEntry]:
        from isaaclab_arena.assets.object_reference import ObjectReference

        assets: list[tuple[CollisionObject, bool]] = [
            (asset, asset.is_anchor or isinstance(asset, ObjectReference)) for asset in pool.objects
        ]
        assets.extend((asset, True) for asset in pool.collision_objects)
        entries: list[_OverlayEntry] = []
        seen: set[int] = set()
        for asset, is_static in assets:
            asset_id = id(asset)
            if asset_id in seen:
                continue
            seen.add(asset_id)
            kind = classify_entity_kind(asset)
            mesh = asset.get_collision_mesh() if object_uses_mesh_collision(asset, self._default_mode) else None
            entries.append(
                _OverlayEntry(
                    name=getattr(asset, "name", kind),
                    kind=kind,
                    color=KIND_COLORS.get(kind, KIND_COLORS["other"]),
                    asset=asset,
                    mesh=mesh,
                    is_static=is_static,
                )
            )
        return entries

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
        for index, entry in enumerate(self._entries):
            if not entry.is_static:
                continue

            prim_name = Tf.MakeValidIdentifier(f"{index}_{entry.name}")
            try:
                world_bbox = entry.asset.get_world_bounding_box()
                min_pt = tuple(float(v) for v in world_bbox.min_point[0].tolist())
                max_pt = tuple(float(v) for v in world_bbox.max_point[0].tolist())
                starts, ends = oriented_bbox_edge_segments(min_pt, max_pt, IDENTITY_POS, IDENTITY_XYZW)
                self._define_curve(stage, f"{STATIC_GEOMETRY_ROOT}/{prim_name}_bbox", starts, ends, entry, 0.01)
            except Exception as exc:  # noqa: BLE001 — visualizer must not crash the runner
                print(f"[placement_debug] skip static AABB for '{entry.name}': {exc}", flush=True)

            if entry.mesh is None:
                continue
            pose = _live_pose_for_asset(env, entry.asset)
            if pose is None:
                continue
            vertices = transform_trimesh_vertices(entry.mesh.vertices, pose.position_xyz, pose.rotation_xyzw)
            starts, ends = trimesh_edge_segments(vertices, entry.mesh.faces, max_edges=None)
            self._define_curve(stage, f"{STATIC_GEOMETRY_ROOT}/{prim_name}_mesh", starts, ends, entry, 0.006)
            print(f"[placement_debug] drew static '{entry.name}' once with {len(starts)} full edges", flush=True)

    @staticmethod
    def _define_curve(stage, path: str, starts, ends, entry: _OverlayEntry, width: float) -> None:
        """Define persistent linear curves for line segments."""
        from pxr import UsdGeom

        points = [point for segment in zip(starts, ends, strict=True) for point in segment]
        curves = UsdGeom.BasisCurves.Define(stage, path)
        curves.CreateTypeAttr(UsdGeom.Tokens.linear)
        curves.CreateCurveVertexCountsAttr([2] * len(starts))
        curves.CreatePointsAttr(points)
        curves.CreateWidthsAttr([width])
        curves.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        curves.CreateDisplayColorAttr([entry.color[:3]])
        curves.CreateDisplayOpacityAttr([entry.color[3]])


def classify_entity_kind(asset: CollisionObject) -> str:
    """Return the legend kind for a collision / placement asset."""
    from isaaclab_arena.assets.background import Background
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
    from isaaclab_arena.relations.background_collision_object import FixedCollisionObject

    if isinstance(asset, (FixedCollisionObject, Background)):
        return "background"
    if isinstance(asset, EmbodimentBase):
        return "embodiment"
    if isinstance(asset, ObjectReference):
        return "object_reference"
    if isinstance(asset, Object):
        return "object"
    return "other"


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
    try:
        scene_asset = scene[scene_key]
    except KeyError:
        scene_asset = None
    if scene_asset is not None and hasattr(scene_asset, "data") and hasattr(scene_asset.data, "root_pos_w"):
        pos = scene_asset.data.root_pos_w[0].detach().cpu().tolist()
        # Isaac Lab root_quat_w is (x, y, z, w) — same as Arena Pose.rotation_xyzw.
        quat_xyzw = scene_asset.data.root_quat_w[0].detach().cpu().tolist()
        return Pose(
            position_xyz=tuple(float(v) for v in pos),
            rotation_xyzw=tuple(float(v) for v in quat_xyzw),
        )

    return asset._get_initial_pose_as_pose() or Pose.identity()

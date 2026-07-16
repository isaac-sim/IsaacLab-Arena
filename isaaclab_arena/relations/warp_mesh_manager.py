# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Warp mesh management and greedy sphere decomposition for mesh-based collision."""

from __future__ import annotations

import numpy as np
import torch
import trimesh
from collections import defaultdict
from heapq import heappop, heappush
from typing import TYPE_CHECKING

import warp as wp

from isaaclab_arena.relations.warp_sdf_kernels import has_sdf_sentinel, sdf_sentinel_count

if TYPE_CHECKING:
    from isaaclab_arena.relations.collision_object import CollisionObject


def _mesh_content_hash(mesh: trimesh.Trimesh) -> int:
    """Content-based hash for a trimesh. Safe across GC cycles unlike id()."""
    return hash((mesh.vertices.tobytes(), mesh.faces.tobytes()))


def greedy_sphere_decomposition(
    mesh: trimesh.Trimesh,
    num_spheres: int = 30,
    sphere_radius: float = 0.01,
    n_candidates: int = 200,
    n_surface: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """Decompose a mesh into bounding spheres via greedy set-cover.

    Args:
        mesh: Input trimesh (must be watertight or convex-hull-repairable).
        num_spheres: Maximum number of output spheres.
        sphere_radius: Inflation added to tangent sphere radii (safety margin).
        n_candidates: Number of candidate sphere centers sampled.
        n_surface: Number of surface points for coverage tracking.
        seed: RNG seed for reproducible surface sampling.

    Returns:
        (K, 4) array of [cx, cy, cz, radius] in mesh-local frame. K <= num_spheres.
    """
    n_candidates = max(num_spheres, n_candidates)
    n_surface = max(n_candidates, n_surface)

    rng = np.random.default_rng(seed)
    points = trimesh.sample.sample_surface(mesh, n_surface, seed=rng)[0]
    cloud = trimesh.PointCloud(points)

    work_mesh = mesh if mesh.is_watertight else mesh.convex_hull
    candidates = points[:n_candidates]
    try:
        centers, radii = trimesh.proximity.max_tangent_sphere(work_mesh, candidates)
    except (IndexError, ValueError) as e:
        print(f"  [SphereDecomp] max_tangent_sphere failed ({e}), using uniform fallback — coverage may be poor")
        centers = candidates[:num_spheres]
        radii = np.full(len(centers), sphere_radius)
        return np.column_stack([centers, radii])

    radii = radii + sphere_radius

    max_radius = np.linalg.norm(mesh.extents) / 2
    valid = (radii <= max_radius) & np.isfinite(radii)
    centers, radii = centers[valid], radii[valid]

    if len(centers) == 0:
        print("  [SphereDecomp] All tangent spheres filtered (degenerate mesh?) — using uniform fallback")
        pts = points[:num_spheres]
        return np.column_stack([pts, np.full(len(pts), sphere_radius)])

    outgoing: dict[int, set[int]] = defaultdict(set)
    incoming: dict[int, set[int]] = defaultdict(set)
    for idx, (center, radius) in enumerate(zip(centers, radii)):
        covered = cloud.kdtree.query_ball_point(center, r=radius, eps=1e-6)
        for pt_idx in covered:
            outgoing[idx].add(pt_idx)
            incoming[pt_idx].add(idx)

    selected: list[int] = []
    queue: list[tuple[int, int]] = []
    for idx in outgoing:
        heappush(queue, (-len(outgoing[idx]), idx))

    while queue and len(selected) < num_spheres:
        neg_count, idx = heappop(queue)
        if len(outgoing[idx]) != -neg_count:
            heappush(queue, (-len(outgoing[idx]), idx))
            continue
        if neg_count == 0:
            break
        for pt_idx in list(outgoing[idx]):
            for other_idx in incoming[pt_idx]:
                outgoing[other_idx].discard(pt_idx)
        selected.append(idx)

    if not selected:
        print("  [SphereDecomp] Set-cover selected 0 spheres — using uniform fallback")
        pts = points[:num_spheres]
        return np.column_stack([pts, np.full(len(pts), sphere_radius)])

    return np.column_stack([centers[selected], radii[selected]])


class WarpMeshAndSphereCache:
    """Cache for Warp BVH meshes and sphere decompositions."""

    def __init__(
        self,
        num_spheres: int = 30,
        sphere_radius: float = 0.01,
        device: str = "cuda:0",
    ):
        self._num_spheres = num_spheres
        self._sphere_radius = sphere_radius
        self._device = device
        self._warp_mesh_cache: dict[tuple, wp.Mesh] = {}
        self._sphere_cache: dict[tuple, torch.Tensor] = {}
        self._trimesh_cache: dict[tuple, trimesh.Trimesh | None] = {}
        self._sentinel_warned: bool = False
        self._raw_open_mesh_warned: set[tuple] = set()

    def reset_sentinel_warning(self) -> None:
        """Re-arm for a new solve/validation pass."""
        self._sentinel_warned = False

    def warn_sdf_sentinel(self, sdf_values: torch.Tensor) -> None:
        """Warn (once per pass) if any query hit the no-face sentinel."""
        if self._sentinel_warned:
            return
        if has_sdf_sentinel(sdf_values):
            self._sentinel_warned = True
            n_bad = sdf_sentinel_count(sdf_values)
            print(
                f"  [MeshSDF] WARNING: {n_bad}/{len(sdf_values)} sphere queries returned sentinel SDF "
                "(no mesh face found). Collision detection may be incomplete for these points."
            )

    def get_collision_mesh(self, obj: CollisionObject) -> trimesh.Trimesh | None:
        """Return the cached collision mesh, extracting from USD on first access."""
        from isaaclab_arena.assets.object import Object

        if not isinstance(obj, Object) or obj.usd_path is None:
            return obj.get_collision_mesh()
        usd_path = obj.usd_path
        scale = tuple(obj.scale)
        key = (usd_path, scale)
        if key not in self._trimesh_cache:
            from isaaclab_arena.utils.usd_helpers import extract_trimesh_from_usd  # deferred: pxr import

            try:
                self._trimesh_cache[key] = extract_trimesh_from_usd(usd_path, scale)
            except ValueError as e:
                # Permanent: bad USD content, cache None to avoid re-parsing.
                print(f"  [WarpMeshAndSphereCache] Could not extract mesh for '{obj.name}': {e}")
                self._trimesh_cache[key] = None
            except OSError as e:
                # Transient: file I/O failure, don't cache so next call retries.
                print(f"  [WarpMeshAndSphereCache] Could not extract mesh for '{obj.name}': {e}")
                return None
        return self._trimesh_cache[key]

    @property
    def device(self) -> str:
        """Target Warp device string (e.g. 'cuda:0', 'cpu')."""
        return self._device

    def _cache_key(self, mesh: trimesh.Trimesh, obj: CollisionObject | None = None) -> tuple:
        """Compute cache key. Uses (usd_path, scale) for USD objects, content hash otherwise."""
        from isaaclab_arena.assets.object import Object

        repair_non_watertight = obj.repair_collision_mesh_non_watertight if obj is not None else True
        if isinstance(obj, Object) and obj.usd_path is not None:
            return (obj.usd_path, tuple(obj.scale), repair_non_watertight, self._num_spheres, self._sphere_radius)
        return (_mesh_content_hash(mesh), repair_non_watertight, self._num_spheres, self._sphere_radius)

    def get_warp_mesh(self, mesh: trimesh.Trimesh, obj: CollisionObject | None = None) -> wp.Mesh:
        """Get or create a Warp BVH mesh for SDF queries.

        Non-watertight meshes are replaced by their convex hull for reliable
        inside/outside signs unless ``repair_collision_mesh_non_watertight`` is False,
        which preserves concavities but may yield unreliable SDF signs.
        """
        key = self._cache_key(mesh, obj)
        if key not in self._warp_mesh_cache:
            repair_non_watertight = obj.repair_collision_mesh_non_watertight if obj is not None else True
            if not mesh.is_watertight and repair_non_watertight:
                name = obj.name if obj is not None else repr(mesh)
                print(
                    f"  [WarpMeshAndSphereCache] '{name}' mesh is not watertight — using convex hull (concavities will"
                    " be filled)"
                )
            if not mesh.is_watertight and not repair_non_watertight and key not in self._raw_open_mesh_warned:
                self._raw_open_mesh_warned.add(key)
                name = obj.name if obj is not None else repr(mesh)
                print(f"  [WarpMeshAndSphereCache] '{name}' raw mesh is not watertight; SDF signs may be unreliable.")
            work_mesh = mesh if mesh.is_watertight or not repair_non_watertight else mesh.convex_hull
            vertices = wp.array(np.asarray(work_mesh.vertices, dtype=np.float32), dtype=wp.vec3, device=self._device)
            indices = wp.array(
                np.asarray(work_mesh.faces, dtype=np.int32).flatten(), dtype=wp.int32, device=self._device
            )
            self._warp_mesh_cache[key] = wp.Mesh(points=vertices, indices=indices)
        return self._warp_mesh_cache[key]

    def get_query_spheres(self, mesh: trimesh.Trimesh, obj: CollisionObject | None = None) -> torch.Tensor:
        """Get or compute sphere decomposition as (K, 4) tensor [cx, cy, cz, radius]."""
        key = self._cache_key(mesh, obj)
        if key not in self._sphere_cache:
            spheres_np = greedy_sphere_decomposition(
                mesh,
                num_spheres=self._num_spheres,
                sphere_radius=self._sphere_radius,
            )
            self._sphere_cache[key] = torch.from_numpy(spheres_np).float()
        return self._sphere_cache[key]

# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Warp mesh management and greedy sphere decomposition for mesh-based collision."""

from __future__ import annotations

import copy

import numpy as np
import torch
from collections import defaultdict
from heapq import heappop, heappush
from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    import trimesh


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

    Based on the greedy_sample_mesh algorithm by Caelan Garrett (NVIDIA).
    Uses trimesh.proximity.max_tangent_sphere for candidate generation,
    then greedy selection maximising surface coverage.

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
    import trimesh

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


class WarpMeshManager:
    """Manages Warp mesh creation, caching, and sphere decomposition for collision objects.

    Caches results by content hash (in-memory trimesh) or (usd_path, scale) for USD objects.
    """

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

    def __deepcopy__(self, memo):
        """Deep copy, dropping lazy Warp caches (unpicklable C pointers); rebuilt on demand."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in ("_warp_mesh_cache", "_sphere_cache"):
                setattr(result, k, {})
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    @property
    def device(self) -> str:
        """Target Warp device string (e.g. 'cuda:0', 'cpu')."""
        return self._device

    def _cache_key(self, mesh: trimesh.Trimesh, obj=None) -> tuple:
        """Compute cache key. Uses (usd_path, scale) for USD objects, content hash otherwise."""
        usd_path = getattr(obj, "usd_path", None) if obj is not None else None
        if usd_path is not None:
            scale = tuple(getattr(obj, "scale", (1.0, 1.0, 1.0)))
            return (usd_path, scale, self._num_spheres, self._sphere_radius)
        return (_mesh_content_hash(mesh), self._num_spheres, self._sphere_radius)

    def get_warp_mesh(self, mesh: trimesh.Trimesh, obj=None) -> wp.Mesh:
        """Get or create a Warp BVH mesh for SDF queries.

        Non-watertight meshes are replaced by their convex hull so that
        mesh_query_point_sign_normal produces correct inside/outside signs.
        """
        key = self._cache_key(mesh, obj)
        if key not in self._warp_mesh_cache:
            if not mesh.is_watertight:
                name = getattr(obj, "name", None) or "unknown"
                print(
                    f"  [MeshManager] '{name}' mesh is not watertight — using convex hull (concavities will be filled)"
                )
            work_mesh = mesh if mesh.is_watertight else mesh.convex_hull
            vertices = wp.array(np.asarray(work_mesh.vertices, dtype=np.float32), dtype=wp.vec3, device=self._device)
            indices = wp.array(
                np.asarray(work_mesh.faces, dtype=np.int32).flatten(), dtype=wp.int32, device=self._device
            )
            self._warp_mesh_cache[key] = wp.Mesh(points=vertices, indices=indices)
        return self._warp_mesh_cache[key]

    def get_query_spheres(self, mesh: trimesh.Trimesh, obj=None) -> torch.Tensor:
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

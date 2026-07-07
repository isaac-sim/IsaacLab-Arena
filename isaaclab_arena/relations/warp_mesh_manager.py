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
    from isaaclab_arena.assets.object_base import ObjectBase


def _mesh_content_hash(mesh: trimesh.Trimesh) -> int:
    """Content-based hash for a trimesh. Safe across GC cycles unlike id()."""
    return hash((mesh.vertices.tobytes(), mesh.faces.tobytes()))


def _box_from_bounding_box(obj: ObjectBase) -> trimesh.Trimesh | None:
    """Return a watertight box matching obj's local axis-aligned bounding box, or None if unavailable."""
    bbox = obj.get_bounding_box()
    lo = bbox.min_point[0].tolist()
    hi = bbox.max_point[0].tolist()
    extents = [hi[i] - lo[i] for i in range(3)]
    if any(extent <= 0.0 for extent in extents):
        return None
    box = trimesh.creation.box(extents=extents)
    box.apply_translation([(hi[i] + lo[i]) / 2.0 for i in range(3)])
    return box


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
        self._cavity_trimesh_cache: dict[int, trimesh.Trimesh | None] = {}
        self._sentinel_warned: bool = False

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

    def get_collision_mesh(self, obj: ObjectBase) -> trimesh.Trimesh | None:
        """Return the cached collision mesh, extracting from USD on first access."""
        # ObjectBase doesn't guarantee usd_path; only Object subclasses set it.
        usd_path = getattr(obj, "usd_path", None)
        if usd_path is None:
            return obj.get_collision_mesh()
        scale = tuple(getattr(obj, "scale", (1.0, 1.0, 1.0)))
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

    def _cache_key(self, mesh: trimesh.Trimesh, obj: ObjectBase | None = None) -> tuple:
        """Compute cache key. Uses (usd_path, scale) for USD objects, content hash otherwise."""
        usd_path = getattr(obj, "usd_path", None) if obj is not None else None
        if usd_path is not None:
            scale = tuple(getattr(obj, "scale", (1.0, 1.0, 1.0)))
            return (usd_path, scale, self._num_spheres, self._sphere_radius)
        return (_mesh_content_hash(mesh), self._num_spheres, self._sphere_radius)

    def get_warp_mesh(self, mesh: trimesh.Trimesh, obj: ObjectBase | None = None) -> wp.Mesh:
        """Get or create a Warp BVH mesh for SDF queries.

        Non-watertight meshes are replaced by their convex hull to ensure
        correct inside/outside signs.
        """
        key = self._cache_key(mesh, obj)
        if key not in self._warp_mesh_cache:
            if not mesh.is_watertight:
                name = obj.name if obj is not None else repr(mesh)
                print(
                    f"  [WarpMeshAndSphereCache] '{name}' mesh is not watertight — using convex hull (concavities will"
                    " be filled)"
                )
            work_mesh = mesh if mesh.is_watertight else mesh.convex_hull
            vertices = wp.array(np.asarray(work_mesh.vertices, dtype=np.float32), dtype=wp.vec3, device=self._device)
            indices = wp.array(
                np.asarray(work_mesh.faces, dtype=np.int32).flatten(), dtype=wp.int32, device=self._device
            )
            self._warp_mesh_cache[key] = wp.Mesh(points=vertices, indices=indices)
        return self._warp_mesh_cache[key]

    def get_cavity_warp_mesh(self, obj: ObjectBase) -> wp.Mesh | None:
        """Get the Warp BVH mesh for obj's interior cavity, or None if none is available.

        Prefers an explicitly authored proxy (``get_cavity_mesh()``); otherwise derives a watertight
        cavity from the container's own collision mesh by capping its opening (``fill_holes``) — using
        the real geometry rather than the convex hull, which would fill the cavity. Returns None (so
        the ``In`` precondition fails loud) when no watertight cavity can be obtained. Cached so it
        never collides with the object's outer collision-mesh cache entry.
        """
        cavity = obj.get_cavity_mesh()
        if cavity is None:
            cavity = self._derive_cavity_trimesh(obj)
        if cavity is None:
            return None
        return self.get_warp_mesh(cavity, obj=None)

    def _derive_cavity_trimesh(self, obj: ObjectBase) -> trimesh.Trimesh | None:
        """Derive a watertight interior-cavity proxy from obj's collision mesh, or None.

        Caps the mesh's open boundaries (e.g. a bowl/bin's top opening) to make it watertight so its
        signed-distance field has a correct inside/outside sign. Returns None when the result is still
        non-watertight (fails loud upstream) rather than guessing a wrong cavity.
        """
        if id(obj) in self._cavity_trimesh_cache:
            return self._cavity_trimesh_cache[id(obj)]
        proxy = None
        mesh = self.get_collision_mesh(obj)
        if mesh is not None:
            capped = mesh.copy()
            capped.fill_holes()
            if capped.is_watertight:
                proxy = capped
        if proxy is None:
            # Many real container meshes (thick walls, handles, multiple openings) can't be capped
            # into a watertight interior. Fall back to a box the size of the object's bounding box:
            # coarse but watertight, and a good interior approximation for box-like containers
            # (bins/crates/pails). Author an explicit get_cavity_mesh() for concave shapes (e.g. bowls)
            # where the box would over-cover.
            proxy = _box_from_bounding_box(obj)
            if proxy is not None:
                print(
                    f"  [WarpMeshAndSphereCache] '{obj.name}' cavity: mesh not watertight-cappable; "
                    "using a bounding-box cavity approximation."
                )
        self._cavity_trimesh_cache[id(obj)] = proxy
        return proxy

    def get_query_spheres(self, mesh: trimesh.Trimesh, obj: ObjectBase | None = None) -> torch.Tensor:
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

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
from collections.abc import Sequence
from heapq import heappop, heappush
from typing import TYPE_CHECKING

import warp as wp

from isaaclab_arena.relations.warp_sdf_kernels import has_sdf_sentinel, sdf_sentinel_count

if TYPE_CHECKING:
    from isaaclab_arena.relations.collision_object import CollisionObject
    from isaaclab_arena.relations.placement_asset import PlaceableAsset


def _mesh_content_hash(mesh: trimesh.Trimesh) -> int:
    """Content-based hash for a trimesh. Safe across GC cycles unlike id()."""
    return hash((mesh.vertices.tobytes(), mesh.faces.tobytes()))


def _repair_non_watertight_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Repair connected components independently without filling gaps between them."""
    repaired_components = [
        component if component.is_watertight else component.convex_hull
        for component in mesh.split(only_watertight=False)
        if component.is_watertight or len(component.vertices) >= 4
    ]
    assert repaired_components, "Collision mesh has no volumetric components to repair."
    return trimesh.util.concatenate(repaired_components)


def greedy_sphere_decomposition(
    mesh: trimesh.Trimesh,
    num_spheres: int = 30,
    sphere_radius: float = 0.01,
    n_candidates: int = 200,
    n_surface: int = 1000,
    seed: int = 42,
    repair_non_watertight: bool = True,
) -> np.ndarray:
    """Decompose a mesh into bounding spheres via greedy set-cover.

    Args:
        mesh: Input trimesh (must be watertight or convex-hull-repairable).
        num_spheres: Maximum number of output spheres.
        sphere_radius: Inflation added to tangent sphere radii (safety margin).
        n_candidates: Number of candidate sphere centers sampled.
        n_surface: Number of surface points for coverage tracking.
        seed: RNG seed for reproducible surface sampling.
        repair_non_watertight: Repair each non-watertight connected component with its convex hull.

    Returns:
        (K, 4) array of [cx, cy, cz, radius] in mesh-local frame. K <= num_spheres.
    """
    n_candidates = max(num_spheres, n_candidates)
    n_surface = max(n_candidates, n_surface)

    rng = np.random.default_rng(seed)
    points = trimesh.sample.sample_surface(mesh, n_surface, seed=rng)[0]
    cloud = trimesh.PointCloud(points)

    work_mesh = mesh if mesh.is_watertight or not repair_non_watertight else _repair_non_watertight_mesh(mesh)
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

    def get_collision_mesh(
        self,
        obj: CollisionObject,
        excluded_prim_paths: Sequence[str] = (),
    ) -> trimesh.Trimesh | None:
        """Return the cached collision mesh, extracting from USD on first access."""
        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.relations.placement_asset import PlaceableAsset

        if isinstance(obj, PlaceableAsset) and not isinstance(obj, Object):
            assert not excluded_prim_paths, "USD prim exclusions do not apply to placeable collision components."
            key = ("components", id(obj))
            if key not in self._trimesh_cache:
                self._trimesh_cache[key] = self._combine_mesh_collision_components(obj)
            return self._trimesh_cache[key]

        if not isinstance(obj, Object) or obj.usd_path is None:
            assert not excluded_prim_paths, "USD prim exclusions require an Object with a usd_path."
            return obj.get_collision_mesh()
        usd_path = obj.usd_path
        scale = obj.scale
        exclusions = tuple(sorted(excluded_prim_paths))
        key = (usd_path, scale, exclusions)
        if key not in self._trimesh_cache:
            from isaaclab_arena.utils.usd_helpers import (  # deferred: pxr import
                AllCollisionMeshesExcludedError,
                NoCollisionMeshError,
                UnsupportedCollisionGeometryError,
                extract_trimesh_from_usd,
            )

            try:
                self._trimesh_cache[key] = extract_trimesh_from_usd(
                    usd_path,
                    scale,
                    excluded_prim_paths=exclusions,
                )
            except AllCollisionMeshesExcludedError:
                raise
            except UnsupportedCollisionGeometryError as e:
                print(f"  [WarpMeshAndSphereCache] Could not extract mesh for '{obj.name}': {e}")
                self._trimesh_cache[key] = None
            except NoCollisionMeshError:
                self._trimesh_cache[key] = None
            except ValueError as e:
                # Permanent: bad USD content, cache None to avoid re-parsing.
                print(f"  [WarpMeshAndSphereCache] Could not extract mesh for '{obj.name}': {e}")
                self._trimesh_cache[key] = None
            except OSError as e:
                # Transient: file I/O failure, don't cache so next call retries.
                print(f"  [WarpMeshAndSphereCache] Could not extract mesh for '{obj.name}': {e}")
                return None
        return self._trimesh_cache[key]

    @staticmethod
    def _combine_mesh_collision_components(obj: PlaceableAsset) -> trimesh.Trimesh | None:
        """Combine only mesh-backed components in the asset root frame."""
        meshes = WarpMeshAndSphereCache._collision_component_meshes(obj)
        return trimesh.util.concatenate(meshes) if meshes else None

    @staticmethod
    def _collision_component_meshes(obj: PlaceableAsset) -> list[trimesh.Trimesh]:
        """Return mesh-backed collision components transformed into the asset frame."""
        meshes: list[trimesh.Trimesh] = []
        for component in obj.get_collision_components():
            if component.mesh is None:
                continue
            mesh = component.mesh.copy()
            x, y, z, w = component.local_pose.rotation_xyzw
            transform = trimesh.transformations.quaternion_matrix((w, x, y, z))
            transform[:3, 3] = component.local_pose.position_xyz
            mesh.apply_transform(transform)
            meshes.append(mesh)
        return meshes

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
                    f"  [WarpMeshAndSphereCache] '{name}' mesh is not watertight — "
                    "repairing each connected component with its convex hull"
                )
            if not mesh.is_watertight and not repair_non_watertight and key not in self._raw_open_mesh_warned:
                self._raw_open_mesh_warned.add(key)
                name = obj.name if obj is not None else repr(mesh)
                print(f"  [WarpMeshAndSphereCache] '{name}' raw mesh is not watertight; SDF signs may be unreliable.")
            work_mesh = mesh if mesh.is_watertight or not repair_non_watertight else _repair_non_watertight_mesh(mesh)
            vertices = wp.array(np.asarray(work_mesh.vertices, dtype=np.float32), dtype=wp.vec3, device=self._device)
            indices = wp.array(
                np.asarray(work_mesh.faces, dtype=np.int32).flatten(), dtype=wp.int32, device=self._device
            )
            self._warp_mesh_cache[key] = wp.Mesh(points=vertices, indices=indices)
        return self._warp_mesh_cache[key]

    def get_query_spheres(self, mesh: trimesh.Trimesh, obj: CollisionObject | None = None) -> torch.Tensor:
        """Get or compute sphere decomposition as (K, 4) tensor [cx, cy, cz, radius]."""
        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.relations.placement_asset import PlaceableAsset

        if isinstance(obj, PlaceableAsset) and not isinstance(obj, Object):
            key = ("component_spheres", id(obj), self._num_spheres, self._sphere_radius)
            if key not in self._sphere_cache:
                self._sphere_cache[key] = torch.from_numpy(self._decompose_collision_components(obj)).float()
            return self._sphere_cache[key]

        key = self._cache_key(mesh, obj)
        if key not in self._sphere_cache:
            spheres_np = greedy_sphere_decomposition(
                mesh,
                num_spheres=self._num_spheres,
                sphere_radius=self._sphere_radius,
                repair_non_watertight=obj.repair_collision_mesh_non_watertight if obj is not None else True,
            )
            self._sphere_cache[key] = torch.from_numpy(spheres_np).float()
        return self._sphere_cache[key]

    def _decompose_collision_components(self, obj: PlaceableAsset) -> np.ndarray:
        """Build query spheres per collision component without flattening their geometry."""
        component_meshes = self._collision_component_meshes(obj)
        assert component_meshes, f"Collision object '{obj.name}' has no collision components."
        base_budget, remainder = divmod(self._num_spheres, len(component_meshes))
        spheres = [
            greedy_sphere_decomposition(
                component_mesh,
                num_spheres=max(1, base_budget + (component_idx < remainder)),
                sphere_radius=self._sphere_radius,
                repair_non_watertight=obj.repair_collision_mesh_non_watertight,
            )
            for component_idx, component_mesh in enumerate(component_meshes)
        ]
        return np.concatenate(spheres)

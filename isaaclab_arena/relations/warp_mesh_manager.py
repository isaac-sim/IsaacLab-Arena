# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Manages loading and caching of NVIDIA Warp meshes for GPU-accelerated SDF queries."""

from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    import trimesh

    from isaaclab_arena.assets.object_base import ObjectBase


_WARP_MESH_CACHE: dict[tuple[str, tuple[float, float, float], str], wp.Mesh] = {}
"""Module-level cache for wp.Mesh objects, keyed by (usd_path, scale, device_str)."""

_VERTEX_TENSOR_CACHE: dict[tuple[str, tuple[float, float, float], str, int], torch.Tensor] = {}
"""Module-level cache for subsampled vertex tensors on a specific device."""

DEFAULT_MAX_QUERY_VERTICES = 128
"""Maximum number of vertices used for SDF queries.  Meshes with more
vertices are uniformly subsampled down to this count.  The full mesh is
still used as the SDF *target* (via ``wp.Mesh``); only the *query* point
set is reduced."""


class WarpMeshManager:
    """Loads trimesh collision meshes and converts them to ``wp.Mesh`` on a target device.

    Each ``wp.Mesh`` wraps a BVH built on the GPU, enabling fast signed-distance
    queries via ``wp.mesh_query_point``.  Meshes are cached globally so that
    repeated calls for the same (usd_path, scale, device) return the same object.

    For objects that do not provide a collision mesh, a box fallback is created
    from the object's axis-aligned bounding box.
    """

    def __init__(self, device: str = "cuda:0", max_query_vertices: int = DEFAULT_MAX_QUERY_VERTICES):
        self._device = device
        self._max_query_vertices = max_query_vertices

    @property
    def device(self) -> str:
        return self._device

    def get_warp_mesh(self, obj: ObjectBase) -> wp.Mesh:
        """Return a ``wp.Mesh`` for *obj*, creating and caching it if necessary."""
        cache_key = self._cache_key(obj)
        if cache_key not in _WARP_MESH_CACHE:
            _WARP_MESH_CACHE[cache_key] = self._build_warp_mesh(obj)
        return _WARP_MESH_CACHE[cache_key]

    def get_query_vertices(self, obj: ObjectBase, device: torch.device | str = "cuda:0") -> torch.Tensor:
        """Return a cached (M, 3) float32 tensor of subsampled local-frame vertices.

        The tensor is created once, subsampled to ``max_query_vertices``, moved
        to *device*, and cached for all subsequent calls.
        """
        device_str = str(device)
        vkey = (*self._cache_key(obj), hash(device_str))
        if vkey not in _VERTEX_TENSOR_CACHE:
            verts_np = self._get_subsampled_vertices(obj)
            _VERTEX_TENSOR_CACHE[vkey] = torch.tensor(verts_np, dtype=torch.float32, device=device)
        return _VERTEX_TENSOR_CACHE[vkey]

    def get_mesh_vertices(self, obj: ObjectBase) -> np.ndarray:
        """Return the local-frame vertices (N, 3) for *obj*'s collision mesh.

        These are the raw vertices before any world-space transform.  The caller
        is responsible for translating / rotating them to world coordinates.
        """
        tm = self._get_trimesh(obj)
        return np.asarray(tm.vertices, dtype=np.float32)

    def _get_subsampled_vertices(self, obj: ObjectBase) -> np.ndarray:
        """Return at most ``max_query_vertices`` uniformly spaced vertices."""
        all_verts = self.get_mesh_vertices(obj)
        n = len(all_verts)
        if n <= self._max_query_vertices:
            return all_verts
        indices = np.linspace(0, n - 1, self._max_query_vertices, dtype=int)
        return all_verts[indices]

    def _cache_key(self, obj: ObjectBase) -> tuple[str, tuple[float, float, float], str]:
        usd_path = getattr(obj, "usd_path", None) or obj.name
        scale = getattr(obj, "scale", (1.0, 1.0, 1.0))
        return (usd_path, scale, self._device)

    def _get_trimesh(self, obj: ObjectBase) -> trimesh.Trimesh:
        """Get or create a trimesh for *obj*, falling back to an AABB box."""
        import trimesh as _trimesh

        mesh = obj.get_collision_mesh()
        if mesh is not None:
            return mesh
        bbox = obj.get_bounding_box()
        size = bbox.size[0].tolist()
        center = bbox.center[0].tolist()
        box = _trimesh.creation.box(extents=size)
        box.apply_translation(center)
        return box

    def _build_warp_mesh(self, obj: ObjectBase) -> wp.Mesh:
        tm = self._get_trimesh(obj)
        vertices = np.asarray(tm.vertices, dtype=np.float32)
        faces = np.asarray(tm.faces.flatten(), dtype=np.int32)

        wp_points = wp.array(vertices, dtype=wp.vec3, device=self._device)
        wp_indices = wp.array(faces, dtype=wp.int32, device=self._device)
        return wp.Mesh(points=wp_points, indices=wp_indices)

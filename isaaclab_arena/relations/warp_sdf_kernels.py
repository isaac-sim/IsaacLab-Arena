# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Warp SDF kernels and PyTorch autograd bridge for mesh-based collision loss."""

from __future__ import annotations

import torch

import warp as wp


@wp.kernel
def _sdf_query_kernel(
    mesh_id: wp.uint64,
    query_points: wp.array(dtype=wp.vec3),
    sdf_out: wp.array(dtype=wp.float32),
    grad_out: wp.array(dtype=wp.vec3),
):
    """Query signed distance and gradient for each point against a Warp mesh.

    Points must be in mesh-local frame. Sign convention: negative = inside mesh.
    Points with no enclosing face write a large sentinel value (~1e6).
    """
    tid = wp.tid()
    p = query_points[tid]

    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)

    found = wp.mesh_query_point_sign_normal(mesh_id, p, 1.0e6, sign, face_index, face_u, face_v)

    if found:
        closest = wp.mesh_eval_position(mesh_id, face_index, face_u, face_v)
        diff = p - closest
        dist = wp.length(diff)
        sdf_out[tid] = sign * dist
        if dist > 1.0e-8:
            grad_out[tid] = sign * wp.normalize(diff)
        else:
            grad_out[tid] = wp.vec3(0.0, 0.0, 0.0)
    else:
        sdf_out[tid] = float(1.0e6)
        grad_out[tid] = wp.vec3(0.0, 0.0, 0.0)


class _MeshSDFFunction(torch.autograd.Function):
    """Autograd bridge for single-mesh SDF queries."""

    @staticmethod
    def forward(ctx, points: torch.Tensor, mesh: wp.Mesh) -> torch.Tensor:
        """Compute SDF values for query points against a Warp mesh.

        Args:
            points: (N, 3) float32 tensor of query positions (must be on same device as mesh).
            mesh: Warp Mesh with BVH built.

        Returns:
            (N,) tensor of signed distance values (negative = inside).
        """
        device = points.device
        n = points.shape[0]
        wp_device = str(device)

        wp_points = wp.from_torch(points.contiguous(), dtype=wp.vec3)
        sdf_wp = wp.zeros(n, dtype=wp.float32, device=wp_device)
        grad_wp = wp.zeros(n, dtype=wp.vec3, device=wp_device)

        wp.launch(
            kernel=_sdf_query_kernel,
            dim=n,
            inputs=[mesh.id, wp_points, sdf_wp, grad_wp],
            device=wp_device,
        )

        sdf_torch = wp.to_torch(sdf_wp)
        grad_torch = wp.to_torch(grad_wp).reshape(n, 3)

        ctx.save_for_backward(grad_torch)
        return sdf_torch

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (grad_sdf,) = ctx.saved_tensors
        grad_points = grad_output.unsqueeze(-1) * grad_sdf
        return grad_points, None


def mesh_sdf(points: torch.Tensor, warp_mesh: wp.Mesh) -> torch.Tensor:
    """Differentiable signed distance query.

    Args:
        points: (N, 3) query points on the same device as the mesh.
        warp_mesh: Warp Mesh object with BVH.

    Returns:
        (N,) signed distance values. Negative = penetrating.
    """
    return _MeshSDFFunction.apply(points, warp_mesh)


# The kernels above write 1.0e6 when the BVH finds no enclosing face; 1e5 catches
# these while staying safely above any realistic SDF magnitude.
_SDF_SENTINEL = 1.0e5


def has_sdf_sentinel(sdf_values: torch.Tensor) -> bool:
    """True when any query hit the no-face sentinel, so the collision result is unreliable."""
    return bool((sdf_values >= _SDF_SENTINEL).any())


def sdf_sentinel_count(sdf_values: torch.Tensor) -> int:
    """Number of queries that hit the no-face sentinel."""
    return int((sdf_values >= _SDF_SENTINEL).sum().item())


def clamp_sdf_sentinel(sdf_values: torch.Tensor) -> torch.Tensor:
    """Replace sentinel SDF values with 0 so no-face hits produce a constant penalty with zero gradient."""
    return torch.where(sdf_values >= _SDF_SENTINEL, torch.zeros_like(sdf_values), sdf_values)


# ---------------------------------------------------------------------------
# Multi-mesh kernel: query multiple meshes in a single launch
# ---------------------------------------------------------------------------


@wp.kernel
def _multi_mesh_sdf_kernel(
    mesh_ids: wp.array(dtype=wp.uint64),
    mesh_indices: wp.array(dtype=wp.int32),
    query_points: wp.array(dtype=wp.vec3),
    sdf_out: wp.array(dtype=wp.float32),
    grad_out: wp.array(dtype=wp.vec3),
):
    """Query signed distance per point against its assigned mesh (indexed by mesh_indices).

    Points must be in mesh-local frame. Sign convention: negative = inside mesh.
    Points with no enclosing face write a large sentinel value (~1e6).
    """
    tid = wp.tid()
    p = query_points[tid]
    mesh_id = mesh_ids[mesh_indices[tid]]

    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)

    found = wp.mesh_query_point_sign_normal(mesh_id, p, 1.0e6, sign, face_index, face_u, face_v)

    if found:
        closest = wp.mesh_eval_position(mesh_id, face_index, face_u, face_v)
        diff = p - closest
        dist = wp.length(diff)
        sdf_out[tid] = sign * dist
        if dist > 1.0e-8:
            grad_out[tid] = sign * wp.normalize(diff)
        else:
            grad_out[tid] = wp.vec3(0.0, 0.0, 0.0)
    else:
        sdf_out[tid] = float(1.0e6)
        grad_out[tid] = wp.vec3(0.0, 0.0, 0.0)


class _MultiMeshSDFFunction(torch.autograd.Function):
    """Autograd bridge for multi-mesh SDF queries (each point addressed to its own mesh)."""

    @staticmethod
    def forward(
        ctx,
        points: torch.Tensor,
        mesh_id_array: wp.array,
        mesh_indices: wp.array,
    ) -> torch.Tensor:
        """Compute SDF values for query points, each against its own target mesh.

        Args:
            points: (N, 3) float32 tensor of query positions.
            mesh_id_array: Warp array of uint64 mesh IDs.
            mesh_indices: Warp array of int32 indices into mesh_id_array (one per point).

        Returns:
            (N,) tensor of signed distance values.
        """
        device = points.device
        n = points.shape[0]
        wp_device = str(device)

        wp_points = wp.from_torch(points.contiguous(), dtype=wp.vec3)
        sdf_wp = wp.zeros(n, dtype=wp.float32, device=wp_device)
        grad_wp = wp.zeros(n, dtype=wp.vec3, device=wp_device)

        wp.launch(
            kernel=_multi_mesh_sdf_kernel,
            dim=n,
            inputs=[mesh_id_array, mesh_indices, wp_points, sdf_wp, grad_wp],
            device=wp_device,
        )

        sdf_torch = wp.to_torch(sdf_wp)
        grad_torch = wp.to_torch(grad_wp).reshape(n, 3)

        ctx.save_for_backward(grad_torch)
        return sdf_torch

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (grad_sdf,) = ctx.saved_tensors
        grad_points = grad_output.unsqueeze(-1) * grad_sdf
        return grad_points, None, None


def multi_mesh_sdf(
    points: torch.Tensor,
    mesh_id_array: wp.array,
    mesh_indices: wp.array,
) -> torch.Tensor:
    """Differentiable multi-mesh SDF query. Single kernel launch for all points.

    Args:
        points: (N, 3) query points.
        mesh_id_array: Warp uint64 array of mesh IDs (one per unique target mesh).
        mesh_indices: Warp int32 array (N,) mapping each point to its target mesh index.

    Returns:
        (N,) signed distance values. Negative = penetrating.
    """
    return _MultiMeshSDFFunction.apply(points, mesh_id_array, mesh_indices)

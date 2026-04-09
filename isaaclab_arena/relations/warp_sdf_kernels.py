# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Warp SDF kernels and PyTorch autograd bridge for differentiable mesh collision.

The main entry point is :func:`mesh_sdf_loss`, which computes a differentiable
penetration loss between query points and a ``wp.Mesh``.  Gradients flow back
to the query-point positions via a custom ``torch.autograd.Function``.
"""

from __future__ import annotations

import torch
import warp as wp

wp.init()


@wp.kernel
def _sdf_kernel(
    mesh_id: wp.uint64,
    query_points: wp.array(dtype=wp.vec3),
    sdf_out: wp.array(dtype=wp.float32),
    closest_points: wp.array(dtype=wp.vec3),
):
    """For each query point, compute signed distance to *mesh_id* and the closest surface point."""
    tid = wp.tid()
    p = query_points[tid]
    q = wp.mesh_query_point(mesh_id, p, 1.0e6)
    if q.result:
        cp = wp.mesh_eval_position(mesh_id, q.face, q.u, q.v)
        d = wp.length(p - cp)
        sdf_out[tid] = q.sign * d
        closest_points[tid] = cp
    else:
        sdf_out[tid] = 1.0e6
        closest_points[tid] = p


class _MeshSDFFunction(torch.autograd.Function):
    """Custom autograd function bridging Warp SDF queries into PyTorch's graph.

    Forward: runs the Warp SDF kernel, returning signed distances.
    Backward: analytical gradient  ``d(sdf)/d(query_point) = (point - closest) / dist``.
    """

    @staticmethod
    def forward(
        ctx,
        query_points_torch: torch.Tensor,
        mesh: wp.Mesh,
        device_str: str,
    ) -> torch.Tensor:
        """Compute signed distances from *query_points_torch* to *mesh*.

        Args:
            query_points_torch: (N, 3) float32 tensor of query positions.
            mesh: Target ``wp.Mesh``.
            device_str: Warp device string (e.g. ``"cuda:0"``).

        Returns:
            (N,) float32 tensor of signed distances (negative = inside mesh).
        """
        n = query_points_torch.shape[0]

        wp_points = wp.from_torch(query_points_torch.contiguous(), dtype=wp.vec3)
        wp_sdf = wp.zeros(n, dtype=wp.float32, device=device_str)
        wp_closest = wp.zeros(n, dtype=wp.vec3, device=device_str)

        wp.launch(
            kernel=_sdf_kernel,
            dim=n,
            inputs=[mesh.id, wp_points, wp_sdf, wp_closest],
            device=device_str,
        )

        sdf_torch = wp.to_torch(wp_sdf)
        closest_torch = wp.to_torch(wp_closest).reshape(n, 3)

        ctx.save_for_backward(query_points_torch, closest_torch, sdf_torch)
        return sdf_torch

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        query_pts, closest_pts, sdf_vals = ctx.saved_tensors

        direction = query_pts - closest_pts
        dist = sdf_vals.abs().clamp(min=1e-8).unsqueeze(-1)
        grad_sdf_wrt_pts = direction / dist

        grad_pts = grad_output.unsqueeze(-1) * grad_sdf_wrt_pts
        return grad_pts, None, None


def mesh_sdf(
    query_points: torch.Tensor,
    mesh: wp.Mesh,
    device: str = "cuda:0",
) -> torch.Tensor:
    """Differentiable signed-distance query: (N, 3) points → (N,) SDF values.

    Negative SDF means the point is inside the mesh.  Gradients flow back to
    *query_points* via the analytical SDF gradient.
    """
    return _MeshSDFFunction.apply(query_points, mesh, device)


def mesh_penetration_loss(
    query_points: torch.Tensor,
    mesh: wp.Mesh,
    device: str = "cuda:0",
) -> torch.Tensor:
    """Scalar penetration loss: ``ReLU(-sdf).mean()`` over all query points.

    Returns zero when no point penetrates the mesh.
    """
    sdf = mesh_sdf(query_points, mesh, device)
    return torch.nn.functional.relu(-sdf).mean()

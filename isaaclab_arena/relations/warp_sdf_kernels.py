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
    """Query signed distance and gradient for each point against a Warp mesh."""
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
    """Autograd bridge: forward computes SDF values, backward propagates via analytical gradients."""

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
        """Backprop through SDF: dL/dpoints = dL/dsdf * dsdf/dpoints."""
        (grad_sdf,) = ctx.saved_tensors
        # grad_output: (N,), grad_sdf: (N, 3) -- analytical SDF gradient
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


def sphere_penetration_loss(
    sphere_centers_world: torch.Tensor,
    sphere_radii: torch.Tensor,
    warp_mesh: wp.Mesh,
    clearance_m: float = 0.0,
) -> torch.Tensor:
    """Compute ReLU penetration loss for spheres against a mesh SDF.

    Loss per sphere = ReLU(effective_radius - sdf).
    Total loss = mean over all spheres.

    Args:
        sphere_centers_world: (K, 3) world-space sphere centers.
        sphere_radii: (K,) sphere radii.
        warp_mesh: Target Warp mesh to check against.
        clearance_m: Additional clearance added to radii.

    Returns:
        Scalar loss tensor (differentiable w.r.t. sphere_centers_world).
    """
    sdf_values = mesh_sdf(sphere_centers_world, warp_mesh)

    # SDF returns 1e6 for points where no mesh face was found (degenerate query).
    # These read as "collision-free" (relu(r - 1e6) = 0) which is silently wrong.
    _SDF_SENTINEL = 1.0e5
    if not hasattr(sphere_penetration_loss, "_warned_sentinel"):
        sphere_penetration_loss._warned_sentinel = False  # type: ignore[attr-defined]
    if not sphere_penetration_loss._warned_sentinel and (sdf_values >= _SDF_SENTINEL).any():  # type: ignore[attr-defined]
        sphere_penetration_loss._warned_sentinel = True  # type: ignore[attr-defined]
        n_bad = int((sdf_values >= _SDF_SENTINEL).sum().item())
        print(
            f"  [MeshSDF] WARNING: {n_bad}/{len(sdf_values)} sphere queries returned sentinel SDF "
            "(no mesh face found). Collision detection may be incomplete for these points."
        )

    effective_radii = sphere_radii + clearance_m
    penetration = torch.relu(effective_radii - sdf_values)
    return penetration.mean()

# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Measure asset bounds and rigid-body point velocities with batched Torch tensors."""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

Bounds = Sequence[Sequence[float]] | torch.Tensor


def point_velocity(
    linear_velocity: torch.Tensor,
    angular_velocity: torch.Tensor,
    point_position: torch.Tensor,
    reference_position: torch.Tensor,
) -> torch.Tensor:
    """Transport a rigid body's velocity from a reference point to another point.

    Args:
        linear_velocity: Velocity at the reference point, commonly the center of mass.
        angular_velocity: Angular velocity of the same rigid body.
        point_position: Position at which to evaluate the linear velocity.
        reference_position: Position where the supplied linear velocity is measured.

    Returns:
        Linear velocity at the requested point. All inputs use one common frame
        and broadcastable batches of three-component vectors.
    """
    angular_velocity, offset = torch.broadcast_tensors(angular_velocity, point_position - reference_position)
    return linear_velocity + torch.linalg.cross(angular_velocity, offset, dim=-1)


def _bounds(bounds: Bounds, reference: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(bounds, dtype=reference.dtype, device=reference.device)
    assert values.shape[-2:] == (2, 3), "Bounds must end in local minimum and maximum XYZ vectors."
    return values


def _rotation(T_P_O: torch.Tensor) -> torch.Tensor:
    """Return R_P_O from an XYZ+XYZW pose, allowing quaternion normalization drift."""
    assert T_P_O.shape[-1] == 7, "A pose contains XYZ translation followed by an XYZW quaternion."
    quaternion = torch.nn.functional.normalize(T_P_O[..., 3:], dim=-1)
    x, y, z, w = quaternion.unbind(-1)
    rows = (
        1 - 2 * (y * y + z * z),
        2 * (x * y - z * w),
        2 * (x * z + y * w),
        2 * (x * y + z * w),
        1 - 2 * (x * x + z * z),
        2 * (y * z - x * w),
        2 * (x * z - y * w),
        2 * (y * z + x * w),
        1 - 2 * (x * x + y * y),
    )
    return torch.stack(rows, dim=-1).reshape(*T_P_O.shape[:-1], 3, 3)


def box_corners(bounds: Bounds, reference: torch.Tensor) -> torch.Tensor:
    """Return the eight local corners of Blender-authored minimum/maximum bounds.

    Args:
        bounds: Minimum/maximum XYZ vectors with shape (..., 2, 3).
        reference: Tensor supplying the output floating dtype and device.

    Returns:
        Local corner coordinates with shape (..., 8, 3).
    """
    values = _bounds(bounds, reference)
    selectors = torch.tensor(
        ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)),
        dtype=torch.bool,
        device=reference.device,
    )
    return torch.where(selectors, values[..., 1, None, :], values[..., 0, None, :])


def box_contained(T_P_O: torch.Tensor, bounds: Bounds, region_bounds: Bounds) -> torch.Tensor:
    """Require every oriented object corner inside an axis-aligned parent-frame box.

    Args:
        T_P_O: Batched pose mapping object frame O into parent frame P, in XYZ+XYZW order.
        bounds: Object-local bounds of shape (..., 2, 3), broadcastable over pose batches.
        region_bounds: Parent-frame acceptance bounds, broadcastable over pose batches.

    Returns:
        One containment boolean per pose; touching the region boundary is accepted.
    """
    corners = box_corners(bounds, T_P_O)
    points = torch.matmul(corners, _rotation(T_P_O).transpose(-1, -2)) + T_P_O[..., None, :3]
    region = _bounds(region_bounds, T_P_O)
    return (
        ((points >= region[..., 0, None, :] - 1e-6) & (points <= region[..., 1, None, :] + 1e-6))
        .all(dim=-1)
        .all(dim=-1)
    )


def box_contained_on_support(
    T_P_O: torch.Tensor, bounds: Bounds, region_bounds: Bounds, *, floor_allowance_m: float
) -> torch.Tensor:
    """Check containment with a contact allowance only at the region's lower Z face.

    Args:
        T_P_O: Batched XYZ+XYZW object poses in the region's parent frame.
        bounds: Object-local minimum and maximum XYZ vectors.
        region_bounds: Parent-frame interior bounds; the lower Z face is the support surface.
        floor_allowance_m: Nonnegative penetration allowed at that support, in meters.

    Returns:
        One boolean per pose, preserving the ordinary side and ceiling tolerances.
        The input region is not modified. This does not establish support contact or rest.
    """
    assert math.isfinite(floor_allowance_m) and floor_allowance_m >= 0.0
    region = _bounds(region_bounds, T_P_O).clone()
    region[..., 0, 2] -= floor_allowance_m
    return box_contained(T_P_O, bounds, region)


def box_overlaps(T_P_O: torch.Tensor, bounds: Bounds, region_bounds: Bounds) -> torch.Tensor:
    """Test oriented-object/axis-aligned-region overlap with the complete separating-axis test.

    Args:
        T_P_O: Batched pose mapping object frame O into parent frame P, in XYZ+XYZW order.
        bounds: Object-local bounds of shape (..., 2, 3), broadcastable over pose batches.
        region_bounds: Parent-frame region bounds, broadcastable over pose batches.

    Returns:
        One overlap boolean per pose. Partial overlap and boundary contact count;
        the only tolerance is one micrometer for floating-point roundoff.
    """
    object_bounds = _bounds(bounds, T_P_O)
    region = _bounds(region_bounds, T_P_O)
    R_P_O = _rotation(T_P_O)
    local_center = object_bounds.mean(dim=-2)
    object_center = torch.matmul(R_P_O, local_center.unsqueeze(-1)).squeeze(-1) + T_P_O[..., :3]
    object_half = (object_bounds[..., 1, :] - object_bounds[..., 0, :]) / 2
    region_center = region.mean(dim=-2)
    region_half = (region[..., 1, :] - region[..., 0, :]) / 2

    # Rows enumerate candidate separating axes in P. Face normals alone miss
    # separation between skew edges, so all nine edge cross-products are needed.
    object_axes = R_P_O.transpose(-1, -2)
    region_axes = torch.eye(3, dtype=T_P_O.dtype, device=T_P_O.device).expand_as(object_axes)
    edge_axes = torch.linalg.cross(object_axes[..., :, None, :], region_axes[..., None, :, :], dim=-1)
    edge_axes = edge_axes.flatten(start_dim=-3, end_dim=-2)
    axes = torch.cat((object_axes, region_axes, edge_axes), dim=-2)
    axes = torch.nn.functional.normalize(axes, dim=-1)
    local_axes = torch.matmul(axes, R_P_O)
    object_radius = (local_axes.abs() * object_half.unsqueeze(-2)).sum(dim=-1)
    region_radius = (axes.abs() * region_half.unsqueeze(-2)).sum(dim=-1)
    distance = ((object_center - region_center)[..., None, :] * axes).sum(dim=-1).abs()
    return (distance <= object_radius + region_radius + 1e-6).all(dim=-1)


def sphere_overlaps_cylinder(
    T_P_O: torch.Tensor,
    bounds: Bounds,
    x_range: Sequence[float],
    center_yz: Sequence[float],
    radius: float,
) -> torch.Tensor:
    """Conservatively detect small-object overlap with a finite X-axis cylinder.

    Args:
        T_P_O: Batched pose mapping object frame O into parent frame P, in XYZ+XYZW order.
        bounds: Broadcastable object-local bounds used to circumscribe each object with a sphere.
        x_range: Cylinder endpoints in the parent frame, in increasing order.
        center_yz: Cylinder-axis position in the parent frame's YZ plane.
        radius: Cylinder interior radius in meters.

    Returns:
        One overlap boolean per pose. The enclosing sphere can retain a small
        object as "inside" slightly after a corner leaves; it cannot declare a
        partially contained object clear. Use this for debris, not elongated tools.
    """
    assert len(x_range) == 2 and x_range[0] <= x_range[1], "Cylinder endpoints must be ordered."
    assert len(center_yz) == 2 and radius > 0, "A cylinder requires a YZ center and positive radius."
    object_bounds = _bounds(bounds, T_P_O)
    local_center = object_bounds.mean(dim=-2)
    center = torch.matmul(_rotation(T_P_O), local_center.unsqueeze(-1)).squeeze(-1) + T_P_O[..., :3]
    sphere_radius = torch.linalg.vector_norm((object_bounds[..., 1, :] - object_bounds[..., 0, :]) / 2, dim=-1)
    axial_distance = torch.maximum(x_range[0] - center[..., 0], center[..., 0] - x_range[1]).clamp_min(0)
    radial_distance = (
        torch.linalg.vector_norm(center[..., 1:] - T_P_O.new_tensor(center_yz), dim=-1) - radius
    ).clamp_min(0)
    return axial_distance.square() + radial_distance.square() <= (sphere_radius + 1e-6).square()

# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Ordered cable-winding geometry from the Isaac Cap task."""

from __future__ import annotations

import math
import torch

_ROUTE_RADIAL_CUTOFF = 0.05
_ROUTE_AXIAL_CUTOFF = 0.01475
_ROUTE_COMPLETION_WINDING = 2.6
_ROUTE_MAXIMUM_COMPLETION_WINDING = 2.0 * math.pi + 0.25
_ROUTE_MAXIMUM_LOCAL_CABLE_LENGTH = 0.25


def cable_route_success_from_geometry(
    cable_points_w: torch.Tensor,
    peg_positions_w: torch.Tensor,
    *,
    route_peg_indices: tuple[int, ...] = (0, 1),
    route_directions: tuple[float, ...] = (-1.0, 1.0),
) -> torch.Tensor:
    """Return whether ordered cable points complete every configured peg route.

    A route direction of zero accepts either winding direction. Positive and
    negative values retain the medium task's ordered directional semantics.

    Args:
        cable_points_w: Ordered cable point positions with shape ``(N, S, 3)``.
        peg_positions_w: Peg positions with shape ``(N, P, 3)``.
        route_peg_indices: Indices of the pegs that define the route.
        route_directions: Required winding sign per route peg; zero accepts either.

    Returns:
        Boolean success tensor with shape ``(N,)``.
    """
    assert (
        cable_points_w.ndim == 3 and cable_points_w.shape[-1] == 3
    ), f"Expected cable points with shape (N, S, 3), got {tuple(cable_points_w.shape)}."
    assert (
        peg_positions_w.ndim == 3 and peg_positions_w.shape[-1] == 3
    ), f"Expected peg positions with shape (N, P, 3), got {tuple(peg_positions_w.shape)}."
    assert (
        cable_points_w.shape[0] == peg_positions_w.shape[0]
    ), "Cable points and peg positions must contain the same number of environments."
    assert cable_points_w.shape[1] >= 2, "At least two ordered cable points are required."

    finite_geometry = torch.isfinite(cable_points_w).all(dim=(1, 2)) & torch.isfinite(peg_positions_w).all(dim=(1, 2))
    safe_cable_points = torch.where(finite_geometry[:, None, None], cable_points_w, torch.zeros_like(cable_points_w))
    safe_peg_positions = torch.where(finite_geometry[:, None, None], peg_positions_w, torch.zeros_like(peg_positions_w))

    relative_xy = safe_cable_points[:, None, :, :2] - safe_peg_positions[:, :, None, :2]
    local_points = torch.linalg.vector_norm(relative_xy, dim=-1) <= _ROUTE_RADIAL_CUTOFF
    relative_z = safe_cable_points[:, None, :, 2] - safe_peg_positions[:, :, None, 2]
    local_points &= relative_z.abs() <= _ROUTE_AXIAL_CUTOFF

    angle = torch.atan2(relative_xy[..., 1], relative_xy[..., 0])
    angle_delta = angle[..., 1:] - angle[..., :-1]
    angle_delta = torch.atan2(torch.sin(angle_delta), torch.cos(angle_delta))
    local_edges = local_points[..., :-1] & local_points[..., 1:]
    clockwise_winding = -torch.where(local_edges, angle_delta, 0.0).sum(dim=-1)

    previous_local_edges = torch.zeros_like(local_edges)
    previous_local_edges[..., 1:] = local_edges[..., :-1]
    local_span_count = (local_edges & ~previous_local_edges).sum(dim=-1)
    edge_lengths = torch.linalg.vector_norm(safe_cable_points[:, 1:] - safe_cable_points[:, :-1], dim=-1)
    local_cable_length = torch.where(local_edges, edge_lengths[:, None, :], 0.0).sum(dim=-1)
    geometrically_eligible = (local_span_count == 1) & (local_cable_length <= _ROUTE_MAXIMUM_LOCAL_CABLE_LENGTH)

    assert len(route_peg_indices) == len(route_directions) > 0, "Each route peg must have one direction."
    assert all(
        0 <= index < peg_positions_w.shape[1] for index in route_peg_indices
    ), "Route peg index is outside the supplied peg positions."
    route_peg_indices_tensor = torch.tensor(route_peg_indices, device=cable_points_w.device).unsqueeze(0)
    route_peg_indices_tensor = route_peg_indices_tensor.expand(cable_points_w.shape[0], -1)
    route_directions_tensor = torch.tensor(
        route_directions,
        device=cable_points_w.device,
        dtype=cable_points_w.dtype,
    ).unsqueeze(0)
    route_directions_tensor = route_directions_tensor.expand(cable_points_w.shape[0], -1)
    selected_winding = torch.gather(clockwise_winding, 1, route_peg_indices_tensor)
    directed_winding = torch.where(
        route_directions_tensor == 0.0,
        selected_winding.abs(),
        selected_winding * route_directions_tensor,
    )
    route_geometry_valid = torch.gather(geometrically_eligible, 1, route_peg_indices_tensor)
    route_steps_complete = (
        route_geometry_valid
        & (directed_winding >= _ROUTE_COMPLETION_WINDING)
        & (directed_winding <= _ROUTE_MAXIMUM_COMPLETION_WINDING)
    )
    return finite_geometry & route_steps_complete.all(dim=1)

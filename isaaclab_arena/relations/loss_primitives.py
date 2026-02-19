# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch


def single_boundary_linear_loss(
    value: torch.Tensor,
    boundary: torch.Tensor | float,
    slope: float = 1.0,
    penalty_side: str = "greater",
) -> torch.Tensor:
    """ReLU-style loss: zero at boundary, linear growth for violations.

    This is an unbounded loss function ideal for optimization since it provides
    constant gradient regardless of distance from the boundary.

    Args:
        value: Measured value (tensor for gradient flow).
        boundary: The boundary threshold value (can be tensor or float).
        slope: Gradient magnitude when violating (default: 1.0).
               Loss increases by `slope` per unit of violation.
        penalty_side: Which side to penalize:
            - 'greater': Penalize if value > boundary
            - 'less': Penalize if value < boundary

    Returns:
        Loss value (unbounded):
        - 0 when constraint satisfied
        - slope * |violation| when constraint violated

    Examples:
        >>> # Penalize if x > 0.5 (should be to the left of boundary)
        >>> loss = single_boundary_linear_loss(x, 0.5, slope=10.0, penalty_side='greater')

        >>> # Penalize if x < 0.5 (should be to the right of boundary)
        >>> loss = single_boundary_linear_loss(x, 0.5, slope=10.0, penalty_side='less')
    """
    assert isinstance(value, torch.Tensor), f"value must be a torch.Tensor, got {type(value)}"

    # Convert boundary to tensor if needed
    if not isinstance(boundary, torch.Tensor):
        boundary = torch.tensor(boundary, dtype=value.dtype, device=value.device)

    zero = torch.tensor(0.0, dtype=value.dtype, device=value.device)

    if penalty_side == "greater":
        # Penalize if value > boundary
        violation = torch.maximum(zero, value - boundary)
    elif penalty_side == "less":
        # Penalize if value < boundary
        violation = torch.maximum(zero, boundary - value)
    else:
        raise ValueError(f"penalty_side must be 'greater' or 'less', got '{penalty_side}'")

    # Linear penalty: slope * violation (ReLU-style)
    loss = slope * violation

    return loss


def linear_band_loss(
    value: torch.Tensor,
    lower_bound: torch.Tensor | float,
    upper_bound: torch.Tensor | float,
    slope: float = 1.0,
) -> torch.Tensor:
    """ReLU-style band loss: zero inside band, linear growth outside.

    Loss is zero within [lower_bound, upper_bound] and grows linearly
    with distance outside the band.

    Args:
        value: Measured value (tensor for gradient flow).
        lower_bound: Lower bound threshold (can be tensor or float).
        upper_bound: Upper bound threshold (can be tensor or float).
        slope: Gradient magnitude when violating (default: 1.0).

    Returns:
        Loss value (unbounded):
        - 0 when value is within [lower_bound, upper_bound]
        - slope * |violation| when outside the band
    """
    loss_lower = single_boundary_linear_loss(value, lower_bound, slope, penalty_side="less")
    loss_upper = single_boundary_linear_loss(value, upper_bound, slope, penalty_side="greater")

    return loss_lower + loss_upper


def single_point_linear_loss(
    value: torch.Tensor,
    target: torch.Tensor | float,
    slope: float = 1.0,
) -> torch.Tensor:
    """V-shaped loss centered at target value.

    Loss is zero only at the exact target, increasing linearly in both directions.
    This is useful when you want to constrain a value to a specific point rather
    than a range/band.

    Args:
        value: Measured value (tensor for gradient flow).
        target: The target value where loss is zero (can be tensor or float).
        slope: Gradient magnitude (default: 1.0).
               Loss increases by `slope` per unit of distance from target.

    Returns:
        Loss value (unbounded):
        - 0 when value == target
        - slope * |value - target| otherwise

    Examples:
        >>> # Target x = 1.0, penalize deviation in either direction
        >>> loss = single_point_linear_loss(x, 1.0, slope=10.0)
    """
    assert isinstance(value, torch.Tensor), f"value must be a torch.Tensor, got {type(value)}"

    # Convert target to tensor if needed (float is allowed per type hint)
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target, dtype=value.dtype, device=value.device)

    return slope * torch.abs(value - target)


def interval_overlap_axis_loss(
    a_min: torch.Tensor,
    a_max: torch.Tensor,
    b_min: torch.Tensor,
    b_max: torch.Tensor,
) -> torch.Tensor:
    """ReLU-style interval overlap: zero when separated, linear in overlap length otherwise.

    Computes the intersection length of two intervals [a_min, a_max] and
    [b_min, b_max] on one axis. Unbounded and differentiable for optimization.

    Args:
        a_min: First interval minimum on this axis (tensor for gradient flow).
        a_max: First interval maximum on this axis.
        b_min: Second interval minimum on this axis.
        b_max: Second interval maximum on this axis.

    Returns:
        Loss value (unbounded):
        - 0 when the two intervals do not overlap
        - overlap length (min(a_max, b_max) - max(a_min, b_min)) when they overlap
    """
    overlap_high = torch.minimum(a_max, b_max)
    overlap_low = torch.maximum(a_min, b_min)
    return single_boundary_linear_loss(
        overlap_high, overlap_low, slope=1.0, penalty_side="greater"
    )

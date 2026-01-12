# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

from isaaclab_arena.relations.relation_loss import (
    linear_band_loss,
    single_boundary_linear_loss,
    single_point_linear_loss,
)


def test_single_boundary_linear_loss_no_violation_returns_zero_greater():
    """Test that loss is zero when constraint is satisfied (greater side)."""
    value = torch.tensor(0.03)
    boundary = 0.05
    loss = single_boundary_linear_loss(value, boundary, slope=1.0, penalty_side="greater")
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_single_boundary_linear_loss_no_violation_returns_zero_less():
    """Test that loss is zero when constraint is satisfied (less side)."""
    value = torch.tensor(0.07)
    boundary = 0.05
    loss = single_boundary_linear_loss(value, boundary, slope=1.0, penalty_side="less")
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_single_boundary_linear_loss_violation_gives_linear_loss_greater():
    """Test that violations give linear loss for 'greater' side."""
    boundary = 0.05
    slope = 10.0
    violation = 0.02  # 2cm violation

    value = torch.tensor(boundary + violation)
    loss = single_boundary_linear_loss(value, boundary, slope=slope, penalty_side="greater")

    expected = slope * violation
    assert torch.isclose(loss, torch.tensor(expected), rtol=1e-5)


def test_single_boundary_linear_loss_violation_gives_linear_loss_less():
    """Test that violations give linear loss for 'less' side."""
    boundary = 0.05
    slope = 10.0
    violation = 0.02  # 2cm violation

    value = torch.tensor(boundary - violation)
    loss = single_boundary_linear_loss(value, boundary, slope=slope, penalty_side="less")

    expected = slope * violation
    assert torch.isclose(loss, torch.tensor(expected), rtol=1e-5)


def test_single_boundary_linear_loss_greater_side_penalizes_correctly():
    """Test that 'greater' side penalizes values above boundary."""
    boundary = 0.05

    # Value below boundary: no penalty
    value_below = torch.tensor(0.03)
    loss_below = single_boundary_linear_loss(value_below, boundary, penalty_side="greater")
    assert torch.isclose(loss_below, torch.tensor(0.0), atol=1e-6)

    # Value above boundary: penalty
    value_above = torch.tensor(0.07)
    loss_above = single_boundary_linear_loss(value_above, boundary, penalty_side="greater")
    assert loss_above > 0.0


def test_single_boundary_linear_loss_less_side_penalizes_correctly():
    """Test that 'less' side penalizes values below boundary."""
    boundary = 0.05

    # Value above boundary: no penalty
    value_above = torch.tensor(0.07)
    loss_above = single_boundary_linear_loss(value_above, boundary, penalty_side="less")
    assert torch.isclose(loss_above, torch.tensor(0.0), atol=1e-6)

    # Value below boundary: penalty
    value_below = torch.tensor(0.03)
    loss_below = single_boundary_linear_loss(value_below, boundary, penalty_side="less")
    assert loss_below > 0.0


def test_linear_band_loss_inside_band_returns_zero():
    """Test that loss is zero when value is within bounds."""
    lower_bound = 0.04
    upper_bound = 0.06

    # Test center of band
    value_center = torch.tensor(0.05)
    loss_center = linear_band_loss(value_center, lower_bound, upper_bound)
    assert torch.isclose(loss_center, torch.tensor(0.0), atol=1e-6)

    # Test near lower bound (inside)
    value_lower = torch.tensor(0.041)
    loss_lower = linear_band_loss(value_lower, lower_bound, upper_bound)
    assert torch.isclose(loss_lower, torch.tensor(0.0), atol=1e-6)

    # Test near upper bound (inside)
    value_upper = torch.tensor(0.059)
    loss_upper = linear_band_loss(value_upper, lower_bound, upper_bound)
    assert torch.isclose(loss_upper, torch.tensor(0.0), atol=1e-6)


def test_linear_band_loss_below_lower_bound():
    """Test penalty for values below lower bound."""
    lower_bound = 0.04
    upper_bound = 0.06
    slope = 10.0
    violation = 0.02  # 2cm below lower bound

    value = torch.tensor(lower_bound - violation)
    loss = linear_band_loss(value, lower_bound, upper_bound, slope=slope)

    expected = slope * violation
    assert torch.isclose(loss, torch.tensor(expected), rtol=1e-5)


def test_linear_band_loss_above_upper_bound():
    """Test penalty for values above upper bound."""
    lower_bound = 0.04
    upper_bound = 0.06
    slope = 10.0
    violation = 0.02  # 2cm above upper bound

    value = torch.tensor(upper_bound + violation)
    loss = linear_band_loss(value, lower_bound, upper_bound, slope=slope)

    expected = slope * violation
    assert torch.isclose(loss, torch.tensor(expected), rtol=1e-5)


def test_single_point_linear_loss_at_target_returns_zero():
    """Test that loss is zero when value equals target."""
    target = 0.5
    value = torch.tensor(target)
    loss = single_point_linear_loss(value, target, slope=10.0)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_single_point_linear_loss_above_target():
    """Test linear penalty for values above target."""
    target = 0.5
    slope = 10.0
    deviation = 0.1  # 10cm above target

    value = torch.tensor(target + deviation)
    loss = single_point_linear_loss(value, target, slope=slope)

    expected = slope * deviation
    assert torch.isclose(loss, torch.tensor(expected), rtol=1e-5)


def test_single_point_linear_loss_below_target():
    """Test linear penalty for values below target."""
    target = 0.5
    slope = 10.0
    deviation = 0.1  # 10cm below target

    value = torch.tensor(target - deviation)
    loss = single_point_linear_loss(value, target, slope=slope)

    expected = slope * deviation
    assert torch.isclose(loss, torch.tensor(expected), rtol=1e-5)


def test_single_point_linear_loss_symmetric():
    """Test that loss is symmetric around target."""
    target = 0.5
    slope = 10.0
    deviation = 0.15

    value_above = torch.tensor(target + deviation)
    value_below = torch.tensor(target - deviation)

    loss_above = single_point_linear_loss(value_above, target, slope=slope)
    loss_below = single_point_linear_loss(value_below, target, slope=slope)

    assert torch.isclose(loss_above, loss_below, rtol=1e-5)

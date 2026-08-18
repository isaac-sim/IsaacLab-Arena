# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the released-placement condition."""

from __future__ import annotations

import math
import torch

import pytest

from isaaclab_arena.tasks.predicates.released_placement import (
    released_contact_condition,
    released_placement_condition,
    update_consecutive_true_counts,
)
from isaaclab_arena_environments.etl_pnp_maple_table_environment import EtlPnpMapleTableEnvironmentCfg


def _condition(**overrides) -> torch.Tensor:
    values = {
        "object_pose_w": torch.tensor([[0.01, 0.0, 0.09, 0.0, 0.0, 0.0, 1.0]]),
        "object_velocity_w": torch.zeros((1, 6)),
        "destination_position_w": torch.zeros((1, 3)),
        "gripper_joint_position": torch.tensor([0.0]),
        "end_effector_position_w": torch.tensor([[0.01, 0.0, 0.25]]),
        "destination_contact_force": torch.tensor([0.2]),
        "max_horizontal_offset": 0.02,
        "min_vertical_offset": 0.06,
        "max_vertical_offset": 0.115,
        "max_axis_tilt": 0.5,
        "max_linear_speed": 0.03,
        "max_angular_speed": 0.2,
        "max_open_joint_position": 0.1,
        "min_end_effector_distance": 0.15,
        "min_contact_force": 0.1,
    }
    values.update(overrides)
    return released_placement_condition(**values)


def test_released_stable_centered_placement_passes():
    assert _condition().item()


def test_object_mass_override_must_be_positive_when_enabled():
    assert EtlPnpMapleTableEnvironmentCfg().pick_up_object_mass_kg is None
    with pytest.raises(AssertionError):
        EtlPnpMapleTableEnvironmentCfg(pick_up_object_mass_kg=0.0)


def _contact_condition(**overrides) -> torch.Tensor:
    values = {
        "object_position_w": torch.tensor([[0.08, 0.0, 0.01]]),
        "object_velocity_w": torch.zeros((1, 6)),
        "destination_position_w": torch.zeros((1, 3)),
        "gripper_joint_position": torch.tensor([0.0]),
        "end_effector_position_w": torch.tensor([[0.08, 0.0, 0.20]]),
        "destination_contact_force": torch.tensor([1.01]),
        "max_horizontal_offset": 0.12,
        "max_linear_speed": 0.1,
        "max_angular_speed": 0.2,
        "max_open_joint_position": 0.1,
        "min_end_effector_distance": 0.15,
        "min_contact_force": 1.0,
    }
    values.update(overrides)
    return released_contact_condition(**values)


def test_released_contact_accepts_stable_tipped_object_without_prescribing_height():
    assert _contact_condition().item()


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"object_position_w": torch.tensor([[0.121, 0.0, 0.01]])}, "too far from destination"),
        ({"object_velocity_w": torch.tensor([[0.101, 0.0, 0.0, 0.0, 0.0, 0.0]])}, "moving"),
        ({"object_velocity_w": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.201]])}, "rotating"),
        ({"gripper_joint_position": torch.tensor([0.101])}, "gripper closed"),
        ({"end_effector_position_w": torch.tensor([[0.08, 0.0, 0.159]])}, "not withdrawn"),
        ({"destination_contact_force": torch.tensor([1.0])}, "Arena force threshold is strict"),
    ],
)
def test_invalid_released_contact_is_rejected(overrides, reason):
    assert not _contact_condition(**overrides).item(), reason


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"object_pose_w": torch.tensor([[0.021, 0.0, 0.09, 0.0, 0.0, 0.0, 1.0]])}, "outside"),
        ({"object_pose_w": torch.tensor([[0.0, 0.0, 0.12, 0.0, 0.0, 0.0, 1.0]])}, "too high"),
        ({"object_velocity_w": torch.tensor([[0.031, 0.0, 0.0, 0.0, 0.0, 0.0]])}, "moving"),
        ({"object_velocity_w": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.21]])}, "rotating"),
        ({"gripper_joint_position": torch.tensor([0.11])}, "gripper closed"),
        ({"end_effector_position_w": torch.tensor([[0.01, 0.0, 0.20]])}, "not retreated"),
        ({"destination_contact_force": torch.tensor([0.09])}, "no support contact"),
        (
            {"object_pose_w": torch.tensor([[0.0, 0.0, 0.09, math.sin(0.26), 0.0, 0.0, math.cos(0.26)]])},
            "tilted",
        ),
    ],
)
def test_invalid_released_placement_is_rejected(overrides, reason):
    assert not _condition(**overrides).item(), reason


def test_released_placement_requires_consecutive_steps():
    counts = torch.zeros(2, dtype=torch.long)
    for _ in range(7):
        counts = update_consecutive_true_counts(counts, torch.tensor([True, True]))
    assert counts.tolist() == [7, 7]

    counts = update_consecutive_true_counts(counts, torch.tensor([True, False]))
    assert counts.tolist() == [8, 0]

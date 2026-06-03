# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for the torch-only SE(3) geometry (no pytorch3d / Isaac Sim required)."""

import torch

from isaaclab_arena_datagen.geometry.rotation import (
    Rotation,
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_quaternion,
    quaternion_to_matrix,
)
from isaaclab_arena_datagen.geometry.transform_se3 import TransformSE3


def _random_unit_quats(n: int) -> torch.Tensor:
    """Random scalar-first (w, x, y, z) unit quaternions with non-negative w."""
    torch.manual_seed(0)
    q = torch.nn.functional.normalize(torch.randn(n, 4), dim=-1)
    return torch.where(q[..., :1] < 0, -q, q)


def test_quaternion_to_matrix_is_orthonormal():
    q = _random_unit_quats(16)
    R = quaternion_to_matrix(q)
    eye = torch.eye(3).expand(16, 3, 3)
    assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-5)
    assert torch.allclose(torch.linalg.det(R), torch.ones(16), atol=1e-5)


def test_quaternion_matrix_round_trip():
    q = _random_unit_quats(16)
    assert torch.allclose(q, matrix_to_quaternion(quaternion_to_matrix(q)), atol=1e-5)


def test_known_90deg_about_z():
    q = torch.tensor([[0.70710678, 0.0, 0.0, 0.70710678]])  # wxyz, +90deg about z
    expected = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert torch.allclose(quaternion_to_matrix(q)[0], expected, atol=1e-5)


def test_axis_angle_round_trip():
    torch.manual_seed(1)
    aa = torch.randn(8, 3) * 0.7
    R = axis_angle_to_matrix(aa)
    # round-trip through axis-angle reproduces the same rotation matrix
    assert torch.allclose(R, axis_angle_to_matrix(matrix_to_axis_angle(R)), atol=1e-5)


def test_transform_apply_inverse_compose():
    Rz = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    M = torch.eye(4)
    M[:3, :3] = Rz
    M[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
    T = TransformSE3.from_matrix(M)

    # (1,0,0) rotated +90 about z -> (0,1,0), then translated by (1,2,3)
    assert torch.allclose(T.apply(torch.tensor([1.0, 0.0, 0.0])), torch.tensor([1.0, 3.0, 3.0]), atol=1e-5)
    # T @ T^-1 == identity
    assert torch.allclose((T @ T.inverse()).to_matrix()[0, 0], torch.eye(4), atol=1e-5)
    # from_matrix / to_matrix round trip
    assert torch.allclose(T.to_matrix()[0, 0], M, atol=1e-5)


def test_rotation_create_identity():
    R = Rotation.create_identity(batch_size=4, device=torch.device("cpu"))
    assert torch.allclose(R.R, torch.eye(3).expand(4, 1, 3, 3), atol=1e-6)

# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""3D rotation representation in SO(3).

Self-contained, torch-only port of the nvblox_next ``Rotation`` class. The
quaternion / axis-angle conversions are implemented directly in PyTorch so this
module has **no** dependency on ``pytorch3d`` (or on Isaac Sim), which keeps the
geometry layer importable and unit-testable without a simulator.

Quaternions follow the ``[w, x, y, z]`` (scalar-first) convention, matching the
original nvblox_next API.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Pure-torch rotation conversions (drop-in replacements for the pytorch3d
# helpers the original code relied on). Quaternions are scalar-first (w,x,y,z).
# ---------------------------------------------------------------------------


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert scalar-first ``[w, x, y, z]`` quaternions to rotation matrices.

    Args:
        quaternions: Tensor of shape ``[..., 4]``.

    Returns:
        Rotation matrices of shape ``[..., 3, 3]``.
    """
    quaternions = torch.nn.functional.normalize(quaternions, dim=-1)
    w, x, y, z = torch.unbind(quaternions, dim=-1)
    tx, ty, tz = 2.0 * x, 2.0 * y, 2.0 * z
    twx, twy, twz = tx * w, ty * w, tz * w
    txx, txy, txz = tx * x, ty * x, tz * x
    tyy, tyz, tzz = ty * y, tz * y, tz * z
    matrix = torch.stack(
        [
            1.0 - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            1.0 - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            1.0 - (txx + tyy),
        ],
        dim=-1,
    )
    return matrix.reshape(quaternions.shape[:-1] + (3, 3))


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to scalar-first ``[w, x, y, z]`` quaternions.

    Args:
        matrix: Rotation matrices of shape ``[..., 3, 3]``.

    Returns:
        Quaternions of shape ``[..., 4]`` with a non-negative real part.
    """
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected matrix [..., 3, 3], got {matrix.shape}")
    batch_dim = matrix.shape[:-2]
    m = matrix.reshape(batch_dim + (9,))
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(m, dim=-1)

    q_abs = torch.stack(
        [
            1.0 + m00 + m11 + m22,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        ],
        dim=-1,
    )
    q_abs = torch.sqrt(torch.clamp(q_abs, min=0.0))

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )
    flr = torch.tensor(0.1, dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    best = q_abs.argmax(dim=-1)
    best = best[..., None, None].expand(batch_dim + (1, 4))
    out = torch.gather(quat_candidates, -2, best).squeeze(-2)
    # Canonicalize to a non-negative real (w) component.
    out = torch.where(out[..., :1] < 0, -out, out)
    return out


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle vectors to rotation matrices via Rodrigues' formula.

    Args:
        axis_angle: Tensor of shape ``[..., 3]`` whose direction is the rotation
            axis and whose magnitude is the angle in radians.

    Returns:
        Rotation matrices of shape ``[..., 3, 3]``.
    """
    angle = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    eps = torch.finfo(axis_angle.dtype).eps
    axis = axis_angle / torch.clamp(angle, min=eps)
    x, y, z = torch.unbind(axis, dim=-1)
    zeros = torch.zeros_like(x)
    k = torch.stack([zeros, -z, y, z, zeros, -x, -y, x, zeros], dim=-1).reshape(axis_angle.shape[:-1] + (3, 3))
    eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    sin = torch.sin(angle)[..., None]
    cos = torch.cos(angle)[..., None]
    return eye + sin * k + (1.0 - cos) * (k @ k)


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to axis-angle vectors.

    Args:
        matrix: Rotation matrices of shape ``[..., 3, 3]``.

    Returns:
        Axis-angle vectors of shape ``[..., 3]``.
    """
    quat = matrix_to_quaternion(matrix)
    w = torch.clamp(quat[..., 0], -1.0, 1.0)
    angles = 2.0 * torch.acos(w)
    sin_half = torch.sqrt(torch.clamp(1.0 - w * w, min=0.0))
    eps = torch.finfo(matrix.dtype).eps
    scale = torch.where(sin_half > eps, angles / torch.clamp(sin_half, min=eps), 2.0 * torch.ones_like(angles))
    return quat[..., 1:] * scale[..., None]


@dataclass
class Rotation:
    """3D rotation representation in SO(3).

    Represents SO(3) rotations stored as 3x3 rotation matrices.
    Supports batching and per-element rotations.

    Convention: R_<target>_from_<source> rotates vectors from source to target:
        v_target = R @ v_source

    Attributes:
        R: Rotation matrix of shape [B, N, 3, 3] where B is batch size and N is number of
            elements per batch item. N=1 for single-transform use, N>1 for per-element
            (e.g. surface elements of an object).
    """

    # [B, N, 3, 3]; N=1 for single-transform use, N>1 for per-element
    # (e.g. surface elements of an object)
    R: torch.Tensor

    def __post_init__(self) -> None:
        """Validate shape and properties of rotation matrix after initialization."""
        # [3, 3] -> [1, 3, 3] -> [1, 1, 3, 3]
        if self.R.ndim == 2:
            self.R = self.R.unsqueeze(0).unsqueeze(0)
        # [B, 3, 3] -> [B, 1, 3, 3]
        elif self.R.ndim == 3:
            self.R = self.R.unsqueeze(1)
        # [B, N, 3, 3] -> keep as is

        # Guarded: torch.compile traces __post_init__ and cannot access attributes
        # on the partially-constructed proxy object, causing InternalTorchDynamoError.
        if not torch.compiler.is_compiling():
            assert self.R.ndim == 4, f"R must be [B, N, 3, 3], got shape {self.R.shape}"
            assert self.R.shape[-2:] == (3, 3), f"R must be [B, N, 3, 3], got shape {self.R.shape}"

            # Check the rotation is in the SO(3) group:
            # - det(R) = 1
            # - R @ R^T = I
            assert torch.allclose(torch.linalg.det(self.R), torch.ones_like(self.R[..., 0, 0]), atol=1e-4)
            assert torch.allclose(
                self.R @ self.R.transpose(-2, -1),
                torch.diag_embed(torch.ones_like(self.R[..., 0])),
                atol=1e-4,
            )

    @property
    def num_elements(self) -> int:
        """Get number of elements per batch item."""
        return int(self.R.shape[1])

    @classmethod
    def create_identity(cls, batch_size: int | None = None, device: torch.device | None = None) -> Rotation:
        """Create identity rotation(s).

        Args:
            batch_size: Batch size (None for single rotation).
            device: Torch device (defaults to CUDA if available, else CPU).

        Returns:
            Identity rotation(s).
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = batch_size if batch_size else 1
        R = torch.diag_embed(torch.ones(batch_size, 3, device=device))
        return cls(R=R)

    @classmethod
    def from_quaternion(cls, q: torch.Tensor) -> Rotation:
        """Create rotation from quaternion(s).

        Args:
            q: Quaternion(s) in [w, x, y, z] format, shape [4] or [B, 4].

        Returns:
            Rotation.
        """
        R = quaternion_to_matrix(q)
        return cls(R=R)

    def to_quaternion(self) -> torch.Tensor:
        """Convert to quaternion(s) in [w, x, y, z] format.

        Returns:
            Quaternion(s), shape [B, N, 4].
        """
        return matrix_to_quaternion(self.R)

    @classmethod
    def from_axis_angle(cls, axis_angle: torch.Tensor) -> Rotation:
        """Create rotation from axis-angle representation.

        Args:
            axis_angle: Axis-angle vector where direction is rotation axis and
                       magnitude is angle in radians. Shape [3] or [B, 3].

        Returns:
            Rotation.
        """
        R = axis_angle_to_matrix(axis_angle)
        return cls(R=R)

    def to_axis_angle(self) -> torch.Tensor:
        """Convert to axis-angle representation.

        Returns:
            Axis-angle vector [B, N, 3] where direction is rotation axis and
            magnitude is angle in radians.
        """
        return matrix_to_axis_angle(self.R)

    def to_matrix(self) -> torch.Tensor:
        """Get raw rotation matrix tensor.

        Returns:
            Rotation matrix [B, N, 3, 3].
        """
        return self.R

    def to(self, device: torch.device | str) -> Rotation:
        """Move rotation to device.

        Args:
            device: Target device.

        Returns:
            Rotation on target device.
        """
        return Rotation(R=self.R.to(device))

    def transpose(self) -> Rotation:
        """Get transpose of rotation: R^T.

        For rotation matrices, R^T = R^-1 (transpose equals inverse).
        Each individual rotation matrix in the batch is transposed independently.

        Returns:
            Transposed rotation.
        """
        return Rotation(R=self.R.transpose(-2, -1))

    def __matmul__(self, other: Rotation) -> Rotation:
        """Compose rotations using @ operator.

        Example:
            R_C_from_A = R_C_from_B @ R_B_from_A

        Args:
            other: Another Rotation (R_B_from_A when self is R_C_from_B).

        Returns:
            Composed rotation R_C_from_A.
        """
        return Rotation(R=self.R @ other.R)

    def apply(self, vectors: torch.Tensor) -> torch.Tensor:
        """Apply rotation: v_out[i] = R[i] @ v_in[i] for each batch element i.

        Args:
            vectors: Vectors to rotate:
                     - [3]: single vector (requires batch_size=1, num_elements=1)
                     - [B, 3]: one vector per batch (B must match batch_size, num_elements=1)
                     - [B, N, 3]: N vectors per batch (B must match batch_size)

        Returns:
            Rotated points, same shape as input.
        """
        if not torch.compiler.is_compiling():
            assert vectors.shape[-1] == 3, f"Expected [..., 3], got {vectors.shape}"
            assert vectors.ndim in (1, 2, 3), f"Expected [3], [B, 3], or [B, N, 3], got {vectors.shape}"

        R_t = self.R.transpose(-2, -1)  # [B, N, 3, 3]

        if vectors.ndim == 1:  # [3]
            if not torch.compiler.is_compiling():
                assert self.batch_size == 1, f"[3] input requires batch_size=1, got {self.batch_size}"
            return vectors @ R_t[0, 0]

        # [B, 3] or [B, N, 3]
        if not torch.compiler.is_compiling():
            assert (
                vectors.shape[0] == self.batch_size
            ), f"Batch size mismatch: vectors {vectors.shape[0]}, R {self.batch_size}"

        if vectors.ndim == 2:  # [B, 3]
            return (vectors.unsqueeze(1) @ R_t[:, 0]).squeeze(1)

        # [B, N, 3]: unsqueeze for matmul, then squeeze back to preserve N
        return (vectors.unsqueeze(-2) @ R_t).squeeze(-2)

    @property
    def batch_size(self) -> int:
        """Get batch size."""
        return int(self.R.shape[0])

    @property
    def device(self) -> torch.device:
        """Get device of rotation matrix."""
        return self.R.device

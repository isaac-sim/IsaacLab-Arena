# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Rigid transform representation in SE(3)."""

from __future__ import annotations

import torch
from dataclasses import dataclass

from isaaclab_arena_datagen.geometry.rotation import Rotation
from isaaclab_arena_datagen.geometry.translation import Translation


@dataclass
class TransformSE3:
    """Rigid transform (rotation + translation) in SE(3).

    Represents rigid body transformations stored as a rotation and a translation.
    Supports batching of transforms.

    Convention: T_<target>_from_<source> transforms points from source to target:
        p_target = R @ p_source + t

    Attributes:
        rotation: Component that rotates a point to target frame from the source frame
        translation: The origin of source frame in the target frame
    """

    rotation: Rotation
    translation: Translation

    def __post_init__(self) -> None:
        """Validate transform after initialization."""
        # Guarded: torch.compile traces __post_init__ and cannot access attributes
        # on the partially-constructed proxy object, causing InternalTorchDynamoError.
        if not torch.compiler.is_compiling():
            self.validate_batch_size()
            self.validate_num_elements()
            self.validate_device()

    def validate_batch_size(self) -> None:
        """Check if the batch size of the rotation and translation are the same."""
        assert (
            self.rotation.batch_size == self.translation.batch_size
        ), f"Batch size mismatch: rotation[{self.rotation.batch_size}], translation[{self.translation.batch_size}]"

    def validate_num_elements(self) -> None:
        """Check if the number of elements of the rotation and translation are the same."""
        assert self.rotation.num_elements == self.translation.num_elements, (
            f"Num elements mismatch: rotation[{self.rotation.num_elements}], "
            f"translation[{self.translation.num_elements}]"
        )

    def validate_device(self) -> None:
        """Check if the device of the rotation and translation are the same."""
        assert (
            self.rotation.device == self.translation.device
        ), f"Device mismatch: rotation[{self.rotation.device}], translation[{self.translation.device}]"

    @classmethod
    def collate(
        cls,
        batch: list[TransformSE3],
        device: torch.device | None = None,
    ) -> TransformSE3:
        """Collate a list of TransformSE3 into a single batched transform.

        Args:
            batch: List of transforms to collate, each with batch size 1.
            device: Optional target device to move data to.

        Returns:
            Single batched transform with batch size equal to len(batch).
        """
        R = torch.cat([item.rotation.R for item in batch], dim=0)
        t = torch.cat([item.translation.t for item in batch], dim=0)
        if device is not None:
            R = R.to(device)
            t = t.to(device)
        return cls(rotation=Rotation(R=R), translation=Translation(t=t))

    @classmethod
    def create_identity(
        cls,
        batch_size: int | None = None,
        device: torch.device | None = None,
    ) -> TransformSE3:
        """Create identity transform(s).

        Args:
            batch_size: If provided, creates a batch of identity transforms.
            device: Torch device (defaults to CUDA if available, else CPU).

        Returns:
            Identity transform(s).
        """
        rotation = Rotation.create_identity(batch_size=batch_size, device=device)
        translation = Translation.create_zero(batch_size=batch_size, device=device)
        return cls(rotation=rotation, translation=translation)

    @classmethod
    def from_rotation_translation(cls, rotation: Rotation, translation: Translation) -> TransformSE3:
        """Create from Rotation and Translation objects.

        Args:
            rotation: Rotation object.
            translation: Translation object.

        Returns:
            Transform.
        """
        return cls(rotation=rotation, translation=translation)

    @classmethod
    def from_matrix(cls, matrix: torch.Tensor) -> TransformSE3:
        """Create from 4x4 homogeneous matrix.

        Args:
            matrix: Homogeneous matrix [4, 4], [B, 4, 4], or [B, N, 4, 4].

        Returns:
            Transform.
        """
        assert matrix.shape[-2:] == (4, 4), f"Expected matrix[..., 4, 4], got {matrix.shape}"
        if matrix.ndim == 2:
            matrix = matrix.unsqueeze(0).unsqueeze(0)  # [4, 4] -> [1, 1, 4, 4]
        elif matrix.ndim == 3:
            matrix = matrix.unsqueeze(1)  # [B, 4, 4] -> [B, 1, 4, 4]
        # [B, N, 4, 4]: keep as is
        rotation = Rotation(R=matrix[..., :3, :3])
        translation = Translation(t=matrix[..., :3, 3])
        return cls(rotation=rotation, translation=translation)

    def to_matrix(self) -> torch.Tensor:
        """Convert to 4x4 homogeneous matrix.

        Returns:
            Homogeneous matrix [B, N, 4, 4].
        """
        R = self.rotation.R  # [B, N, 3, 3]
        t = self.translation.t  # [B, N, 3]

        B, N = R.shape[0], R.shape[1]
        matrix = torch.zeros(B, N, 4, 4, device=self.device, dtype=R.dtype)
        matrix[..., :3, :3] = R
        matrix[..., :3, 3] = t
        matrix[..., 3, 3] = 1.0

        return matrix

    def to(self, device: torch.device | str) -> TransformSE3:
        """Move transform to device.

        Args:
            device: Target device.

        Returns:
            Transform on target device.
        """
        return TransformSE3(
            rotation=self.rotation.to(device),
            translation=self.translation.to(device),
        )

    def inverse(self) -> TransformSE3:
        """Compute inverse transform.

        Returns:
            Inverse transform such that T @ T.inverse() = identity.
        """
        # calculate T_A_from_B = inverse(T_B_from_A)

        # R_A_from_B = transpose(R_B_from_A)
        rotation_inv = self.rotation.transpose()

        # A_t_A_B = -inverse(R_B_from_A) @ B_t_B_A
        t_inv = -rotation_inv.apply(self.translation.to_vector())

        # Return T_A_from_B
        return TransformSE3(rotation=rotation_inv, translation=Translation(t=t_inv))

    def __matmul__(self, other: TransformSE3) -> TransformSE3:
        """Compose transforms using @ operator.

        Example:
            T_C_from_A = T_C_from_B @ T_B_from_A

        Args:
            other: Another TransformSE3 (T_B_from_A when self is T_C_from_B).

        Returns:
            Composed transform T_C_from_A.
        """
        # Compose rotations: R_C_from_A = R_C_from_B @ R_B_from_A
        rotation_new = self.rotation @ other.rotation

        # Compose translations: C_t_C_A = R_C_from_B @ B_t_B_A + C_t_C_B
        t_rotated = self.rotation.apply(other.translation.to_vector())
        translation_new = Translation(t=t_rotated) + self.translation

        # Return the composed transform: T_C_from_A
        return TransformSE3(rotation=rotation_new, translation=translation_new)

    def apply(self, vectors: torch.Tensor) -> torch.Tensor:
        """Apply transform: v_out[i] = R[i] @ v_in[i] + t[i] for each batch element i.

        Args:
            vectors: Vectors to transform:
                     - [3]: single vector (requires batch_size=1)
                     - [B, 3]: one vector per batch (B must match batch_size)
                     - [B, N, 3]: N vectors per batch (B must match batch_size)

        Returns:
            Transformed points, same shape as input.
        """
        # Apply rotation, then translation
        rotated_vectors = self.rotation.apply(vectors)
        return self.translation.apply(rotated_vectors)

    @property
    def batch_size(self) -> int:
        """Get batch size."""
        self.validate_batch_size()
        return self.rotation.batch_size

    @property
    def num_elements(self) -> int:
        """Get number of elements per batch item."""
        return self.rotation.num_elements

    @property
    def device(self) -> torch.device:
        """Get device of transform."""
        self.validate_device()
        return self.rotation.device

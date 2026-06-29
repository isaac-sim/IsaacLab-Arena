# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""3D translation representation."""

from __future__ import annotations

import torch
from dataclasses import dataclass


@dataclass
class Translation:
    """3D translation vector.

    Represents translations in 3D space.
    Supports batching and per-element translations.

    Attributes:
        t: Translation vector of shape [B, N, 3] where B is batch size and N is number of
            elements per batch item. N=1 for single-transform use, N>1 for per-element
            (e.g. surface elements of an object).
    """

    # [B, N, 3]; N=1 for single-transform use, N>1 for per-element
    # (e.g. surface elements of an object)
    t: torch.Tensor

    def __post_init__(self) -> None:
        """Validate and normalize shape of translation vector after initialization."""
        # [3] -> [1, 3] -> [1, 1, 3]
        if self.t.ndim == 1:
            self.t = self.t.unsqueeze(0).unsqueeze(0)
        # [B, 3] -> [B, 1, 3]
        elif self.t.ndim == 2:
            self.t = self.t.unsqueeze(1)
        # [B, N, 3] -> keep as is

        # Guarded: torch.compile traces __post_init__ and cannot access attributes
        # on the partially-constructed proxy object, causing InternalTorchDynamoError.
        if not torch.compiler.is_compiling():
            assert self.t.ndim == 3, f"t must be [B, N, 3], got shape {self.t.shape}"
            assert self.t.shape[-1] == 3, f"t must be [B, N, 3], got shape {self.t.shape}"

    @classmethod
    def create_zero(cls, batch_size: int | None = None, device: torch.device | None = None) -> Translation:
        """Create zero translation(s).

        Args:
            batch_size: Batch size (None for single translation).
            device: Torch device (defaults to CUDA if available, else CPU).

        Returns:
            Zero translation(s).
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = batch_size if batch_size else 1
        t = torch.zeros(batch_size, 3, device=device)
        return cls(t=t)

    @classmethod
    def from_xyz(
        cls,
        x: float | torch.Tensor,
        y: float | torch.Tensor,
        z: float | torch.Tensor,
        device: torch.device | None = None,
    ) -> Translation:
        """Create translation from x, y, z components.

        Args:
            x: X component (scalar or batch [B]).
            y: Y component (scalar or batch [B]).
            z: Z component (scalar or batch [B]).
            device: Torch device (defaults to CUDA if available, else CPU).

        Returns:
            Translation from components.
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_t = torch.as_tensor(x, device=device)
        y_t = torch.as_tensor(y, device=device)
        z_t = torch.as_tensor(z, device=device)
        t = torch.stack([x_t, y_t, z_t], dim=-1)
        return cls(t=t)

    @property
    def num_elements(self) -> int:
        """Get number of elements per batch item."""
        return int(self.t.shape[1])

    def to_vector(self) -> torch.Tensor:
        """Convert to translation vector tensor.

        Returns:
            Translation vector [B, N, 3].
        """
        return self.t

    def to(self, device: torch.device | str) -> Translation:
        """Move translation to device.

        Args:
            device: Target device.

        Returns:
            Translation on target device.
        """
        return Translation(t=self.t.to(device))

    def __add__(self, other: Translation) -> Translation:
        """Compose translations: t1 + t2."""
        assert self.t.shape == other.t.shape, f"Translation shapes must match: {self.t.shape} != {other.t.shape}"
        return Translation(t=self.t + other.t)

    def __neg__(self) -> Translation:
        """Inverse translation: -t."""
        return Translation(t=-self.t)

    def __repr__(self) -> str:
        """String representation."""
        if self.batch_size == 1 and self.num_elements == 1:  # Single translation [1, 1, 3]
            return f"Translation(x={self.x.item():.4f}, y={self.y.item():.4f}, z={self.z.item():.4f})"
        return f"Translation(batch_size={self.batch_size}, device={self.device})"

    def apply(self, vectors: torch.Tensor) -> torch.Tensor:
        """Apply translation: v_out[i] = v_in[i] + t[i] for each batch element i.

        Args:
            vectors: Vectors to translate:
                     - [3]: single vector (requires batch_size=1, num_elements=1)
                     - [B, 3]: one vector per batch (B must match batch_size, num_elements=1)
                     - [B, N, 3]: N vectors per batch (B must match batch_size)

        Returns:
            Translated points, same shape as input.
        """
        assert vectors.shape[-1] == 3, f"Expected [..., 3], got {vectors.shape}"
        assert vectors.ndim in (1, 2, 3), f"Expected [3], [B, 3], or [B, N, 3], got {vectors.shape}"

        if vectors.ndim == 1:  # [3]
            assert self.batch_size == 1, f"[3] input requires batch_size=1, got {self.batch_size}"
            return vectors + self.t[0, 0]

        # [B, 3] or [B, N, 3]
        assert (
            vectors.shape[0] == self.batch_size
        ), f"Batch size mismatch: vectors {vectors.shape[0]}, t {self.batch_size}"

        if vectors.ndim == 2:  # [B, 3]
            return vectors + self.t[:, 0]

        return vectors + self.t  # [B, N, 3]

    @property
    def x(self) -> torch.Tensor:
        """X component(s)."""
        return self.t[..., 0]

    @property
    def y(self) -> torch.Tensor:
        """Y component(s)."""
        return self.t[..., 1]

    @property
    def z(self) -> torch.Tensor:
        """Z component(s)."""
        return self.t[..., 2]

    @property
    def batch_size(self) -> int:
        """Get batch size (always >= 1)."""
        return int(self.t.shape[0])

    @property
    def device(self) -> torch.device:
        """Get device of translation vector."""
        return self.t.device

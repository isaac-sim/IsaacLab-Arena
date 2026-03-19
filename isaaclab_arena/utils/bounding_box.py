# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for computing spatial relationships between objects in a scene.
This module provides functions for:
- Computing bounding boxes from USD assets
- Calculating placement poses based on semantic relationships (e.g., "on_top_of")
- Supporting randomized placement with specified constraints
- AxisAlignedBoundingBox with always-tensor storage supporting single (N=1) and batched (N>1) modes
"""

import torch
from typing import Union

from isaaclab_arena.utils.pose import Pose


class AxisAlignedBoundingBox:
    """Axis-aligned bounding box with tensor storage.

    Stores min/max extents as (N, 3) tensors where N=1 for a single bbox
    and N>1 for batched (one bbox per environment).

    Constructor accepts tuples, 1D tensors (auto-wrapped to N=1), or (N, 3) tensors.
    """

    def __init__(
        self,
        min_point: Union[tuple[float, float, float], torch.Tensor],
        max_point: Union[tuple[float, float, float], torch.Tensor],
    ):
        self._min_point = self._to_2d_tensor(min_point)
        self._max_point = self._to_2d_tensor(max_point)
        assert self._min_point.shape == self._max_point.shape
        assert self._min_point.shape[-1] == 3

    @staticmethod
    def _to_2d_tensor(v: Union[tuple[float, float, float], torch.Tensor]) -> torch.Tensor:
        if isinstance(v, tuple):
            return torch.tensor([v], dtype=torch.float32)
        if v.dim() == 1:
            return v.unsqueeze(0).float()
        return v.float()

    @property
    def min_point(self) -> torch.Tensor:
        """Minimum extent (x, y, z). Shape (N, 3)."""
        return self._min_point

    @property
    def max_point(self) -> torch.Tensor:
        """Maximum extent (x, y, z). Shape (N, 3)."""
        return self._max_point

    @property
    def num_envs(self) -> int:
        """Number of environments (leading dimension N)."""
        return self._min_point.shape[0]

    @property
    def size(self) -> torch.Tensor:
        """Size (width, depth, height). Shape (N, 3)."""
        return self._max_point - self._min_point

    @property
    def center(self) -> torch.Tensor:
        """Center point. Shape (N, 3)."""
        return (self._min_point + self._max_point) * 0.5

    @property
    def top_surface_z(self) -> torch.Tensor:
        """Z-coordinate of the top surface. Shape (N,)."""
        return self._max_point[:, 2]

    @property
    def bottom_surface_z(self) -> torch.Tensor:
        """Z-coordinate of the bottom surface. Shape (N,)."""
        return self._min_point[:, 2]

    def get_corners_at(self, pos: torch.Tensor | None = None) -> torch.Tensor:
        """Get 8 corners, optionally offset by position.

        Args:
            pos: World position to offset corners by. Shape (3,) or (N, 3).
                 If None, returns corners in local frame.

        Returns:
            Tensor of shape (N, 8, 3) with corners ordered: bottom 4, then top 4.
        """
        mn, mx = self._min_point, self._max_point
        corners = torch.stack(
            [
                torch.stack([mn[:, 0], mn[:, 1], mn[:, 2]], dim=1),
                torch.stack([mx[:, 0], mn[:, 1], mn[:, 2]], dim=1),
                torch.stack([mx[:, 0], mx[:, 1], mn[:, 2]], dim=1),
                torch.stack([mn[:, 0], mx[:, 1], mn[:, 2]], dim=1),
                torch.stack([mn[:, 0], mn[:, 1], mx[:, 2]], dim=1),
                torch.stack([mx[:, 0], mn[:, 1], mx[:, 2]], dim=1),
                torch.stack([mx[:, 0], mx[:, 1], mx[:, 2]], dim=1),
                torch.stack([mn[:, 0], mx[:, 1], mx[:, 2]], dim=1),
            ],
            dim=1,
        )
        if pos is not None:
            if pos.dim() == 1:
                pos = pos.unsqueeze(0)
            corners = corners + pos.unsqueeze(1)
        return corners

    def scaled(self, scale: Union[tuple[float, float, float], torch.Tensor]) -> "AxisAlignedBoundingBox":
        """Return a new bounding box with scale applied.

        Args:
            scale: Scale factors (x, y, z). Tuple, shape (3,), or (N, 3).
        """
        s = self._coerce_offset(scale)
        return AxisAlignedBoundingBox(min_point=self._min_point * s, max_point=self._max_point * s)

    def translated(self, offset: Union[tuple[float, float, float], torch.Tensor]) -> "AxisAlignedBoundingBox":
        """Return a new bounding box translated by an offset.

        Args:
            offset: Translation (x, y, z). Tuple, shape (3,), or (N, 3).
        """
        o = self._coerce_offset(offset)
        return AxisAlignedBoundingBox(min_point=self._min_point + o, max_point=self._max_point + o)

    def centered(self) -> "AxisAlignedBoundingBox":
        """Return a new bounding box centered at the origin."""
        c = self.center
        return AxisAlignedBoundingBox(min_point=self._min_point - c, max_point=self._max_point - c)

    def overlaps(self, other: "AxisAlignedBoundingBox", margin: float = 0.0) -> torch.Tensor:
        """Check if two AABBs overlap in 3D.

        Args:
            other: The other bounding box to test against.
            margin: Minimum required separation in meters.

        Returns:
            Boolean tensor of shape (N,). Broadcasting applies if self and other have different N.
        """
        return (
            (self._max_point[:, 0] + margin > other._min_point[:, 0])
            & (other._max_point[:, 0] + margin > self._min_point[:, 0])
            & (self._max_point[:, 1] + margin > other._min_point[:, 1])
            & (other._max_point[:, 1] + margin > self._min_point[:, 1])
            & (self._max_point[:, 2] + margin > other._min_point[:, 2])
            & (other._max_point[:, 2] + margin > self._min_point[:, 2])
        )

    def rotated_90_around_z(self, quarters: int) -> "AxisAlignedBoundingBox":
        """Rotate AABB by quarters * 90 degrees around Z axis.

        Only 90 degree increments are supported to preserve axis-alignment.

        Args:
            quarters: Number of 90 degree rotations (0, 1, 2, or 3).
        """
        quarters = quarters % 4
        min_x, min_y, min_z = self._min_point[:, 0], self._min_point[:, 1], self._min_point[:, 2]
        max_x, max_y, max_z = self._max_point[:, 0], self._max_point[:, 1], self._max_point[:, 2]
        if quarters == 0:
            return AxisAlignedBoundingBox(min_point=self._min_point.clone(), max_point=self._max_point.clone())
        elif quarters == 1:
            return AxisAlignedBoundingBox(
                min_point=torch.stack([-max_y, min_x, min_z], dim=1),
                max_point=torch.stack([-min_y, max_x, max_z], dim=1),
            )
        elif quarters == 2:
            return AxisAlignedBoundingBox(
                min_point=torch.stack([-max_x, -max_y, min_z], dim=1),
                max_point=torch.stack([-min_x, -min_y, max_z], dim=1),
            )
        else:
            return AxisAlignedBoundingBox(
                min_point=torch.stack([min_y, -max_x, min_z], dim=1),
                max_point=torch.stack([max_y, -min_x, max_z], dim=1),
            )

    def _coerce_offset(self, v: Union[tuple[float, float, float], torch.Tensor]) -> torch.Tensor:
        """Convert tuple or tensor to broadcastable (1, 3) or (N, 3) tensor."""
        if isinstance(v, tuple):
            return torch.tensor([v], dtype=self._min_point.dtype, device=self._min_point.device)
        if v.dim() == 1:
            return v.unsqueeze(0)
        return v

    def __repr__(self) -> str:
        if self.num_envs == 1:
            mn = tuple(self._min_point[0].tolist())
            mx = tuple(self._max_point[0].tolist())
            return f"AxisAlignedBoundingBox(min_point={mn}, max_point={mx})"
        return f"AxisAlignedBoundingBox(num_envs={self.num_envs}, min_point={self._min_point}, max_point={self._max_point})"


def quaternion_to_90_deg_z_quarters(rotation_wxyz: tuple[float, float, float, float], tol_deg: float = 1.0) -> int:
    """Convert a quaternion to 90 degree rotation quarters around Z axis.

    Only supports rotations that are multiples of 90 degrees around the Z axis.
    Raises AssertionError for any other rotation.

    Args:
        rotation_wxyz: Quaternion as (w, x, y, z).
        tol_deg: Tolerance in degrees for how close the angle must be to a 90 degree multiple.

    Returns:
        Number of 90 degree quarters (0, 1, 2, or 3).
    """
    import math

    w, x, y, z = rotation_wxyz

    assert (
        abs(x) < 1e-3 and abs(y) < 1e-3
    ), f"Only rotations around Z axis are supported. Got quaternion (w={w:.4f}, x={x:.4f}, y={y:.4f}, z={z:.4f})."

    angle_deg = math.degrees(2 * math.atan2(z, w)) % 360
    quarters = round(angle_deg / 90) % 4
    remainder_deg = min(angle_deg % 90, 90 - angle_deg % 90)

    assert remainder_deg < tol_deg, (
        "Only 90 degree rotation multiples around Z are supported. "
        f"Got {angle_deg:.1f} degrees (nearest 90 degree multiple: {quarters * 90} degrees)."
    )

    return quarters


def get_random_pose_within_bounding_box(bbox: AxisAlignedBoundingBox, seed: int | None = None) -> Pose:
    """Generate a random pose with position uniformly sampled within a bounding box.

    Args:
        bbox: Bounding box defining the valid region for sampling (uses first env if batched).
        seed: Optional random seed for reproducibility.

    Returns:
        Pose with random position within bbox and identity rotation.
    """
    if seed is not None:
        torch.manual_seed(seed)

    min_point = bbox.min_point[0]
    max_point = bbox.max_point[0]
    random_position = min_point + (max_point - min_point) * torch.rand(3)

    pose = Pose(position_xyz=tuple(random_position.tolist()), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
    return pose

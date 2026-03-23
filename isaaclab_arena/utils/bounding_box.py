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

from isaaclab_arena.utils.pose import Pose


class AxisAlignedBoundingBox:
    """Axis-aligned bounding box storing local extents. Use get_corners_at(pos) for world-space corners.

    Stores min/max extents as (N, 3) tensors where N is the number of environments.
    Properties return tuples/floats when N=1 and tensors when N>1.
    Constructor accepts tuples, 1D tensors, or (N, 3) tensors.
    """

    def __init__(
        self,
        min_point: tuple[float, float, float] | torch.Tensor,
        max_point: tuple[float, float, float] | torch.Tensor,
    ):
        self._min_point = self._to_batched_tensor(min_point)
        self._max_point = self._to_batched_tensor(max_point)
        assert self._min_point.shape == self._max_point.shape
        assert self._min_point.shape[-1] == 3

    @staticmethod
    def _to_batched_tensor(value: tuple[float, float, float] | torch.Tensor) -> torch.Tensor:
        """Convert tuple, 1-D tensor, or (N, 3) tensor to (N, 3) float32 tensor."""
        if isinstance(value, tuple):
            return torch.tensor([value], dtype=torch.float32)
        if value.dim() == 1:
            return value.unsqueeze(0).float()
        return value.float()

    def _format_output(self, tensor: torch.Tensor):
        """Return tuple when N=1, tensor when N>1."""
        if self._min_point.shape[0] == 1:
            return tuple(tensor[0].tolist())
        return tensor

    def _format_scalar(self, tensor: torch.Tensor):
        """Return float when N=1, tensor when N>1."""
        if self._min_point.shape[0] == 1:
            return tensor.item()
        return tensor

    @property
    def min_point(self) -> tuple[float, float, float] | torch.Tensor:
        """Local minimum extent (x, y, z) relative to object origin."""
        return self._format_output(self._min_point)

    @property
    def max_point(self) -> tuple[float, float, float] | torch.Tensor:
        """Local maximum extent (x, y, z) relative to object origin."""
        return self._format_output(self._max_point)

    @property
    def num_envs(self) -> int:
        """Number of environments (leading dimension N)."""
        return self._min_point.shape[0]

    @property
    def size(self) -> tuple[float, float, float] | torch.Tensor:
        """Returns the size (width, depth, height) of the bounding box."""
        return self._format_output(self._max_point - self._min_point)

    @property
    def center(self) -> tuple[float, float, float] | torch.Tensor:
        """Returns the center point of the bounding box."""
        return self._format_output((self._min_point + self._max_point) * 0.5)

    @property
    def top_surface_z(self) -> float | torch.Tensor:
        """Returns the z-coordinate of the top surface."""
        return self._format_scalar(self._max_point[:, 2])

    @property
    def bottom_surface_z(self) -> float | torch.Tensor:
        """Returns the z-coordinate of the bottom surface."""
        return self._format_scalar(self._min_point[:, 2])

    def get_corners_at(self, pos: torch.Tensor | None = None) -> torch.Tensor:
        """Get 8 corners of this bounding box, optionally offset by position.

        Args:
            pos: If provided, world position (x, y, z) to offset corners by.
                 If None, returns corners in local/object frame.

        Returns:
            Tensor of shape (8, 3) for N=1 or (N, 8, 3) for N>1,
            with corners ordered: bottom 4, then top 4.
        """
        min_pt, max_pt = self._min_point, self._max_point
        corners = torch.stack(
            [
                torch.stack([min_pt[:, 0], min_pt[:, 1], min_pt[:, 2]], dim=1),  # Bottom-front-left
                torch.stack([max_pt[:, 0], min_pt[:, 1], min_pt[:, 2]], dim=1),  # Bottom-front-right
                torch.stack([max_pt[:, 0], max_pt[:, 1], min_pt[:, 2]], dim=1),  # Bottom-back-right
                torch.stack([min_pt[:, 0], max_pt[:, 1], min_pt[:, 2]], dim=1),  # Bottom-back-left
                torch.stack([min_pt[:, 0], min_pt[:, 1], max_pt[:, 2]], dim=1),  # Top-front-left
                torch.stack([max_pt[:, 0], min_pt[:, 1], max_pt[:, 2]], dim=1),  # Top-front-right
                torch.stack([max_pt[:, 0], max_pt[:, 1], max_pt[:, 2]], dim=1),  # Top-back-right
                torch.stack([min_pt[:, 0], max_pt[:, 1], max_pt[:, 2]], dim=1),  # Top-back-left
            ],
            dim=1,
        )
        if pos is not None:
            if pos.dim() == 1:
                pos = pos.unsqueeze(0)
            corners = corners + pos.unsqueeze(1)
        if self._min_point.shape[0] == 1:
            return corners.squeeze(0)
        return corners

    def scaled(self, scale: tuple[float, float, float] | torch.Tensor) -> "AxisAlignedBoundingBox":
        """Return a new bounding box with scale applied.

        Args:
            scale: Scale factors (x, y, z) to apply.

        Returns:
            New AxisAlignedBoundingBox with scaled dimensions.
        """
        scale = self._to_batched_tensor(scale)
        return AxisAlignedBoundingBox(min_point=self._min_point * scale, max_point=self._max_point * scale)

    def translated(self, offset: tuple[float, float, float] | torch.Tensor) -> "AxisAlignedBoundingBox":
        """Return a new bounding box translated by an offset.

        Args:
            offset: Translation offset (x, y, z) to apply.

        Returns:
            New AxisAlignedBoundingBox with translated position.
        """
        offset = self._to_batched_tensor(offset)
        return AxisAlignedBoundingBox(min_point=self._min_point + offset, max_point=self._max_point + offset)

    def centered(self) -> "AxisAlignedBoundingBox":
        """Return a new bounding box centered around the origin.

        The returned bbox has the same size but is shifted so that its
        center is at (0, 0, 0).

        Returns:
            New AxisAlignedBoundingBox centered at origin.
        """
        center = (self._min_point + self._max_point) * 0.5
        return AxisAlignedBoundingBox(min_point=self._min_point - center, max_point=self._max_point - center)

    def overlaps(self, other: "AxisAlignedBoundingBox", margin: float = 0.0) -> bool | torch.Tensor:
        """Check if two AABBs overlap in 3D.

        Args:
            other: The other bounding box to test against.
            margin: Minimum required separation in meters. A positive value
                rejects placements where the gap is smaller than margin.

        Returns:
            bool when both are N=1, or (N,) bool tensor when batched.
        """
        result = (
            (self._max_point[:, 0] + margin > other._min_point[:, 0])
            & (other._max_point[:, 0] + margin > self._min_point[:, 0])
            & (self._max_point[:, 1] + margin > other._min_point[:, 1])
            & (other._max_point[:, 1] + margin > self._min_point[:, 1])
            & (self._max_point[:, 2] + margin > other._min_point[:, 2])
            & (other._max_point[:, 2] + margin > self._min_point[:, 2])
        )
        if result.numel() == 1:
            return bool(result.item())
        return result

    def rotated_90_around_z(self, quarters: int) -> "AxisAlignedBoundingBox":
        """Rotate AABB by quarters * 90° around Z axis.

        Only 90° increments are supported to preserve axis-alignment without size increase.

        Args:
            quarters: Number of 90° rotations (0=0°, 1=90°, 2=180°, 3=270°/-90°).

        Returns:
            New AxisAlignedBoundingBox rotated around Z axis.
        """
        quarters = quarters % 4
        min_x, min_y, min_z = self._min_point[:, 0], self._min_point[:, 1], self._min_point[:, 2]
        max_x, max_y, max_z = self._max_point[:, 0], self._max_point[:, 1], self._max_point[:, 2]
        if quarters == 0:
            return AxisAlignedBoundingBox(min_point=self._min_point.clone(), max_point=self._max_point.clone())
        elif quarters == 1:  # 90° CCW
            return AxisAlignedBoundingBox(
                min_point=torch.stack([-max_y, min_x, min_z], dim=1),
                max_point=torch.stack([-min_y, max_x, max_z], dim=1),
            )
        elif quarters == 2:  # 180°
            return AxisAlignedBoundingBox(
                min_point=torch.stack([-max_x, -max_y, min_z], dim=1),
                max_point=torch.stack([-min_x, -min_y, max_z], dim=1),
            )
        else:  # 270° CCW / -90° (quarters == 3)
            return AxisAlignedBoundingBox(
                min_point=torch.stack([min_y, -max_x, min_z], dim=1),
                max_point=torch.stack([max_y, -min_x, max_z], dim=1),
            )


def quaternion_to_90_deg_z_quarters(rotation_xyzw: tuple[float, float, float, float], tol_deg: float = 1.0) -> int:
    """Convert a quaternion to 90° rotation quarters around Z axis.

    Only supports rotations that are multiples of 90° around the Z axis.
    Raises AssertionError for any other rotation.

    Args:
        rotation_xyzw: Quaternion as (x, y, z, w).
        tol_deg: Tolerance in degrees for how close the angle must be to a 90° multiple.

    Returns:
        Number of 90° quarters (0, 1, 2, or 3).

    Raises:
        AssertionError: If the quaternion is not a pure Z rotation or not a 90° multiple.
    """
    import math

    x, y, z, w = rotation_xyzw

    # Must be a pure Z rotation (x and y components must be ~0)
    assert (
        abs(x) < 1e-3 and abs(y) < 1e-3
    ), f"Only rotations around Z axis are supported. Got quaternion (w={w:.4f}, x={x:.4f}, y={y:.4f}, z={z:.4f})."

    # Compute rotation angle around Z and normalize to [0°, 360°)
    angle_deg = math.degrees(2 * math.atan2(z, w)) % 360
    quarters = round(angle_deg / 90) % 4
    remainder_deg = min(angle_deg % 90, 90 - angle_deg % 90)

    assert remainder_deg < tol_deg, (
        "Only 90° rotation multiples around Z are supported. "
        f"Got {angle_deg:.1f}° (nearest 90° multiple: {quarters * 90}°)."
    )

    return quarters


def get_random_pose_within_bounding_box(bbox: AxisAlignedBoundingBox, seed: int | None = None) -> Pose:
    """Generate a random pose (position and identity rotation) with position uniformly
       sampled within a bounding box.

    Args:
        bbox: Bounding box defining the valid region for sampling
        seed: Optional random seed for reproducibility

    Returns:
        Pose with random position within bbox and identity rotation
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Get workspace bounds
    min_point = torch.tensor(bbox.min_point, dtype=torch.float32)
    max_point = torch.tensor(bbox.max_point, dtype=torch.float32)

    # Sample random position uniformly within workspace bounds
    # random_position = min + (max - min) * rand
    random_position = min_point + (max_point - min_point) * torch.rand(3)

    pose = Pose(position_xyz=tuple(random_position.tolist()), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))

    return pose

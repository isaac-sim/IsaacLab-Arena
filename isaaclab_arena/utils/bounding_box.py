# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for computing spatial relationships between objects in a scene.
This module provides:
- Computing bounding boxes from USD assets
- Calculating placement poses based on semantic relationships (e.g., "on_top_of")
- Supporting randomized placement with specified constraints
- Supporting batched AABBs (BatchedAxisAlignedBoundingBox) and world_bbox_to_min_max_tensors() for relation loss over multiple envs
"""

import torch
from dataclasses import dataclass
from typing import Union, cast

from isaaclab_arena.utils.pose import Pose


@dataclass
class AxisAlignedBoundingBox:
    """Axis-aligned bounding box storing local extents. Use get_corners_at(pos) for world-space corners."""

    min_point: tuple[float, float, float]
    """Local minimum extent (x, y, z) relative to object origin."""

    max_point: tuple[float, float, float]
    """Local maximum extent (x, y, z) relative to object origin."""

    def __post_init__(self):
        assert isinstance(self.min_point, tuple)
        assert isinstance(self.max_point, tuple)
        assert len(self.min_point) == 3
        assert len(self.max_point) == 3

    @property
    def size(self) -> tuple[float, float, float]:
        """Returns the size (width, depth, height) of the bounding box."""
        return (
            self.max_point[0] - self.min_point[0],
            self.max_point[1] - self.min_point[1],
            self.max_point[2] - self.min_point[2],
        )

    @property
    def center(self) -> tuple[float, float, float]:
        """Returns the center point of the bounding box."""
        return (
            (self.min_point[0] + self.max_point[0]) / 2.0,
            (self.min_point[1] + self.max_point[1]) / 2.0,
            (self.min_point[2] + self.max_point[2]) / 2.0,
        )

    @property
    def top_surface_z(self) -> float:
        """Returns the z-coordinate of the top surface."""
        return self.max_point[2]

    @property
    def bottom_surface_z(self) -> float:
        """Returns the z-coordinate of the bottom surface."""
        return self.min_point[2]

    def get_corners_at(self, pos: torch.Tensor | None = None) -> torch.Tensor:
        """Get 8 corners of this bounding box, optionally offset by position.

        Args:
            pos: If provided, world position (x, y, z) to offset corners by.
                 If None, returns corners in local/object frame.

        Returns:
            Tensor of shape (8, 3) with corners ordered: bottom 4, then top 4.
        """
        if pos is None:
            # Local corners directly from min_point/max_point
            min_pt, max_pt = self.min_point, self.max_point
            return torch.tensor([
                [min_pt[0], min_pt[1], min_pt[2]],  # Bottom-front-left
                [max_pt[0], min_pt[1], min_pt[2]],  # Bottom-front-right
                [max_pt[0], max_pt[1], min_pt[2]],  # Bottom-back-right
                [min_pt[0], max_pt[1], min_pt[2]],  # Bottom-back-left
                [min_pt[0], min_pt[1], max_pt[2]],  # Top-front-left
                [max_pt[0], min_pt[1], max_pt[2]],  # Top-front-right
                [max_pt[0], max_pt[1], max_pt[2]],  # Top-back-right
                [min_pt[0], max_pt[1], max_pt[2]],  # Top-back-left
            ])
        else:
            return self.get_corners_at(pos=None) + pos

    def scaled(self, scale: tuple[float, float, float]) -> "AxisAlignedBoundingBox":
        """Return a new bounding box with scale applied.

        Args:
            scale: Scale factors (x, y, z) to apply.

        Returns:
            New AxisAlignedBoundingBox with scaled dimensions.
        """
        return AxisAlignedBoundingBox(
            min_point=(
                self.min_point[0] * scale[0],
                self.min_point[1] * scale[1],
                self.min_point[2] * scale[2],
            ),
            max_point=(
                self.max_point[0] * scale[0],
                self.max_point[1] * scale[1],
                self.max_point[2] * scale[2],
            ),
        )

    def translated(self, offset: tuple[float, float, float]) -> "AxisAlignedBoundingBox":
        """Return a new bounding box translated by an offset.

        Args:
            offset: Translation offset (x, y, z) to apply.

        Returns:
            New AxisAlignedBoundingBox with translated position.
        """
        return AxisAlignedBoundingBox(
            min_point=(
                self.min_point[0] + offset[0],
                self.min_point[1] + offset[1],
                self.min_point[2] + offset[2],
            ),
            max_point=(
                self.max_point[0] + offset[0],
                self.max_point[1] + offset[1],
                self.max_point[2] + offset[2],
            ),
        )

    def centered(self) -> "AxisAlignedBoundingBox":
        """Return a new bounding box centered around the origin.

        The returned bbox has the same size but is shifted so that its
        center is at (0, 0, 0).

        Returns:
            New AxisAlignedBoundingBox centered at origin.
        """
        c = self.center
        return AxisAlignedBoundingBox(
            min_point=(
                self.min_point[0] - c[0],
                self.min_point[1] - c[1],
                self.min_point[2] - c[2],
            ),
            max_point=(
                self.max_point[0] - c[0],
                self.max_point[1] - c[1],
                self.max_point[2] - c[2],
            ),
        )

    def overlaps(self, other: "AxisAlignedBoundingBox", margin: float = 0.0) -> bool:
        """Check if two AABBs overlap in 3D.

        Args:
            other: The other bounding box to test against.
            margin: Minimum required separation in meters. A positive value
                rejects placements where the gap is smaller than margin.

        Returns:
            True if the volumes overlap (or are closer than margin).
        """
        return (
            self.max_point[0] + margin > other.min_point[0]
            and other.max_point[0] + margin > self.min_point[0]
            and self.max_point[1] + margin > other.min_point[1]
            and other.max_point[1] + margin > self.min_point[1]
            and self.max_point[2] + margin > other.min_point[2]
            and other.max_point[2] + margin > self.min_point[2]
        )

    def rotated_90_around_z(self, quarters: int) -> "AxisAlignedBoundingBox":
        """Rotate AABB by quarters * 90° around Z axis.

        Only 90° increments are supported to preserve axis-alignment without size increase.

        Args:
            quarters: Number of 90° rotations (0=0°, 1=90°, 2=180°, 3=270°/-90°).

        Returns:
            New AxisAlignedBoundingBox rotated around Z axis.
        """
        min_x, min_y, min_z = self.min_point
        max_x, max_y, max_z = self.max_point

        quarters = quarters % 4
        if quarters == 0:
            return AxisAlignedBoundingBox(
                min_point=(min_x, min_y, min_z),
                max_point=(max_x, max_y, max_z),
            )
        elif quarters == 1:  # 90° CCW
            return AxisAlignedBoundingBox(
                min_point=(-max_y, min_x, min_z),
                max_point=(-min_y, max_x, max_z),
            )
        elif quarters == 2:  # 180°
            return AxisAlignedBoundingBox(
                min_point=(-max_x, -max_y, min_z),
                max_point=(-min_x, -min_y, max_z),
            )
        else:  # 270° CCW / -90° (quarters == 3)
            return AxisAlignedBoundingBox(
                min_point=(min_y, -max_x, min_z),
                max_point=(max_y, -min_x, max_z),
            )


@dataclass
class BatchedAxisAlignedBoundingBox:
    """Axis-aligned bounding box in world frame, one per environment (batched). Use get_corners_at(pos) for offset corners.

    Same concept as AxisAlignedBoundingBox but for N envs; return types are tensors with leading dimension N.
    """

    min_corner: torch.Tensor
    """Minimum extent (x, y, z) per environment. Shape (N, 3)."""

    max_corner: torch.Tensor
    """Maximum extent (x, y, z) per environment. Shape (N, 3)."""

    @property
    def min_point(self) -> torch.Tensor:
        """Minimum extent per environment. Shape (N, 3). Alias for min_corner (API parity with AxisAlignedBoundingBox)."""
        return self.min_corner

    @property
    def max_point(self) -> torch.Tensor:
        """Maximum extent per environment. Shape (N, 3). Alias for max_corner (API parity with AxisAlignedBoundingBox)."""
        return self.max_corner

    @property
    def size(self) -> torch.Tensor:
        """Returns the size (width, depth, height) of the bounding box per environment. Shape (N, 3)."""
        return self.max_corner - self.min_corner

    @property
    def center(self) -> torch.Tensor:
        """Returns the center point of the bounding box per environment. Shape (N, 3)."""
        return (self.min_corner + self.max_corner) * 0.5

    @property
    def top_surface_z(self) -> torch.Tensor:
        """Returns the z-coordinate of the top surface per environment. Shape (N,)."""
        return self.max_corner[:, 2]

    @property
    def bottom_surface_z(self) -> torch.Tensor:
        """Returns the z-coordinate of the bottom surface per environment. Shape (N,)."""
        return self.min_corner[:, 2]

    def get_corners_at(self, pos: torch.Tensor | None = None) -> torch.Tensor:
        """Get 8 corners of this bounding box per environment, optionally offset by position.

        Args:
            pos: If provided, world position (N, 3) to offset corners by.
                 If None, returns corners in place.

        Returns:
            Tensor of shape (N, 8, 3) with corners ordered: bottom 4, then top 4.
        """
        mn, mx = self.min_corner, self.max_corner
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
            corners = corners + pos.unsqueeze(1)
        return corners

    def scaled(self, scale: torch.Tensor | tuple[float, float, float]) -> "BatchedAxisAlignedBoundingBox":
        """Return a new bounding box with scale applied.

        Args:
            scale: Scale factors (x, y, z). Shape (3,) or (N, 3).

        Returns:
            New BatchedAxisAlignedBoundingBox with scaled dimensions.
        """
        s = cast(
            torch.Tensor,
            (
                scale
                if isinstance(scale, torch.Tensor)
                else torch.tensor(scale, device=self.min_corner.device, dtype=self.min_corner.dtype)
            ),
        )
        if s.dim() == 1:
            s = s.unsqueeze(0).expand(self.min_corner.shape[0], 3)
        return BatchedAxisAlignedBoundingBox(
            min_corner=self.min_corner * s,
            max_corner=self.max_corner * s,
        )

    def translated(self, offset: torch.Tensor | tuple[float, float, float]) -> "BatchedAxisAlignedBoundingBox":
        """Return a new bounding box translated by an offset.

        Args:
            offset: Translation offset (x, y, z). Shape (3,) or (N, 3).

        Returns:
            New BatchedAxisAlignedBoundingBox with translated position.
        """
        o = cast(
            torch.Tensor,
            (
                offset
                if isinstance(offset, torch.Tensor)
                else torch.tensor(offset, device=self.min_corner.device, dtype=self.min_corner.dtype)
            ),
        )
        if o.dim() == 1:
            o = o.unsqueeze(0).expand(self.min_corner.shape[0], 3)
        return BatchedAxisAlignedBoundingBox(
            min_corner=self.min_corner + o,
            max_corner=self.max_corner + o,
        )

    def centered(self) -> "BatchedAxisAlignedBoundingBox":
        """Return a new bounding box centered around the origin.

        The returned bbox has the same size but is shifted so that its
        center is at (0, 0, 0) per environment.

        Returns:
            New BatchedAxisAlignedBoundingBox centered at origin.
        """
        c = self.center
        return BatchedAxisAlignedBoundingBox(
            min_corner=self.min_corner - c,
            max_corner=self.max_corner - c,
        )

    def overlaps(
        self,
        other: Union["AxisAlignedBoundingBox", "BatchedAxisAlignedBoundingBox"],
        margin: float = 0.0,
    ) -> torch.Tensor:
        """Check if this and other AABBs overlap in 3D per environment.

        Args:
            other: The other bounding box (single or batched) to test against.
            margin: Minimum required separation in meters. A positive value
                rejects placements where the gap is smaller than margin.

        Returns:
            Boolean tensor of shape (N,) True where the volumes overlap (or are closer than margin).
        """
        n = self.min_corner.shape[0]
        if isinstance(other, AxisAlignedBoundingBox):
            other_min = (
                torch.tensor(other.min_point, device=self.min_corner.device, dtype=self.min_corner.dtype)
                .unsqueeze(0)
                .expand(n, 3)
            )
            other_max = (
                torch.tensor(other.max_point, device=self.max_corner.device, dtype=self.max_corner.dtype)
                .unsqueeze(0)
                .expand(n, 3)
            )
        else:
            other_min, other_max = other.min_corner, other.max_corner
        return (
            (self.max_corner[:, 0] + margin > other_min[:, 0])
            & (other_max[:, 0] + margin > self.min_corner[:, 0])
            & (self.max_corner[:, 1] + margin > other_min[:, 1])
            & (other_max[:, 1] + margin > self.min_corner[:, 1])
            & (self.max_corner[:, 2] + margin > other_min[:, 2])
            & (other_max[:, 2] + margin > self.min_corner[:, 2])
        )

    def rotated_90_around_z(self, quarters: int) -> "BatchedAxisAlignedBoundingBox":
        """Rotate AABB by quarters * 90° around Z axis per environment.

        Only 90° increments are supported to preserve axis-alignment without size increase.

        Args:
            quarters: Number of 90° rotations (0=0°, 1=90°, 2=180°, 3=270°/-90°).

        Returns:
            New BatchedAxisAlignedBoundingBox rotated around Z axis.
        """
        quarters = quarters % 4
        min_x, min_y, min_z = self.min_corner[:, 0], self.min_corner[:, 1], self.min_corner[:, 2]
        max_x, max_y, max_z = self.max_corner[:, 0], self.max_corner[:, 1], self.max_corner[:, 2]
        if quarters == 0:
            return BatchedAxisAlignedBoundingBox(
                min_corner=self.min_corner.clone(),
                max_corner=self.max_corner.clone(),
            )
        elif quarters == 1:  # 90° CCW
            return BatchedAxisAlignedBoundingBox(
                min_corner=torch.stack([-max_y, min_x, min_z], dim=1),
                max_corner=torch.stack([-min_y, max_x, max_z], dim=1),
            )
        elif quarters == 2:  # 180°
            return BatchedAxisAlignedBoundingBox(
                min_corner=torch.stack([-max_x, -max_y, min_z], dim=1),
                max_corner=torch.stack([-min_x, -min_y, max_z], dim=1),
            )
        else:  # 270° CCW / -90° (quarters == 3)
            return BatchedAxisAlignedBoundingBox(
                min_corner=torch.stack([min_y, -max_x, min_z], dim=1),
                max_corner=torch.stack([max_y, -min_x, max_z], dim=1),
            )


def world_bbox_to_min_max_tensors(
    world_bbox: AxisAlignedBoundingBox | BatchedAxisAlignedBoundingBox,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert any object's world-frame AABB to (min_corner, max_corner) tensors of shape (N, 3).

    Use at the start of a relation loss strategy so the same code supports
    single-env (AxisAlignedBoundingBox) and batched (BatchedAxisAlignedBoundingBox)
    without branching. The argument can be any object's world bbox (e.g. parent
    in On/NextTo, or either object in NoCollision).

    Args:
        world_bbox: Single-env or batched axis-aligned bounding box in world frame.
        device: Device for created tensors (used only when converting from AxisAlignedBoundingBox).
        dtype: Dtype for created tensors (used only when converting from AxisAlignedBoundingBox).

    Returns:
        (min_corner, max_corner) each of shape (N, 3), N >= 1.
    """
    if isinstance(world_bbox, AxisAlignedBoundingBox):
        min_c = torch.tensor(world_bbox.min_point, device=device, dtype=dtype).unsqueeze(0)
        max_c = torch.tensor(world_bbox.max_point, device=device, dtype=dtype).unsqueeze(0)
        return min_c, max_c
    min_c = world_bbox.min_corner
    max_c = world_bbox.max_corner
    if min_c.dim() == 1:
        min_c = min_c.unsqueeze(0)
        max_c = max_c.unsqueeze(0)
    return min_c, max_c


def quaternion_to_90_deg_z_quarters(rotation_wxyz: tuple[float, float, float, float], tol_deg: float = 1.0) -> int:
    """Convert a quaternion to 90° rotation quarters around Z axis.

    Only supports rotations that are multiples of 90° around the Z axis.
    Raises AssertionError for any other rotation.

    Args:
        rotation_wxyz: Quaternion as (w, x, y, z).
        tol_deg: Tolerance in degrees for how close the angle must be to a 90° multiple.

    Returns:
        Number of 90° quarters (0, 1, 2, or 3).

    Raises:
        AssertionError: If the quaternion is not a pure Z rotation or not a 90° multiple.
    """
    import math

    w, x, y, z = rotation_wxyz

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

    # Create pose with random position and identity rotation (w=1, x=0, y=0, z=0)
    pose = Pose(position_xyz=tuple(random_position.tolist()), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))

    return pose

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
"""

import torch
from dataclasses import dataclass

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


def quaternion_to_90_deg_z_quarters(rotation_wxyz: tuple[float, float, float, float], tol: float = 1e-3) -> int:
    """Convert a quaternion to 90° rotation quarters around Z axis.

    Only supports rotations that are multiples of 90° around the Z axis.
    Raises AssertionError for any other rotation.

    Args:
        rotation_wxyz: Quaternion as (w, x, y, z).
        tol: Tolerance for floating point comparison.

    Returns:
        Number of 90° quarters (0, 1, 2, or 3).

    Raises:
        AssertionError: If the quaternion is not a 90° rotation around Z.
    """
    w, x, y, z = rotation_wxyz

    # Must be a pure Z rotation (x and y components must be ~0)
    assert (
        abs(x) < tol and abs(y) < tol
    ), f"Only 90° rotations around Z axis are supported. Got quaternion with x={x:.4f}, y={y:.4f} (expected ~0)."

    # Normalize to handle slight variations
    norm = (w * w + z * z) ** 0.5
    if norm < tol:
        raise ValueError(f"Invalid quaternion: w={w}, z={z}")
    w_norm, z_norm = w / norm, z / norm

    # Match to known 90° rotations around Z
    # 0°:   w=1,     z=0      -> quarters=0
    # 90°:  w=0.707, z=0.707  -> quarters=1
    # 180°: w=0,     z=1      -> quarters=2
    # 270°: w=0.707, z=-0.707 -> quarters=3 (or -90°)
    sqrt2_inv = 0.7071067811865476

    if abs(w_norm - 1.0) < tol and abs(z_norm) < tol:
        return 0
    elif abs(w_norm - sqrt2_inv) < tol and abs(z_norm - sqrt2_inv) < tol:
        return 1
    elif abs(w_norm) < tol and abs(abs(z_norm) - 1.0) < tol:
        return 2
    elif abs(w_norm - sqrt2_inv) < tol and abs(z_norm + sqrt2_inv) < tol:
        return 3
    else:
        raise AssertionError(
            "Only 90° rotations around Z axis are supported. "
            f"Got quaternion (w={w:.4f}, x={x:.4f}, y={y:.4f}, z={z:.4f}) which is not a 90° increment."
        )


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

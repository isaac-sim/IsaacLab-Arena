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
class BoundingBox:
    """Represents an axis-aligned bounding box in 3D space."""

    min_point: tuple[float, float, float]
    """Minimum point (x, y, z) of the bounding box."""

    max_point: tuple[float, float, float]
    """Maximum point (x, y, z) of the bounding box."""

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

    def get_corners_aabb_axis_aligned(self, pos: torch.Tensor) -> torch.Tensor:
        """Get 8 corners of the axis-aligned bounding box (AABB) centered at a given position.

        Args:
            pos: Position tensor (x, y, z) to center the bounding box at.

        Returns:
            Tensor of shape (8, 3) containing the 8 corner positions in 3D space.
            Corners are ordered as:
            - Bottom face (z_min): 4 corners going counter-clockwise when viewed from above
            - Top face (z_max): 4 corners going counter-clockwise when viewed from above
        """
        x, y, z = pos[0], pos[1], pos[2]
        width, depth, height = self.size

        # Get the 8 corners of the bounding box
        # Bottom face (z_min)
        return torch.stack([
            torch.tensor([x - width / 2, y - depth / 2, z - height / 2]),  # Bottom-front-left
            torch.tensor([x + width / 2, y - depth / 2, z - height / 2]),  # Bottom-front-right
            torch.tensor([x + width / 2, y + depth / 2, z - height / 2]),  # Bottom-back-right
            torch.tensor([x - width / 2, y + depth / 2, z - height / 2]),  # Bottom-back-left
            # Top face (z_max)
            torch.tensor([x - width / 2, y - depth / 2, z + height / 2]),  # Top-front-left
            torch.tensor([x + width / 2, y - depth / 2, z + height / 2]),  # Top-front-right
            torch.tensor([x + width / 2, y + depth / 2, z + height / 2]),  # Top-back-right
            torch.tensor([x - width / 2, y + depth / 2, z + height / 2]),  # Top-back-left
        ])


def get_random_pose_within_bounding_box(bbox: BoundingBox, seed: int | None = None) -> Pose:
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

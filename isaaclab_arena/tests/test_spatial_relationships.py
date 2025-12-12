# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the spatial relationships API.

These tests verify that the spatial relationship functions correctly compute
bounding boxes and placement poses.
"""

import pytest

from isaaclab_arena.utils.spatial_relationships import BoundingBox, compute_next_to_pose, compute_on_top_of_pose


class TestBoundingBox:
    """Test the BoundingBox dataclass."""

    def test_size(self):
        """Test size calculation."""
        bbox = BoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(1.0, 2.0, 3.0))
        assert bbox.size == (1.0, 2.0, 3.0)

    def test_center(self):
        """Test center calculation."""
        bbox = BoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(2.0, 4.0, 6.0))
        assert bbox.center == (1.0, 2.0, 3.0)

    def test_top_surface_z(self):
        """Test top surface z coordinate."""
        bbox = BoundingBox(min_point=(0.0, 0.0, 1.0), max_point=(1.0, 1.0, 3.0))
        assert bbox.top_surface_z == 3.0

    def test_bottom_surface_z(self):
        """Test bottom surface z coordinate."""
        bbox = BoundingBox(min_point=(0.0, 0.0, 1.0), max_point=(1.0, 1.0, 3.0))
        assert bbox.bottom_surface_z == 1.0


class TestComputeOnTopOfPose:
    """Test the compute_on_top_of_pose function."""

    def test_basic_placement(self):
        """Test basic on-top placement."""
        # Create a 1x1x1 cube as the object
        object_bbox = BoundingBox(min_point=(-0.5, -0.5, -0.5), max_point=(0.5, 0.5, 0.5))
        # Create a 2x2x1 surface as the target (centered at origin)
        target_bbox = BoundingBox(min_point=(-1.0, -1.0, -0.5), max_point=(1.0, 1.0, 0.5))

        pose = compute_on_top_of_pose(object_bbox, target_bbox)

        # Object should be centered on target horizontally
        assert pose.position_xyz[0] == 0.0
        assert pose.position_xyz[1] == 0.0
        # Object bottom (at -0.5 relative to center) should touch target top (at 0.5)
        # So object center should be at 0.5 + 0.5 = 1.0
        assert pose.position_xyz[2] == 1.0
        # No rotation
        assert pose.rotation_wxyz == (1.0, 0.0, 0.0, 0.0)

    def test_placement_with_clearance(self):
        """Test placement with additional clearance."""
        object_bbox = BoundingBox(min_point=(-0.5, -0.5, -0.5), max_point=(0.5, 0.5, 0.5))
        target_bbox = BoundingBox(min_point=(-1.0, -1.0, -0.5), max_point=(1.0, 1.0, 0.5))

        pose = compute_on_top_of_pose(object_bbox, target_bbox, clearance=0.1)

        # Z position should include the clearance
        assert pose.position_xyz[2] == 1.1

    def test_placement_with_offsets(self):
        """Test placement with x/y offsets."""
        object_bbox = BoundingBox(min_point=(-0.5, -0.5, -0.5), max_point=(0.5, 0.5, 0.5))
        target_bbox = BoundingBox(min_point=(-1.0, -1.0, -0.5), max_point=(1.0, 1.0, 0.5))

        pose = compute_on_top_of_pose(object_bbox, target_bbox, x_offset=0.3, y_offset=-0.2)

        assert pose.position_xyz[0] == 0.3
        assert pose.position_xyz[1] == -0.2


class TestComputeNextToPose:
    """Test the compute_next_to_pose function."""

    def test_placement_right(self):
        """Test placement to the right."""
        object_bbox = BoundingBox(min_point=(-0.5, -0.5, -0.5), max_point=(0.5, 0.5, 0.5))
        target_bbox = BoundingBox(min_point=(-1.0, -1.0, -0.5), max_point=(1.0, 1.0, 0.5))

        pose = compute_next_to_pose(object_bbox, target_bbox, side="right", clearance=0.0)

        # right = -Y direction
        # Target bottom edge (in Y) is at -1.0, object top edge should touch it
        # Object center should be at -1.0 - 0.5 = -1.5
        assert pose.position_xyz[0] == 0.0  # Centered in X
        assert pose.position_xyz[1] == -1.5  # At -Y
        assert pose.position_xyz[2] == 0.0  # Bottom aligned

    def test_placement_left(self):
        """Test placement to the left."""
        object_bbox = BoundingBox(min_point=(-0.5, -0.5, -0.5), max_point=(0.5, 0.5, 0.5))
        target_bbox = BoundingBox(min_point=(-1.0, -1.0, -0.5), max_point=(1.0, 1.0, 0.5))

        pose = compute_next_to_pose(object_bbox, target_bbox, side="left", clearance=0.0)

        # left = +Y direction
        # Target top edge (in Y) is at 1.0, object bottom edge should touch it
        # Object center should be at 1.0 + 0.5 = 1.5
        assert pose.position_xyz[0] == 0.0  # Centered in X
        assert pose.position_xyz[1] == 1.5  # At +Y

    def test_placement_front(self):
        """Test placement in front."""
        object_bbox = BoundingBox(min_point=(-0.5, -0.5, -0.5), max_point=(0.5, 0.5, 0.5))
        target_bbox = BoundingBox(min_point=(-1.0, -1.0, -0.5), max_point=(1.0, 1.0, 0.5))

        pose = compute_next_to_pose(object_bbox, target_bbox, side="front", clearance=0.0)

        # front = -X direction
        # Target left edge (in X) is at -1.0, object right edge should touch it
        # Object center should be at -1.0 - 0.5 = -1.5
        assert pose.position_xyz[0] == -1.5  # At -X
        assert pose.position_xyz[1] == 0.0  # Centered in Y

    def test_placement_back(self):
        """Test placement behind."""
        object_bbox = BoundingBox(min_point=(-0.5, -0.5, -0.5), max_point=(0.5, 0.5, 0.5))
        target_bbox = BoundingBox(min_point=(-1.0, -1.0, -0.5), max_point=(1.0, 1.0, 0.5))

        pose = compute_next_to_pose(object_bbox, target_bbox, side="back", clearance=0.0)

        # back = +X direction
        # Target right edge (in X) is at 1.0, object left edge should touch it
        # Object center should be at 1.0 + 0.5 = 1.5
        assert pose.position_xyz[0] == 1.5  # At +X
        assert pose.position_xyz[1] == 0.0  # Centered in Y

    def test_placement_with_clearance(self):
        """Test placement with clearance."""
        object_bbox = BoundingBox(min_point=(-0.5, -0.5, -0.5), max_point=(0.5, 0.5, 0.5))
        target_bbox = BoundingBox(min_point=(-1.0, -1.0, -0.5), max_point=(1.0, 1.0, 0.5))

        pose = compute_next_to_pose(object_bbox, target_bbox, side="right", clearance=0.1)

        # right = -Y, with 0.1 clearance, object center should be at -1.0 - 0.1 - 0.5 = -1.6
        assert pose.position_xyz[1] == -1.6

    def test_invalid_side(self):
        """Test that invalid side raises ValueError."""
        object_bbox = BoundingBox(min_point=(-0.5, -0.5, -0.5), max_point=(0.5, 0.5, 0.5))
        target_bbox = BoundingBox(min_point=(-1.0, -1.0, -0.5), max_point=(1.0, 1.0, 0.5))

        with pytest.raises(ValueError, match="Invalid side"):
            compute_next_to_pose(object_bbox, target_bbox, side="invalid")

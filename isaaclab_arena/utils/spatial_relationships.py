# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
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

from dataclasses import dataclass

from pxr import Gf, Usd, UsdGeom

from isaaclab_arena.utils.pose import Pose


@dataclass
class BoundingBox:
    """Represents an axis-aligned bounding box in 3D space."""

    min_point: tuple[float, float, float]
    """Minimum point (x, y, z) of the bounding box."""

    max_point: tuple[float, float, float]
    """Maximum point (x, y, z) of the bounding box."""

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


def compute_bounding_box_from_usd(
    usd_path: str,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    pose: Pose | None = None,
) -> BoundingBox:
    """
    Compute the world-space bounding box of a USD asset.

    Args:
        usd_path: Path to the USD file.
        scale: Scale to apply to the asset (x, y, z).
        pose: Optional pose of the asset. If None, uses identity pose.

    Returns:
        BoundingBox containing the min and max points in world space.
    """
    # Open the USD stage
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise ValueError(f"Failed to open USD file: {usd_path}")

    # Get the default prim (or pseudo root if no default prim)
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        default_prim = stage.GetPseudoRoot()

    # Compute the bounding box using USD's built-in functionality
    # This computes the bounding box in the local space of the prim
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(default_prim)

    # Get the range (bounding box)
    bbox_range = bbox.ComputeAlignedBox()
    min_point = bbox_range.GetMin()
    max_point = bbox_range.GetMax()

    # Apply scale
    min_point = Gf.Vec3d(min_point[0] * scale[0], min_point[1] * scale[1], min_point[2] * scale[2])
    max_point = Gf.Vec3d(max_point[0] * scale[0], max_point[1] * scale[1], max_point[2] * scale[2])

    # Apply pose transformation if provided
    if pose is not None:
        # Transform the bounding box corners by the pose
        # For simplicity in MVP, we'll just translate the bbox by the position
        # A more sophisticated implementation would rotate the bbox as well
        min_point = Gf.Vec3d(
            min_point[0] + pose.position_xyz[0],
            min_point[1] + pose.position_xyz[1],
            min_point[2] + pose.position_xyz[2],
        )
        max_point = Gf.Vec3d(
            max_point[0] + pose.position_xyz[0],
            max_point[1] + pose.position_xyz[1],
            max_point[2] + pose.position_xyz[2],
        )

    return BoundingBox(
        min_point=(min_point[0], min_point[1], min_point[2]),
        max_point=(max_point[0], max_point[1], max_point[2]),
    )


def compute_on_top_of_pose(
    object_bbox: BoundingBox,
    target_bbox: BoundingBox,
    clearance: float = 0.0,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
) -> Pose:
    """
    Compute a pose that places an object on top of a target object.

    The object will be placed such that its bottom surface sits on the target's top surface,
    centered on the target (unless offsets are provided).

    Args:
        object_bbox: Bounding box of the object to be placed.
        target_bbox: Bounding box of the target (base) object.
        clearance: Additional vertical clearance between objects (default: 0.0).
        x_offset: Horizontal offset in x direction from center (default: 0.0).
        y_offset: Horizontal offset in y direction from center (default: 0.0).

    Returns:
        Pose for the object that places it on top of the target.
    """
    # Calculate the z position: target's top + clearance + object's height/2
    # We need to account for the object's center being at its geometric center
    object_half_height = object_bbox.size[2] / 2.0

    # The bottom of the object should be at the top of the target
    z_position = target_bbox.top_surface_z + clearance + object_half_height

    # Center the object on the target (with optional offsets)
    x_position = target_bbox.center[0] + x_offset
    y_position = target_bbox.center[1] + y_offset

    # Return pose with identity rotation (no rotation by default)
    return Pose(
        position_xyz=(x_position, y_position, z_position),
        rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
    )


def compute_next_to_pose(
    object_bbox: BoundingBox,
    target_bbox: BoundingBox,
    side: str = "right",
    clearance: float = 0.01,
    align_bottom: bool = True,
) -> Pose:
    """
    Compute a pose that places an object next to a target object.

    **Important**: Directions are defined in the **world coordinate frame**, not relative
    to the target object's orientation:
    - "right" = -Y direction in world frame
    - "left" = +Y direction in world frame
    - "front" = -X direction in world frame
    - "back" = +X direction in world frame

    This means the placement does NOT account for the target object's rotation.
    This is a limitation of the MVP implementation.

    Args:
        object_bbox: Bounding box of the object to be placed.
        target_bbox: Bounding box of the target object.
        side: Which side to place the object ("left", "right", "front", "back").
              Directions are in world frame, not relative to target's orientation.
        clearance: Horizontal clearance between objects (default: 0.01).
        align_bottom: If True, align bottoms; if False, center vertically (default: True).

    Returns:
        Pose for the object that places it next to the target.

    Note:
        For rotated objects, "right" will still be -Y in world coordinates,
        which may not correspond to the intuitive "right side" of the object.
    """
    # Calculate the base z position
    if align_bottom:
        # Align the bottom surfaces
        object_half_height = object_bbox.size[2] / 2.0
        z_position = target_bbox.bottom_surface_z + object_half_height
    else:
        # Center vertically with the target
        z_position = target_bbox.center[2]

    # Calculate horizontal position based on side
    # Convention: right = -Y, left = +Y, front = -X, back = +X
    object_half_width = object_bbox.size[0] / 2.0
    object_half_depth = object_bbox.size[1] / 2.0
    target_half_width = target_bbox.size[0] / 2.0
    target_half_depth = target_bbox.size[1] / 2.0

    if side == "right":
        # right = -Y
        x_position = target_bbox.center[0]
        y_position = target_bbox.center[1] - target_half_depth - clearance - object_half_depth
    elif side == "left":
        # left = +Y
        x_position = target_bbox.center[0]
        y_position = target_bbox.center[1] + target_half_depth + clearance + object_half_depth
    elif side == "front":
        # front = -X
        x_position = target_bbox.center[0] - target_half_width - clearance - object_half_width
        y_position = target_bbox.center[1]
    elif side == "back":
        # back = +X
        x_position = target_bbox.center[0] + target_half_width + clearance + object_half_width
        y_position = target_bbox.center[1]
    else:
        raise ValueError(f"Invalid side: {side}. Must be 'left', 'right', 'front', or 'back'.")

    return Pose(
        position_xyz=(x_position, y_position, z_position),
        rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
    )

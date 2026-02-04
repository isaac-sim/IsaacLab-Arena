# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from isaaclab_arena.utils.pose import PoseRange

if TYPE_CHECKING:
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_reference import ObjectReference


class Side(Enum):
    """Side of an object for spatial relationships."""

    FRONT = "front"  # -Y
    BACK = "back"  # +Y
    LEFT = "left"  # -X
    RIGHT = "right"  # +X


class RelationBase:
    """Base for all relation-like concepts on objects.

    This is the common base class for both spatial relations (On, NextTo, etc.)
    and markers (IsAnchor). It allows the Object class to store both types
    in its relations list.
    """

    pass


class Relation(RelationBase):
    """Base class for spatial relationships between objects."""

    def __init__(self, parent: Object | ObjectReference, relation_loss_weight: float = 1.0):
        """
        Args:
            parent: The parent asset in the relationship (Object or ObjectReference).
            relation_loss_weight: Weight for the relationship loss function.
        """
        self.parent = parent
        self.relation_loss_weight = relation_loss_weight


class NextTo(Relation):
    """Represents a 'next to' relationship between objects.

    This relation specifies that a child object should be placed adjacent to
    the parent object on a specified side, at a given distance.

    Note: Loss computation is handled by NextToLossStrategy in relation_loss_strategies.py.
    """

    def __init__(
        self,
        parent: Object | ObjectReference,
        relation_loss_weight: float = 1.0,
        distance_m: float = 0.05,
        side: Side = Side.RIGHT,
    ):
        """
        Args:
            parent: The parent asset that this object should be placed next to.
            relation_loss_weight: Weight for the relationship loss function.
            distance_m: Target distance from parent's boundary in meters (default: 5cm).
            side: Which side to place object (default: Side.RIGHT).
        """
        super().__init__(parent, relation_loss_weight)
        assert distance_m > 0.0, f"Distance must be positive, got {distance_m}"
        self.distance_m = distance_m
        self.side = side


class On(Relation):
    """Represents an 'on top of' relationship between objects.

    This relation specifies that a child object should be placed on top of
    the parent object, with X/Y bounded within the parent's extent and Z
    positioned on the parent's top surface.

    Note: Loss computation is handled by OnLossStrategy in relation_loss_strategies.py.
    """

    def __init__(
        self,
        parent: Object | ObjectReference,
        relation_loss_weight: float = 1.0,
        clearance_m: float = 0.01,
    ):
        """
        Args:
            parent: The parent asset that this object should be placed on top of.
            relation_loss_weight: Weight for the relationship loss function.
            clearance_m: Safety clearance above parent's surface in meters (default: 1cm).
        """
        super().__init__(parent, relation_loss_weight)
        assert clearance_m >= 0.0, f"Clearance must be non-negative, got {clearance_m}"
        self.clearance_m = clearance_m


class IsAnchor(RelationBase):
    """Marker indicating this object is an anchor for relation solving.

    Anchor objects are fixed references that won't be optimized during
    relation solving. Multiple objects can be marked as anchors.
    Each anchor must have an initial_pose set before calling ObjectPlacer.place().

    Usage:
        table.set_initial_pose(Pose(position_xyz=(1.0, 0.0, 0.0), ...))
        table.add_relation(IsAnchor())  # Mark as anchor

        chair.set_initial_pose(Pose(position_xyz=(2.0, 0.0, 0.0), ...))
        chair.add_relation(IsAnchor())  # Another anchor

        mug.add_relation(On(table))
        bin.add_relation(NextTo(chair))
    """

    pass


class RandomAroundSolution(RelationBase):
    """Marker indicating the solver solution should be used as center of a PoseRange.

    When ObjectPlacer applies positions, objects with this marker will get a PoseRange
    (enabling randomization at environment reset) instead of a fixed Pose.

    The half extents define a box centered on the solved position. At each environment
    reset, a random position within this box will be sampled.

    Note: This is NOT a spatial relation - the RelationSolver ignores it. It only
    affects how ObjectPlacer applies the solved position to the object.

    Usage:
        box.add_relation(On(desk))
        box.add_relation(RandomAroundSolution(x_half_m=0.1, y_half_m=0.1))
        # -> ObjectPlacer sets a PoseRange spanning ±0.1m in X and Y around solved position
    """

    def __init__(
        self,
        x_half_m: float = 0.0,
        y_half_m: float = 0.0,
        z_half_m: float = 0.0,
        roll_half_rad: float = 0.0,
        pitch_half_rad: float = 0.0,
        yaw_half_rad: float = 0.0,
    ):
        """
        Args:
            x_half_m: Half-extent in X direction (meters). Position will be randomized ±x_half_m.
            y_half_m: Half-extent in Y direction (meters). Position will be randomized ±y_half_m.
            z_half_m: Half-extent in Z direction (meters). Position will be randomized ±z_half_m.
            roll_half_rad: Half-extent for roll (radians). Rotation will be randomized ±roll_half_rad.
            pitch_half_rad: Half-extent for pitch (radians). Rotation will be randomized ±pitch_half_rad.
            yaw_half_rad: Half-extent for yaw (radians). Rotation will be randomized ±yaw_half_rad.
        """
        self.x_half_m = x_half_m
        self.y_half_m = y_half_m
        self.z_half_m = z_half_m
        self.roll_half_rad = roll_half_rad
        self.pitch_half_rad = pitch_half_rad
        self.yaw_half_rad = yaw_half_rad

    def to_pose_range(self, position: tuple[float, float, float]) -> PoseRange:
        """Create a PoseRange centered on the given position.

        Args:
            position: Center position (x, y, z) for the range.

        Returns:
            PoseRange spanning ± half-extents around the position.
        """
        return PoseRange(
            position_xyz_min=(
                position[0] - self.x_half_m,
                position[1] - self.y_half_m,
                position[2] - self.z_half_m,
            ),
            position_xyz_max=(
                position[0] + self.x_half_m,
                position[1] + self.y_half_m,
                position[2] + self.z_half_m,
            ),
            rpy_min=(
                -self.roll_half_rad,
                -self.pitch_half_rad,
                -self.yaw_half_rad,
            ),
            rpy_max=(
                self.roll_half_rad,
                self.pitch_half_rad,
                self.yaw_half_rad,
            ),
        )


class AtPosition(RelationBase):
    """Constrains object to specific world coordinates.

    This is a unary relation (no parent) that pins an object's position to
    specific x, y, and/or z world coordinates. Any axis set to None is
    unconstrained by this relation (allowing other relations like On to
    control that axis).

    Note: Loss computation is handled by AtPositionLossStrategy in relation_loss_strategies.py.

    Usage:
        # Pin object to x=0.5, y=1.0 in world coords (z controlled by On relation)
        mug.add_relation(On(table))
        mug.add_relation(AtPosition(x=0.5, y=1.0))
    """

    def __init__(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        relation_loss_weight: float = 1.0,
    ):
        """
        Args:
            x: Target x world coordinate, or None to leave unconstrained.
            y: Target y world coordinate, or None to leave unconstrained.
            z: Target z world coordinate, or None to leave unconstrained.
            relation_loss_weight: Weight for the relationship loss function.
        """
        assert (
            x is not None or y is not None or z is not None
        ), "At least one of x, y, or z must be specified for AtPosition"
        self.x = x
        self.y = y
        self.z = z
        self.relation_loss_weight = relation_loss_weight


def get_anchor_objects(objects: list[Object | ObjectReference]) -> list[Object | ObjectReference]:
    """Get all anchor objects from a list of objects.

    Anchor objects are marked with IsAnchor() relation and serve as
    fixed reference points for relation solving.

    Args:
        objects: List of objects to filter.

    Returns:
        List of anchor objects (may be empty if no anchors found).
    """
    return [obj for obj in objects if any(isinstance(r, IsAnchor) for r in obj.get_relations())]

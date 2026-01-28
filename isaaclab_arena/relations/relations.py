# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_arena.assets.object import Object


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

    def __init__(self, parent: Object, relation_loss_weight: float = 1.0):
        """
        Args:
            parent: The parent asset in the relationship.
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
        parent: Object,
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
        parent: Object,
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


def get_anchor_objects(objects: list[Object]) -> list[Object]:
    """Get all anchor objects from a list of objects.

    Anchor objects are marked with IsAnchor() relation and serve as
    fixed reference points for relation solving.

    Args:
        objects: List of objects to filter.

    Returns:
        List of anchor objects (may be empty if no anchors found).
    """
    return [obj for obj in objects if any(isinstance(r, IsAnchor) for r in obj.get_relations())]

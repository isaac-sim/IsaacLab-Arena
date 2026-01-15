# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import TYPE_CHECKING

# Avoid circular import by using TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab_arena.assets.dummy_object import DummyObject


class Side(Enum):
    """Side of an object for spatial relationships."""

    FRONT = "front"  # -Y
    BACK = "back"  # +Y
    LEFT = "left"  # -X
    RIGHT = "right"  # +X


class Relation:
    """Base class for spatial relationships between objects."""

    def __init__(self, parent: "DummyObject", relation_loss_weight: float = 1.0):
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
        parent: "DummyObject",
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
        parent: "DummyObject",
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

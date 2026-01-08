# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Spatial relationship classes for object placement constraints.

These classes are pure data containers representing relationships between objects.
Loss computation is handled by LossStrategy classes in loss_strategies.py.
"""

from typing import TYPE_CHECKING

# Avoid circular import by using TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab_arena.assets.dummy_object import DummyObject


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

    Note: Loss computation is handled by NextToLossStrategy in loss_strategies.py.
    """

    def __init__(
        self,
        parent: "DummyObject",
        relation_loss_weight: float = 1.0,
        distance_m: float = 0.05,
        side: str = "right",
    ):
        """
        Args:
            parent: The parent asset that this object should be placed next to.
            relation_loss_weight: Weight for the relationship loss function.
            distance_m: Target distance from parent's boundary in meters (default: 5cm).
            side: Which side to place object: "front" (-Y), "back" (+Y),
                  "left" (-X), or "right" (+X).
        """
        super().__init__(parent, relation_loss_weight)
        assert distance_m >= 0.0, f"Distance must be non-negative, got {distance_m}"
        self.distance_m = distance_m
        assert side in ["front", "back", "left", "right"]
        self.side = side

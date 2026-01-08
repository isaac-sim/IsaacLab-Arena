# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Spatial relationship classes for object placement constraints.
"""

import torch
from typing import TYPE_CHECKING

from isaaclab_arena.utils.relation_loss import linear_band_loss, single_boundary_linear_loss

# Avoid circular import by using TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab_arena.assets.dummy_object import DummyObject


class Relation:
    """Base class for spatial relationships between objects."""

    def __init__(self, parent: "DummyObject", relation_loss_weight: float):
        """
        Args:
            parent: The parent asset in the relationship.
            relation_loss_weight: Weight for the relationship loss function.
        """
        self.parent = parent
        self.relation_loss_weight = relation_loss_weight

    def compute_relation_loss(
        self, child: "DummyObject", child_pos: torch.Tensor | None = None, parent_pos: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute the loss for this relationship constraint.

        Args:
            child: The child object in the relationship.
            child_pos: Optional position tensor (x, y, z) for the child object.
                      If None, uses child's initial_pose.
            parent_pos: Optional position tensor (x, y, z) for the parent object.
                       If None, uses parent's initial_pose.

        Returns:
            Scalar loss tensor representing the constraint violation.
        """
        pass


class NextTo(Relation):
    """Represents a 'next to' relationship between objects.

    This relation enforces that a child object should be placed at a target distance
    from the parent object's boundary, with a tolerance band. The loss is near-zero
    within the acceptable distance range and increases exponentially outside it.
    """

    def __init__(
        self,
        parent: "DummyObject",
        relation_loss_weight: float = 1.0,
        distance: float = 0.05,
        tolerance: float = 0.01,
        slope: float = 10.0,
        side: str = "right",
    ):
        """
        Args:
            parent: The parent asset that this object should be placed next to.
            relation_loss_weight: Weight for the relationship loss function.
            distance: Target distance from parent's boundary in meters (default: 5cm).
            tolerance: Tolerance band around target distance in meters (default: 1cm).
            slope: Gradient magnitude for linear loss (default: 10.0).
                   Loss increases by `slope` per meter of violation.
            side: Which side to place object: "front" (-X), "back" (+X), "left" (+Y), or "right" (-Y).
        """
        super().__init__(parent, relation_loss_weight)
        self.distance = distance
        self.tolerance = tolerance
        self.slope = slope
        self.side = side
        print("NextTo side initialized", side)

    def compute_relation_loss(
        self, child: "DummyObject", child_pos: torch.Tensor | None = None, parent_pos: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute the loss for the 'next to' relationship constraint.

        This loss ensures the child is placed at the target distance from the parent's
        boundary, with exponential penalties for deviations outside the tolerance band.

        Args:
            child: The child object that should be placed next to the parent.
            child_pos: Optional position tensor (x, y, z) for the child object.
                      If None, uses child's initial_pose.
            parent_pos: Optional position tensor (x, y, z) for the parent object.
                       If None, uses parent's initial_pose.

        Returns:
            Scalar loss tensor representing the constraint violation.
        """
        # Get parent position
        if parent_pos is None:
            parent_pose = self.parent.get_initial_pose()
            assert parent_pose is not None, f"Parent pose is None for {self.parent.name}"
            parent_pos = torch.tensor(parent_pose.position_xyz, dtype=torch.float32)

        # Get child position
        if child_pos is None:
            child_pose = child.get_initial_pose()
            assert child_pose is not None, f"Child pose is None for {child.name}"
            child_pos = torch.tensor(child_pose.position_xyz, dtype=torch.float32)

        # Get bounding boxes
        parent_bbox = self.parent.get_bounding_box()
        child_bbox = child.get_bounding_box()

        # Just type assertions for type checker
        assert parent_pos is not None
        assert child_pos is not None

        # We set a half plane loss on the right side of the parent.
        parent_right_bound = parent_pos[0] + parent_bbox.size[0] / 2
        right_side_loss = single_boundary_linear_loss(
            child_pos[0],
            parent_right_bound,
            slope=self.slope,
            penalty_side="less",
        )

        # We set a band loss on the top and bottom sides of the parent.
        parent_top_bound = parent_pos[1] + parent_bbox.size[1] / 2
        parent_bottom_bound = parent_pos[1] - parent_bbox.size[1] / 2
        top_bottom_band_loss = linear_band_loss(child_pos[1], parent_top_bound, parent_bottom_bound)

        # Add a loss that pushed the object into a certain distance from the parent.
        assert self.distance >= 0.0, f"Next to distance must be non-negative, got {self.distance}"

        lower_bound = parent_pos[0] + parent_bbox.size[0] / 2 + self.distance
        upper_bound = lower_bound + child_bbox.size[0]

        next_to_distance_loss = linear_band_loss(child_pos[0], lower_bound, upper_bound)

        total_loss = right_side_loss + top_bottom_band_loss + next_to_distance_loss
        return total_loss

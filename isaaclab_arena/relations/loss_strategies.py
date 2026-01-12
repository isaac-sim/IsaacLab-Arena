# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from isaaclab_arena.relations.relation_loss import (
    linear_band_loss,
    single_boundary_linear_loss,
    single_point_linear_loss,
)
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

if TYPE_CHECKING:
    from isaaclab_arena.relations.relations import NextTo, On, Relation

from isaaclab_arena.relations.relations import Side


class LossStrategy(ABC):
    """Abstract base class for relation loss computation strategies."""

    @abstractmethod
    def compute_loss(
        self,
        relation: "Relation",
        child_pos: torch.Tensor,
        parent_pos: torch.Tensor,
        child_bbox: AxisAlignedBoundingBox,
        parent_bbox: AxisAlignedBoundingBox,
    ) -> torch.Tensor:
        """Compute the loss for a relation constraint.

        Args:
            relation: The relation object containing relationship metadata.
            child_pos: Child object position tensor (x, y, z).
            parent_pos: Parent object position tensor (x, y, z).
            child_bbox: Child object axis-aligned bounding box.
            parent_bbox: Parent object axis-aligned bounding box.

        Returns:
            Scalar loss tensor representing the constraint violation.
        """
        pass


class NextToLossStrategy(LossStrategy):
    """Loss strategy for NextTo relations.

    Computes loss based on:
    1. Half-plane constraint to ensure child is on correct side of parent
    2. Band constraint to keep child aligned with parent's extent
    3. Distance constraint to position child at target distance from parent
    """

    def __init__(self, slope: float = 10.0):
        """
        Args:
            slope: Gradient magnitude for linear loss (default: 10.0).
                   Loss increases by `slope` per meter of violation.
        """
        self.slope = slope

    def compute_loss(
        self,
        relation: "NextTo",
        child_pos: torch.Tensor,
        parent_pos: torch.Tensor,
        child_bbox: AxisAlignedBoundingBox,
        parent_bbox: AxisAlignedBoundingBox,
    ) -> torch.Tensor:
        """Compute loss for NextTo relation.

        Currently only implements "right" side placement.

        Args:
            relation: NextTo relation with side and distance attributes.
            child_pos: Child object position tensor (x, y, z).
            parent_pos: Parent object position tensor (x, y, z).
            child_bbox: Child object axis-aligned bounding box.
            parent_bbox: Parent object axis-aligned bounding box.

        Returns:
            Weighted loss tensor.
        """
        side = relation.side
        distance = relation.distance_m

        if side != Side.RIGHT:
            # TODO(cvolk): Implement support for other sides and make generic.
            raise NotImplementedError(f"Side '{side}' not yet implemented, only 'right' is supported")

        # 1. Half-plane loss: child must be to the right of parent's right edge
        parent_right_bound = parent_pos[0] + parent_bbox.size[0] / 2
        right_side_loss = single_boundary_linear_loss(
            child_pos[0],
            parent_right_bound,
            slope=self.slope,
            penalty_side="less",
        )

        # 2. Band loss: child must be within parent's Y extent
        parent_top_bound = parent_pos[1] + parent_bbox.size[1] / 2
        parent_bottom_bound = parent_pos[1] - parent_bbox.size[1] / 2
        top_bottom_band_loss = linear_band_loss(child_pos[1], parent_top_bound, parent_bottom_bound, slope=self.slope)

        # 3. Distance loss: child should be at exact target distance from parent
        assert distance >= 0.0, f"NextTo distance must be non-negative, got {distance}"

        # Target X: parent's right edge + distance + child's half-width
        target_x = parent_pos[0] + parent_bbox.size[0] / 2 + distance + child_bbox.size[0] / 2
        next_to_distance_loss = single_point_linear_loss(child_pos[0], target_x, slope=self.slope)

        total_loss = right_side_loss + top_bottom_band_loss + next_to_distance_loss
        return relation.relation_loss_weight * total_loss


class OnLossStrategy(LossStrategy):
    """Loss strategy for On relations.

    Computes loss based on:
    1. X band constraint to keep child within parent's X extent
    2. Y band constraint to keep child within parent's Y extent
    3. Z point constraint to position child on parent's top surface + clearance
    """

    def __init__(self, slope: float = 10.0):
        """
        Args:
            slope: Gradient magnitude for linear loss (default: 10.0).
                   Loss increases by `slope` per meter of violation.
        """
        self.slope = slope

    def compute_loss(
        self,
        relation: "On",
        child_pos: torch.Tensor,
        parent_pos: torch.Tensor,
        child_bbox: AxisAlignedBoundingBox,
        parent_bbox: AxisAlignedBoundingBox,
    ) -> torch.Tensor:
        """Compute loss for On relation.

        Args:
            relation: On relation with clearance_m attribute.
            child_pos: Child object position tensor (x, y, z).
            parent_pos: Parent object position tensor (x, y, z).
            child_bbox: Child object axis-aligned bounding box.
            parent_bbox: Parent object axis-aligned bounding box.

        Returns:
            Weighted loss tensor.
        """
        # 1. X band loss: child center within parent's X extent
        x_band_loss = linear_band_loss(
            child_pos[0],
            lower_bound=parent_pos[0] - parent_bbox.size[0] / 2,
            upper_bound=parent_pos[0] + parent_bbox.size[0] / 2,
            slope=self.slope,
        )

        # 2. Y band loss: child center within parent's Y extent
        y_band_loss = linear_band_loss(
            child_pos[1],
            lower_bound=parent_pos[1] - parent_bbox.size[1] / 2,
            upper_bound=parent_pos[1] + parent_bbox.size[1] / 2,
            slope=self.slope,
        )

        # 3. Z point loss: child bottom = parent top + clearance
        # Target Z = parent_top + clearance + child_half_height
        target_z = parent_pos[2] + parent_bbox.size[2] / 2 + relation.clearance_m + child_bbox.size[2] / 2
        z_loss = single_point_linear_loss(child_pos[2], target_z, slope=self.slope)

        total_loss = x_band_loss + y_band_loss + z_loss
        return relation.relation_loss_weight * total_loss

# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from isaaclab_arena.relations.loss_primitives import (
    linear_band_loss,
    single_boundary_linear_loss,
    single_point_linear_loss,
)
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

if TYPE_CHECKING:
    from isaaclab_arena.relations.relations import NextTo, On, Relation

from isaaclab_arena.relations.relations import Side


class RelationLossStrategy(ABC):
    """Abstract base class for relation loss computation strategies.

    Loss strategies compute constraints using world-space extents:
        world_min = position + bbox.min_point
        world_max = position + bbox.max_point
    """

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
            child_pos: Child object position tensor (x, y, z) in world coords.
            parent_pos: Parent object position tensor (x, y, z) in world coords.
            child_bbox: Child object local bounding box (extents relative to origin).
            parent_bbox: Parent object local bounding box (extents relative to origin).

        Returns:
            Scalar loss tensor representing the constraint violation.
        """
        pass


class NextToLossStrategy(RelationLossStrategy):
    """Loss strategy for NextTo relations.

    Computes loss based on:
    1. Half-plane constraint to ensure child is on correct side of parent
    2. Band constraint to keep child aligned with parent's extent
    3. Distance constraint to position child at target distance from parent
    """

    def __init__(self, slope: float = 10.0, debug: bool = False):
        """
        Args:
            slope: Gradient magnitude for linear loss (default: 10.0).
                   Loss increases by `slope` per meter of violation.
            debug: If True, print detailed loss component breakdown.
        """
        self.slope = slope
        self.debug = debug

    def compute_loss(
        self,
        relation: "NextTo",
        child_pos: torch.Tensor,
        parent_pos: torch.Tensor,
        child_bbox: AxisAlignedBoundingBox,
        parent_bbox: AxisAlignedBoundingBox,
    ) -> torch.Tensor:
        """Compute loss for NextTo relation.

        Uses world-space extents (position + bbox.min/max) for origin-agnostic computation.
        Currently only implements "right" side placement.

        Args:
            relation: NextTo relation with side and distance attributes.
            child_pos: Child object position tensor (x, y, z) in world coords.
            parent_pos: Parent object position tensor (x, y, z) in world coords.
            child_bbox: Child object local bounding box.
            parent_bbox: Parent object local bounding box.

        Returns:
            Weighted loss tensor.
        """
        side = relation.side
        distance = relation.distance_m

        if side != Side.RIGHT:
            # TODO(cvolk): Implement support for other sides and make generic.
            raise NotImplementedError(f"Side '{side}' not yet implemented, only 'right' is supported")

        # Compute world-space extents: world = position + local_offset
        parent_x_max = parent_pos[0] + parent_bbox.max_point[0]  # Right edge
        parent_y_min = parent_pos[1] + parent_bbox.min_point[1]  # Front edge
        parent_y_max = parent_pos[1] + parent_bbox.max_point[1]  # Back edge

        # 1. Half-plane loss: child must be to the right of parent's right edge
        right_side_loss = single_boundary_linear_loss(
            child_pos[0],
            parent_x_max,
            slope=self.slope,
            penalty_side="less",
        )

        # 2. Band loss: child must be within parent's Y extent
        top_bottom_band_loss = linear_band_loss(
            child_pos[1],
            lower_bound=parent_y_min,
            upper_bound=parent_y_max,
            slope=self.slope,
        )

        # 3. Distance loss: child's left edge should be at distance from parent's right edge
        assert distance >= 0.0, f"NextTo distance must be non-negative, got {distance}"
        # child_x_min = child_pos[0] + child_bbox.min_point[0], so:
        # target: child_pos[0] + child_bbox.min_point[0] = parent_x_max + distance
        # target_child_pos_x = parent_x_max + distance - child_bbox.min_point[0]
        target_x = parent_x_max + distance - child_bbox.min_point[0]
        next_to_distance_loss = single_point_linear_loss(child_pos[0], target_x, slope=self.slope)

        if self.debug:
            print(
                f"    [NextTo] Side: child_x={child_pos[0].item():.4f}, parent_right={parent_x_max.item():.4f},"
                f" loss={right_side_loss.item():.6f}"
            )
            print(
                f"    [NextTo] Y band: child_y={child_pos[1].item():.4f}, range=[{parent_y_min.item():.4f},"
                f" {parent_y_max.item():.4f}], loss={top_bottom_band_loss.item():.6f}"
            )
            print(
                f"    [NextTo] Distance: child_x={child_pos[0].item():.4f}, target_x={target_x.item():.4f},"
                f" loss={next_to_distance_loss.item():.6f}"
            )

        total_loss = right_side_loss + top_bottom_band_loss + next_to_distance_loss
        return relation.relation_loss_weight * total_loss


class OnLossStrategy(RelationLossStrategy):
    """Loss strategy for On relations.

    Computes loss based on:
    1. X band constraint to keep child within parent's X extent
    2. Y band constraint to keep child within parent's Y extent
    3. Z point constraint to position child on parent's top surface + clearance
    """

    def __init__(self, slope: float = 10.0, debug: bool = False):
        """
        Args:
            slope: Gradient magnitude for linear loss (default: 10.0).
                   Loss increases by `slope` per meter of violation.
            debug: If True, print detailed loss component breakdown.
        """
        self.slope = slope
        self.debug = debug

    def compute_loss(
        self,
        relation: "On",
        child_pos: torch.Tensor,
        parent_pos: torch.Tensor,
        child_bbox: AxisAlignedBoundingBox,
        parent_bbox: AxisAlignedBoundingBox,
    ) -> torch.Tensor:
        """Compute loss for On relation.

        Uses world-space extents (position + bbox.min/max) for origin-agnostic computation.

        Args:
            relation: On relation with clearance_m attribute.
            child_pos: Child object position tensor (x, y, z) in world coords.
            parent_pos: Parent object position tensor (x, y, z) in world coords.
            child_bbox: Child object local bounding box.
            parent_bbox: Parent object local bounding box.

        Returns:
            Weighted loss tensor.
        """
        # Compute parent world-space extents
        parent_x_min = parent_pos[0] + parent_bbox.min_point[0]
        parent_x_max = parent_pos[0] + parent_bbox.max_point[0]
        parent_y_min = parent_pos[1] + parent_bbox.min_point[1]
        parent_y_max = parent_pos[1] + parent_bbox.max_point[1]
        parent_z_max = parent_pos[2] + parent_bbox.max_point[2]  # Top surface

        # Compute valid position ranges such that child's entire footprint is within parent
        # Child left edge = child_pos[0] + child_bbox.min_point[0], must be >= parent_x_min
        # Child right edge = child_pos[0] + child_bbox.max_point[0], must be <= parent_x_max
        valid_x_min = parent_x_min - child_bbox.min_point[0]  # child's left at parent's left
        valid_x_max = parent_x_max - child_bbox.max_point[0]  # child's right at parent's right
        valid_y_min = parent_y_min - child_bbox.min_point[1]
        valid_y_max = parent_y_max - child_bbox.max_point[1]

        # 1. X band loss: child's footprint entirely within parent's X extent
        x_band_loss = linear_band_loss(
            child_pos[0],
            lower_bound=valid_x_min,
            upper_bound=valid_x_max,
            slope=self.slope,
        )

        # 2. Y band loss: child's footprint entirely within parent's Y extent
        y_band_loss = linear_band_loss(
            child_pos[1],
            lower_bound=valid_y_min,
            upper_bound=valid_y_max,
            slope=self.slope,
        )

        # 3. Z point loss: child bottom = parent top + clearance
        target_z = parent_z_max + relation.clearance_m - child_bbox.min_point[2]
        z_loss = single_point_linear_loss(child_pos[2], target_z, slope=self.slope)

        if self.debug:
            print(
                f"    [On] X: child_pos={child_pos[0].item():.4f}, valid_range=[{valid_x_min.item():.4f},"
                f" {valid_x_max.item():.4f}], loss={x_band_loss.item():.6f}"
            )
            print(
                f"    [On] Y: child_pos={child_pos[1].item():.4f}, valid_range=[{valid_y_min.item():.4f},"
                f" {valid_y_max.item():.4f}], loss={y_band_loss.item():.6f}"
            )
            print(
                f"    [On] Z: child_pos={child_pos[2].item():.4f}, target={target_z.item():.4f},"
                f" loss={z_loss.item():.6f}"
            )

        total_loss = x_band_loss + y_band_loss + z_loss
        return relation.relation_loss_weight * total_loss

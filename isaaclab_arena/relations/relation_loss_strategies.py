# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
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


@dataclass(frozen=True)
class SideConfig:
    """Configuration for computing NextTo loss for a given side.

    Attributes:
        primary_axis: Index of the axis along which child is placed (0=X, 1=Y).
                      The band axis is always the perpendicular axis (1 - primary_axis).
        direction: +1 if child should be in positive direction from parent (RIGHT, BACK),
                   -1 if child should be in negative direction (LEFT, FRONT).
    """

    primary_axis: int  # 0=X, 1=Y
    direction: int  # +1 or -1

    @property
    def band_axis(self) -> int:
        """Perpendicular axis for band constraint (always 1 - primary_axis)."""
        return 1 - self.primary_axis


SIDE_CONFIGS: dict[Side, SideConfig] = {
    Side.RIGHT: SideConfig(primary_axis=0, direction=+1),
    Side.LEFT: SideConfig(primary_axis=0, direction=-1),
    Side.BACK: SideConfig(primary_axis=1, direction=+1),
    Side.FRONT: SideConfig(primary_axis=1, direction=-1),
}


class RelationLossStrategy(ABC):
    """Abstract base class defining how a Relation maps to a differentiable loss."""

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
        Supports all four sides: LEFT, RIGHT, FRONT, BACK.

        Args:
            relation: NextTo relation with side and distance attributes.
            child_pos: Child object position tensor (x, y, z) in world coords.
            parent_pos: Parent object position tensor (x, y, z) in world coords.
            child_bbox: Child object local bounding box.
            parent_bbox: Parent object local bounding box.

        Returns:
            Weighted loss tensor.
        """
        cfg = SIDE_CONFIGS[relation.side]
        distance = relation.distance_m
        assert distance >= 0.0, f"NextTo distance must be non-negative, got {distance}"

        # Select parent edge and child offset based on direction
        # direction +1 (RIGHT/BACK): child in positive direction, use parent max edge
        # direction -1 (LEFT/FRONT): child in negative direction, use parent min edge
        if cfg.direction == +1:
            parent_edge = parent_pos[cfg.primary_axis] + parent_bbox.max_point[cfg.primary_axis]
            child_offset = child_bbox.min_point[cfg.primary_axis]
            penalty_side = "less"
        else:  # direction == -1
            parent_edge = parent_pos[cfg.primary_axis] + parent_bbox.min_point[cfg.primary_axis]
            child_offset = child_bbox.max_point[cfg.primary_axis]
            penalty_side = "greater"

        # 1. Half-plane loss: child must be on correct side of parent edge
        half_plane_loss = single_boundary_linear_loss(
            child_pos[cfg.primary_axis],
            parent_edge,
            slope=self.slope,
            penalty_side=penalty_side,
        )

        # 2. Band loss: child's footprint must be within parent's extent on perpendicular axis
        # Compute valid position range such that child's entire footprint stays within parent
        parent_band_min = parent_pos[cfg.band_axis] + parent_bbox.min_point[cfg.band_axis]
        parent_band_max = parent_pos[cfg.band_axis] + parent_bbox.max_point[cfg.band_axis]
        valid_band_min = parent_band_min - child_bbox.min_point[cfg.band_axis]
        valid_band_max = parent_band_max - child_bbox.max_point[cfg.band_axis]
        band_loss = linear_band_loss(
            child_pos[cfg.band_axis],
            lower_bound=valid_band_min,
            upper_bound=valid_band_max,
            slope=self.slope,
        )

        # 3. Distance loss: child edge at target distance from parent edge
        # For direction +1: target = parent_max + distance - child_min
        # For direction -1: target = parent_min - distance - child_max
        target_pos = parent_edge + cfg.direction * distance - child_offset
        distance_loss = single_point_linear_loss(child_pos[cfg.primary_axis], target_pos, slope=self.slope)

        if self.debug:
            axis_name = "X" if cfg.primary_axis == 0 else "Y"
            band_axis_name = "Y" if cfg.primary_axis == 0 else "X"
            print(
                f"    [NextTo] {relation.side.value}: child_{axis_name.lower()}="
                f"{child_pos[cfg.primary_axis].item():.4f}, parent_edge={parent_edge.item():.4f},"
                f" loss={half_plane_loss.item():.6f}"
            )
            print(
                f"    [NextTo] {band_axis_name} band: child_{band_axis_name.lower()}="
                f"{child_pos[cfg.band_axis].item():.4f}, valid_range=[{valid_band_min.item():.4f},"
                f" {valid_band_max.item():.4f}], loss={band_loss.item():.6f}"
            )
            print(
                f"    [NextTo] Distance: child_{axis_name.lower()}="
                f"{child_pos[cfg.primary_axis].item():.4f}, target={target_pos.item():.4f},"
                f" loss={distance_loss.item():.6f}"
            )

        total_loss = half_plane_loss + band_loss + distance_loss
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

# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

from isaaclab_arena.relations.loss_primitives import (
    interval_overlap_axis_loss,
    linear_band_loss,
    single_boundary_linear_loss,
    single_point_linear_loss,
)
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

if TYPE_CHECKING:
    from isaaclab_arena.relations.relations import AtPosition, NextTo, On, Relation, NoCollision

from isaaclab_arena.relations.relations import Side


class Axis(IntEnum):
    """Spatial axis indices for tensor indexing."""

    X = 0
    Y = 1
    Z = 2


class Direction(IntEnum):
    """Direction along an axis."""

    NEGATIVE = -1
    POSITIVE = +1


@dataclass(frozen=True)
class SideConfig:
    """Configuration for computing NextTo loss for a given axis direction.

    Attributes:
        primary_axis: Axis along which child is placed (X or Y).
        direction: POSITIVE if child should be in positive direction from parent,
                   NEGATIVE if child should be in negative direction.
    """

    primary_axis: Axis
    direction: Direction

    @property
    def band_axis(self) -> Axis:
        """Perpendicular axis for band constraint."""
        return Axis(1 - self.primary_axis)


SIDE_CONFIGS: dict[Side, SideConfig] = {
    Side.POSITIVE_X: SideConfig(primary_axis=Axis.X, direction=Direction.POSITIVE),
    Side.NEGATIVE_X: SideConfig(primary_axis=Axis.X, direction=Direction.NEGATIVE),
    Side.POSITIVE_Y: SideConfig(primary_axis=Axis.Y, direction=Direction.POSITIVE),
    Side.NEGATIVE_Y: SideConfig(primary_axis=Axis.Y, direction=Direction.NEGATIVE),
}


class UnaryRelationLossStrategy(ABC):
    """Abstract base class for unary relations (no parent object)."""

    @abstractmethod
    def compute_loss(
        self,
        relation: "Relation",
        child_pos: torch.Tensor,
        child_bbox: AxisAlignedBoundingBox,
        parent_world_bbox: AxisAlignedBoundingBox | None = None,
        parent_world_bbox_batched: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Compute the loss for a unary relation constraint. Parent args ignored (for API consistency)."""
        pass


class RelationLossStrategy(ABC):
    """Abstract base class defining how a Relation maps to a differentiable loss."""

    @abstractmethod
    def compute_loss(
        self,
        relation: "Relation",
        child_pos: torch.Tensor,
        child_bbox: AxisAlignedBoundingBox,
        parent_world_bbox: AxisAlignedBoundingBox,
        parent_world_bbox_batched: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Compute the loss for a relation constraint.

        child_pos can be (3,) or (E, 3). When parent_world_bbox_batched is provided,
        (min (E,3), max (E,3)), use it for batched parent bounds instead of parent_world_bbox.

        Returns:
            Scalar or (E,) loss tensor.
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
        child_bbox: AxisAlignedBoundingBox,
        parent_world_bbox: AxisAlignedBoundingBox,
        parent_world_bbox_batched: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Compute loss for NextTo relation. child_pos can be (3,) or (E, 3); parent can be single or batched."""
        cfg = SIDE_CONFIGS[relation.side]
        distance = relation.distance_m
        assert distance >= 0.0, f"NextTo distance must be non-negative, got {distance}"

        if parent_world_bbox_batched is not None:
            parent_min, parent_max = parent_world_bbox_batched
            parent_edge = parent_max[..., cfg.primary_axis] if cfg.direction == Direction.POSITIVE else parent_min[..., cfg.primary_axis]
            parent_band_min = parent_min[..., cfg.band_axis]
            parent_band_max = parent_max[..., cfg.band_axis]
        else:
            parent_edge = parent_world_bbox.max_point[cfg.primary_axis] if cfg.direction == Direction.POSITIVE else parent_world_bbox.min_point[cfg.primary_axis]
            parent_band_min = parent_world_bbox.min_point[cfg.band_axis]
            parent_band_max = parent_world_bbox.max_point[cfg.band_axis]

        child_offset = child_bbox.min_point[cfg.primary_axis] if cfg.direction == Direction.POSITIVE else child_bbox.max_point[cfg.primary_axis]
        penalty_side = "less" if cfg.direction == Direction.POSITIVE else "greater"

        half_plane_loss = single_boundary_linear_loss(
            child_pos[..., cfg.primary_axis], parent_edge, slope=self.slope, penalty_side=penalty_side
        )
        valid_band_min = parent_band_min - child_bbox.min_point[cfg.band_axis]
        valid_band_max = parent_band_max - child_bbox.max_point[cfg.band_axis]
        t = (relation.cross_position_ratio + 1.0) / 2.0
        target_band_pos = valid_band_min + t * (valid_band_max - valid_band_min)
        band_loss = single_point_linear_loss(child_pos[..., cfg.band_axis], target_band_pos, slope=self.slope)
        target_pos = parent_edge + cfg.direction * distance - child_offset
        distance_loss = single_point_linear_loss(child_pos[..., cfg.primary_axis], target_pos, slope=self.slope)

        if self.debug and child_pos.dim() == 1:
            print(f"    [NextTo] {relation.side.value}: loss={half_plane_loss.item():.6f}, band={band_loss.item():.6f}, dist={distance_loss.item():.6f}")

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
        child_bbox: AxisAlignedBoundingBox,
        parent_world_bbox: AxisAlignedBoundingBox,
        parent_world_bbox_batched: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Compute loss for On relation. child_pos can be (3,) or (E, 3); parent can be single or batched."""
        inset = getattr(relation, "edge_inset_m", 0.0)
        if parent_world_bbox_batched is not None:
            parent_min, parent_max = parent_world_bbox_batched
            parent_x_min = parent_min[..., 0] + inset
            parent_x_max = parent_max[..., 0] - inset
            parent_y_min = parent_min[..., 1] + inset
            parent_y_max = parent_max[..., 1] - inset
            parent_z_max = parent_max[..., 2]
        else:
            parent_x_min = parent_world_bbox.min_point[0] + inset
            parent_x_max = parent_world_bbox.max_point[0] - inset
            parent_y_min = parent_world_bbox.min_point[1] + inset
            parent_y_max = parent_world_bbox.max_point[1] - inset
            parent_z_max = parent_world_bbox.max_point[2]

        valid_x_min = parent_x_min - child_bbox.min_point[0]
        valid_x_max = parent_x_max - child_bbox.max_point[0]
        valid_y_min = parent_y_min - child_bbox.min_point[1]
        valid_y_max = parent_y_max - child_bbox.max_point[1]

        x_band_loss = linear_band_loss(
            child_pos[..., 0], lower_bound=valid_x_min, upper_bound=valid_x_max, slope=self.slope
        )
        y_band_loss = linear_band_loss(
            child_pos[..., 1], lower_bound=valid_y_min, upper_bound=valid_y_max, slope=self.slope
        )
        target_z = parent_z_max + relation.clearance_m - child_bbox.min_point[2]
        z_loss = single_point_linear_loss(child_pos[..., 2], target_z, slope=self.slope)

        if self.debug and child_pos.dim() == 1:
            print(f"    [On] X: child_pos={child_pos[0].item():.4f}, valid=[{valid_x_min:.4f},{valid_x_max:.4f}], loss={x_band_loss.item():.6f}")
            print(f"    [On] Y: child_pos={child_pos[1].item():.4f}, valid=[{valid_y_min:.4f},{valid_y_max:.4f}], loss={y_band_loss.item():.6f}")
            print(f"    [On] Z: child_pos={child_pos[2].item():.4f}, target={target_z:.4f}, loss={z_loss.item():.6f}")

        total_loss = x_band_loss + y_band_loss + z_loss
        return relation.relation_loss_weight * total_loss


class NoCollisionLossStrategy(RelationLossStrategy):
    """Loss strategy for NoCollision relations.

    Computes loss based on:
    1. X overlap: zero when child and parent are separated along X; else overlap length
    2. Y overlap: zero when separated along Y; else overlap length
    3. Z overlap: zero when separated along Z; else overlap length
    4. Volume loss: slope * (overlap_x * overlap_y * overlap_z)
    """

    def __init__(self, slope: float = 10.0, debug: bool = False):
        """
        Args:
            slope: Gradient magnitude for overlap volume loss (default: 10.0).
                   Loss scales with slope times overlap volume.
            debug: If True, print detailed loss component breakdown.
        """
        self.slope = slope
        self.debug = debug

    def compute_loss(
        self,
        relation: "NoCollision",
        child_pos: torch.Tensor,
        child_bbox: AxisAlignedBoundingBox,
        parent_world_bbox: AxisAlignedBoundingBox,
        parent_world_bbox_batched: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Compute loss for NoCollision relation. child_pos can be (3,) or (E, 3); parent can be single or batched."""
        c = relation.clearance_m
        if parent_world_bbox_batched is not None:
            parent_min, parent_max = parent_world_bbox_batched
            parent_x_min = parent_min[..., 0] - c
            parent_x_max = parent_max[..., 0] + c
            parent_y_min = parent_min[..., 1] - c
            parent_y_max = parent_max[..., 1] + c
            parent_z_min = parent_min[..., 2] - c
            parent_z_max = parent_max[..., 2] + c
        else:
            parent_x_min = parent_world_bbox.min_point[0] - c
            parent_x_max = parent_world_bbox.max_point[0] + c
            parent_y_min = parent_world_bbox.min_point[1] - c
            parent_y_max = parent_world_bbox.max_point[1] + c
            parent_z_min = parent_world_bbox.min_point[2] - c
            parent_z_max = parent_world_bbox.max_point[2] + c

        child_world_min = child_pos + torch.tensor(child_bbox.min_point, dtype=child_pos.dtype, device=child_pos.device)
        child_world_max = child_pos + torch.tensor(child_bbox.max_point, dtype=child_pos.dtype, device=child_pos.device)

        overlap_x = interval_overlap_axis_loss(child_world_min[..., 0], child_world_max[..., 0], parent_x_min, parent_x_max)
        overlap_y = interval_overlap_axis_loss(child_world_min[..., 1], child_world_max[..., 1], parent_y_min, parent_y_max)
        overlap_z = interval_overlap_axis_loss(child_world_min[..., 2], child_world_max[..., 2], parent_z_min, parent_z_max)
        overlap_volume = overlap_x * overlap_y * overlap_z
        total_loss = self.slope * overlap_volume

        if self.debug and child_pos.dim() == 1:
            print(f"    [NoCollision] volume={overlap_volume.item():.6f}, loss={total_loss.item():.6f}")

        return relation.relation_loss_weight * total_loss


class AtPositionLossStrategy(UnaryRelationLossStrategy):
    """Loss strategy for AtPosition relations.

    Computes loss based on single-point linear losses for each specified axis.
    Axes set to None in the relation are ignored.
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
        relation: "AtPosition",
        child_pos: torch.Tensor,
        child_bbox: AxisAlignedBoundingBox,
        parent_world_bbox: AxisAlignedBoundingBox | None = None,
        parent_world_bbox_batched: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Compute loss for AtPosition relation. child_pos can be (3,) or (E, 3). Parent args ignored."""
        total_loss = torch.tensor(0.0, dtype=child_pos.dtype, device=child_pos.device)

        # Support batched child_pos (E, 3) via ... indexing
        if relation.x is not None:
            x_loss = single_point_linear_loss(child_pos[..., 0], relation.x, slope=self.slope)
            total_loss = total_loss + x_loss
        if relation.y is not None:
            y_loss = single_point_linear_loss(child_pos[..., 1], relation.y, slope=self.slope)
            total_loss = total_loss + y_loss
        if relation.z is not None:
            z_loss = single_point_linear_loss(child_pos[..., 2], relation.z, slope=self.slope)
            total_loss = total_loss + z_loss

        return relation.relation_loss_weight * total_loss

# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
from isaaclab_arena.relations.relations import Side
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

if TYPE_CHECKING:
    from isaaclab_arena.relations.relations import (
        AtPosition,
        NextTo,
        NotNextTo,
        On,
        PositionLimits,
        Relation,
        UnaryRelation,
    )


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


def next_to_violations(
    cfg: SideConfig,
    child_pos: torch.Tensor,
    child_bbox: AxisAlignedBoundingBox,
    parent_world_bbox: AxisAlignedBoundingBox,
    distance_m: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Side and distance violation magnitudes (meters, >= 0) for a NextTo relation.

    Shared by NextToLossStrategy (scaled by slope into the loss) and the placement validator
    (thresholded against tolerance), so loss and validation can't disagree on the geometry.

    Args:
        cfg: Side configuration (primary/band axis and direction) for the relation's side.
        child_pos: Child position, shape (N, 3), in world coords.
        child_bbox: Child local bounding box (N=1).
        parent_world_bbox: Parent bounding box in world coords.
        distance_m: Target distance from the parent edge.

    Returns:
        (half_plane, distance) tensors of shape (N,), each >= 0 and zero when satisfied.
    """
    if cfg.direction == Direction.POSITIVE:
        parent_edge = parent_world_bbox.max_point[:, cfg.primary_axis]
        child_offset = child_bbox.min_point[:, cfg.primary_axis]
        penalty_side = "less"
    else:
        parent_edge = parent_world_bbox.min_point[:, cfg.primary_axis]
        child_offset = child_bbox.max_point[:, cfg.primary_axis]
        penalty_side = "greater"

    primary = child_pos[:, cfg.primary_axis]
    half_plane = single_boundary_linear_loss(primary, parent_edge, slope=1.0, penalty_side=penalty_side)
    target_pos = parent_edge + cfg.direction * distance_m - child_offset
    distance = single_point_linear_loss(primary, target_pos, slope=1.0)
    return half_plane, distance


def not_next_to_violations(
    cfg: SideConfig,
    child_pos: torch.Tensor,
    child_bbox: AxisAlignedBoundingBox,
    parent_world_bbox: AxisAlignedBoundingBox,
    margin_m: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-route escape distances (meters, >= 0) for a NotNextTo relation.

    Shared by NotNextToLossStrategy and the placement validator. The child clears the keep-out zone
    once either route reaches zero: ``remaining_side`` (cross back over the edge) or
    ``remaining_cross`` (slide past either footprint end), both by ``margin_m``.

    Args:
        cfg: Side configuration for the relation's side.
        child_pos: Child position, shape (N, 3), in world coords.
        child_bbox: Child local bounding box (N=1).
        parent_world_bbox: Parent bounding box in world coords.
        margin_m: Distance past the edge/footprint required to clear the zone.

    Returns:
        (remaining_side, remaining_cross) tensors of shape (N,), each >= 0.
    """
    if cfg.direction == Direction.POSITIVE:
        parent_edge = parent_world_bbox.max_point[:, cfg.primary_axis]
        blocked_side_penalty = "greater"
    else:
        parent_edge = parent_world_bbox.min_point[:, cfg.primary_axis]
        blocked_side_penalty = "less"

    parent_band_min = parent_world_bbox.min_point[:, cfg.band_axis]
    parent_band_max = parent_world_bbox.max_point[:, cfg.band_axis]
    valid_band_min = parent_band_min - child_bbox.min_point[:, cfg.band_axis]
    valid_band_max = parent_band_max - child_bbox.max_point[:, cfg.band_axis]

    primary = child_pos[:, cfg.primary_axis]
    cross = child_pos[:, cfg.band_axis]

    safe_edge = parent_edge - cfg.direction * margin_m
    remaining_side = single_boundary_linear_loss(primary, safe_edge, slope=1.0, penalty_side=blocked_side_penalty)
    safe_band_min = valid_band_min - margin_m
    safe_band_max = valid_band_max + margin_m
    remaining_cross = torch.minimum(
        single_boundary_linear_loss(cross, safe_band_min, slope=1.0, penalty_side="greater"),
        single_boundary_linear_loss(cross, safe_band_max, slope=1.0, penalty_side="less"),
    )
    return remaining_side, remaining_cross


class UnaryRelationLossStrategy(ABC):
    """Abstract base class for unary relations (no parent object)."""

    @abstractmethod
    def compute_loss(
        self,
        relation: UnaryRelation,
        child_pos: torch.Tensor,
        child_bbox: AxisAlignedBoundingBox,
    ) -> torch.Tensor:
        """Compute the loss for a unary relation constraint.

        Args:
            relation: The relation object containing constraint metadata.
            child_pos: Child object position tensor. Accepts (3,) for single-env
                backward compat or (N, 3) for batched.
            child_bbox: Child object local bounding box (N=1).

        Returns:
            Scalar loss tensor when child_pos is (3,), or (N,) tensor when (N, 3).
        """
        pass


class RelationLossStrategy(ABC):
    """Abstract base class defining how a Relation maps to a differentiable loss."""

    @abstractmethod
    def compute_loss(
        self,
        relation: Relation,
        child_pos: torch.Tensor,
        child_bbox: AxisAlignedBoundingBox,
        parent_world_bbox: AxisAlignedBoundingBox,
    ) -> torch.Tensor:
        """Compute the loss for a relation constraint.

        Args:
            relation: The relation object containing relationship metadata.
            child_pos: Child object position tensor. Accepts (3,) for single-env
                backward compat or (N, 3) for batched.
            child_bbox: Child object local bounding box (N=1).
            parent_world_bbox: Parent bounding box in world coordinates.

        Returns:
            Scalar loss tensor when child_pos is (3,), or (N,) tensor when (N, 3).
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
        relation: NextTo,
        child_pos: torch.Tensor,
        child_bbox: AxisAlignedBoundingBox,
        parent_world_bbox: AxisAlignedBoundingBox,
    ) -> torch.Tensor:
        """Compute loss for NextTo relation.

        Supports all four sides: LEFT, RIGHT, FRONT, BACK.

        Args:
            relation: NextTo relation with side and distance attributes.
            child_pos: Child object position (N, 3) in world coords.
            child_bbox: Child object local bounding box (N=1).
            parent_world_bbox: Parent bounding box in world coordinates.

        Returns:
            Weighted loss tensor of shape (N,).
        """
        single_input = child_pos.dim() == 1
        if single_input:
            child_pos = child_pos.unsqueeze(0)

        cfg = SIDE_CONFIGS[relation.side]
        distance = relation.distance_m
        assert distance >= 0.0, f"NextTo distance must be non-negative, got {distance}"

        # 1. & 3. Side (half-plane) and distance share their geometry with the placement validator.
        half_plane_raw, distance_raw = next_to_violations(cfg, child_pos, child_bbox, parent_world_bbox, distance)
        half_plane_loss = self.slope * half_plane_raw
        distance_loss = self.slope * distance_raw

        # 2. Band position loss: child placed at target position within parent's perpendicular extent
        parent_band_min = parent_world_bbox.min_point[:, cfg.band_axis]
        parent_band_max = parent_world_bbox.max_point[:, cfg.band_axis]
        valid_band_min = parent_band_min - child_bbox.min_point[:, cfg.band_axis]
        valid_band_max = parent_band_max - child_bbox.max_point[:, cfg.band_axis]
        # Convert cross_position_ratio [-1, 1] to interpolation factor [0, 1]: -1 = min, 0 = center, 1 = max
        t = (relation.cross_position_ratio + 1.0) / 2.0
        target_band_pos = valid_band_min + t * (valid_band_max - valid_band_min)
        band_loss = single_point_linear_loss(child_pos[:, cfg.band_axis], target_band_pos, slope=self.slope)

        if self.debug and child_pos.shape[0] == 1:
            band_axis_name = cfg.band_axis.name
            print(
                f"    [NextTo] {relation.side.value}: half_plane={half_plane_raw[0].item():.6f},"
                f" distance={distance_raw[0].item():.6f} (m)"
            )
            print(
                f"    [NextTo] {band_axis_name} band: child_{band_axis_name.lower()}="
                f"{child_pos[0, cfg.band_axis].item():.4f}, target={target_band_pos[0].item():.4f}"
                f" (cross_position_ratio={relation.cross_position_ratio:.2f},"
                f" range=[{valid_band_min[0].item():.4f}, {valid_band_max[0].item():.4f}]),"
                f" loss={band_loss[0].item():.6f}"
            )

        total_loss = half_plane_loss + band_loss + distance_loss
        result = relation.relation_loss_weight * total_loss
        return result.squeeze(0) if single_input else result


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
        relation: On,
        child_pos: torch.Tensor,
        child_bbox: AxisAlignedBoundingBox,
        parent_world_bbox: AxisAlignedBoundingBox,
    ) -> torch.Tensor:
        """Compute loss for On relation.

        Args:
            relation: On relation with clearance_m attribute.
            child_pos: Child object position (N, 3) in world coords.
            child_bbox: Child object local bounding box (N=1).
            parent_world_bbox: Parent bounding box in world coordinates.

        Returns:
            Weighted loss tensor of shape (N,).
        """
        single_input = child_pos.dim() == 1
        if single_input:
            child_pos = child_pos.unsqueeze(0)

        # Parent world-space extents from the world bounding box
        parent_x_min = parent_world_bbox.min_point[:, 0]
        parent_x_max = parent_world_bbox.max_point[:, 0]
        parent_y_min = parent_world_bbox.min_point[:, 1]
        parent_y_max = parent_world_bbox.max_point[:, 1]
        parent_z_max = parent_world_bbox.max_point[:, 2]  # Top surface

        # Compute valid position ranges such that child's entire footprint is within parent,
        # with the parent's extent inset by edge_margin_m so the footprint stays off the rim.
        m = relation.edge_margin_m
        valid_x_min = parent_x_min + m - child_bbox.min_point[:, 0]  # child's left at parent's left + margin
        valid_x_max = parent_x_max - m - child_bbox.max_point[:, 0]  # child's right at parent's right - margin
        valid_y_min = parent_y_min + m - child_bbox.min_point[:, 1]
        valid_y_max = parent_y_max - m - child_bbox.max_point[:, 1]

        # The bounds invert (lower > upper) when the margin is too large for the surface or the
        # child is oversized. The loss becomes a non-zero constant with gradient zero.

        # 1. X band loss: child's footprint entirely within parent's X extent
        x_band_loss = linear_band_loss(
            child_pos[:, 0],
            lower_bound=valid_x_min,
            upper_bound=valid_x_max,
            slope=self.slope,
        )

        # 2. Y band loss: child's footprint entirely within parent's Y extent
        y_band_loss = linear_band_loss(
            child_pos[:, 1],
            lower_bound=valid_y_min,
            upper_bound=valid_y_max,
            slope=self.slope,
        )

        # 3. Z point loss: child bottom = parent top + clearance
        target_z = parent_z_max + relation.clearance_m - child_bbox.min_point[:, 2]
        z_loss = single_point_linear_loss(child_pos[:, 2], target_z, slope=self.slope)

        if self.debug and child_pos.shape[0] == 1:
            print(
                f"    [On] X: child_pos={child_pos[0, 0].item():.4f}, valid_range=[{valid_x_min[0].item():.4f},"
                f" {valid_x_max[0].item():.4f}], loss={x_band_loss[0].item():.6f}"
            )
            print(
                f"    [On] Y: child_pos={child_pos[0, 1].item():.4f}, valid_range=[{valid_y_min[0].item():.4f},"
                f" {valid_y_max[0].item():.4f}], loss={y_band_loss[0].item():.6f}"
            )
            print(
                f"    [On] Z: child_pos={child_pos[0, 2].item():.4f}, target={target_z[0].item():.4f},"
                f" loss={z_loss[0].item():.6f}"
            )

        total_loss = x_band_loss + y_band_loss + z_loss
        result = relation.relation_loss_weight * total_loss
        return result.squeeze(0) if single_input else result


class NotNextToLossStrategy(RelationLossStrategy):
    """Loss strategy for ``NotNextTo`` — keep the child out of the half-plane beside the parent.

    Blocked region: everything past the parent's edge on the chosen side and
    within the parent's perpendicular footprint (for ``+Y``: all of ``+Y`` past
    the ``+Y`` edge, clipped to the parent's ``X`` extent). The child escapes by
    one of two routes — cross back over the edge, or step out past either end of
    the footprint::

        loss = slope * min(remaining_side, remaining_cross)

    Each ``remaining`` is the distance the child must still travel to clear that
    route by ``margin_m`` (0 once cleared), so the loss is non-zero everywhere in
    the zone — no flat plateau — and its gradient points to the nearest exit.
    """

    def __init__(self, slope: float = 10.0, margin_m: float = 0.1, debug: bool = False):
        """
        Args:
            slope: Loss magnitude per meter of remaining escape distance.
            margin_m: How far past the edge or footprint the child must reach for
                zero loss (default 10 cm).
            debug: If True, print the per-route remaining distances.
        """
        assert slope >= 0.0, f"slope must be non-negative, got {slope}"
        assert margin_m > 0.0, f"margin_m must be positive, got {margin_m}"
        self.slope = slope
        self.margin_m = margin_m
        self.debug = debug

    def compute_loss(
        self,
        relation: NotNextTo,
        child_pos: torch.Tensor,
        child_bbox: AxisAlignedBoundingBox,
        parent_world_bbox: AxisAlignedBoundingBox,
    ) -> torch.Tensor:
        """Compute loss for ``NotNextTo``."""
        single_input = child_pos.dim() == 1
        if single_input:
            child_pos = child_pos.unsqueeze(0)

        cfg = SIDE_CONFIGS[relation.side]

        # Per-route escape distances share their geometry with the placement validator.
        # Route 1 crosses back over the edge; route 2 slides past either footprint end. Each
        # `remaining` is the distance still to travel to clear that route by margin_m (0 once cleared).
        remaining_side, remaining_cross = not_next_to_violations(
            cfg, child_pos, child_bbox, parent_world_bbox, self.margin_m
        )

        # Clearing either route is enough.
        loss = self.slope * torch.minimum(remaining_side, remaining_cross)

        if self.debug and child_pos.shape[0] == 1:
            print(
                f"    [NotNextTo] {relation.side.value}: "
                f"remaining_side={remaining_side[0].item():.4f} "
                f"remaining_cross={remaining_cross[0].item():.4f} "
                f"-> loss={loss[0].item():.6f}"
            )

        result = relation.relation_loss_weight * loss
        return result.squeeze(0) if single_input else result


class NoCollisionLossStrategy:
    """AABB no-overlap loss between object pairs."""

    def __init__(
        self,
        slope: float = 10.0,
    ):
        """
        Args:
            slope: Gradient magnitude for overlap loss.
        """
        self.slope = slope

    def compute_loss_batched(
        self,
        clearance_m: float,
        subject_min: torch.Tensor,
        subject_max: torch.Tensor,
        obstacle_min: torch.Tensor,
        obstacle_max: torch.Tensor,
    ) -> torch.Tensor:
        """Overlap-volume no-overlap loss for boxes already reduced to world-space extents.

        Args:
            clearance_m: Minimum clearance between boxes in meters.
            subject_min: World-space min extent of the subject box, shape (num_pairs, batch_size, 3).
            subject_max: World-space max extent of the subject box, shape (num_pairs, batch_size, 3).
            obstacle_min: World-space min extent of the obstacle box, shape (num_pairs, batch_size, 3).
            obstacle_max: World-space max extent of the obstacle box, shape (num_pairs, batch_size, 3).

        Returns:
            Per-pair, per-env loss of shape (num_pairs, batch_size).
        """
        assert clearance_m >= 0, f"clearance_m must be non-negative, got {clearance_m}"
        obstacle_min = obstacle_min - clearance_m
        obstacle_max = obstacle_max + clearance_m
        overlap_x = interval_overlap_axis_loss(
            subject_min[..., 0], subject_max[..., 0], obstacle_min[..., 0], obstacle_max[..., 0]
        )
        overlap_y = interval_overlap_axis_loss(
            subject_min[..., 1], subject_max[..., 1], obstacle_min[..., 1], obstacle_max[..., 1]
        )
        overlap_z = interval_overlap_axis_loss(
            subject_min[..., 2], subject_max[..., 2], obstacle_min[..., 2], obstacle_max[..., 2]
        )
        return self.slope * (overlap_x * overlap_y * overlap_z)


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
        relation: AtPosition,
        child_pos: torch.Tensor,
        child_bbox: AxisAlignedBoundingBox,
    ) -> torch.Tensor:
        """Compute loss for AtPosition relation.

        Args:
            relation: AtPosition relation with x, y, z target coordinates.
            child_pos: Child object position (N, 3) in world coords.
            child_bbox: Child object local bounding box (unused, for signature consistency).

        Returns:
            Weighted loss tensor of shape (N,).
        """
        single_input = child_pos.dim() == 1
        if single_input:
            child_pos = child_pos.unsqueeze(0)

        total_loss = torch.zeros(child_pos.shape[0], dtype=child_pos.dtype, device=child_pos.device)

        if relation.x is not None:
            x_loss = single_point_linear_loss(child_pos[:, 0], relation.x, slope=self.slope)
            total_loss = total_loss + x_loss

        if relation.y is not None:
            y_loss = single_point_linear_loss(child_pos[:, 1], relation.y, slope=self.slope)
            total_loss = total_loss + y_loss

        if relation.z is not None:
            z_loss = single_point_linear_loss(child_pos[:, 2], relation.z, slope=self.slope)
            total_loss = total_loss + z_loss

        result = relation.relation_loss_weight * total_loss
        return result.squeeze(0) if single_input else result


class PositionLimitsLossStrategy(UnaryRelationLossStrategy):
    """Loss strategy for PositionLimits relations.

    Per constrained axis: band loss when both bounds are set, single-boundary
    loss when only one bound is set. Unconstrained axes contribute zero loss.
    """

    def __init__(self, slope: float = 100.0):
        """
        Args:
            slope: Gradient magnitude for linear loss (default: 100.0).
                   Loss increases by ``slope`` per meter of violation.
        """
        self.slope = slope

    def compute_loss(
        self,
        relation: PositionLimits,
        child_pos: torch.Tensor,
        child_bbox: AxisAlignedBoundingBox,
    ) -> torch.Tensor:
        """Compute loss for PositionLimits relation.

        Args:
            relation: PositionLimits relation with optional per-axis bounds.
            child_pos: Child object position (N, 3) in world coords.
            child_bbox: Object local bounding box (unused, for signature consistency).

        Returns:
            Weighted loss tensor of shape (N,).
        """
        single_input = child_pos.dim() == 1
        if single_input:
            child_pos = child_pos.unsqueeze(0)

        total_loss = torch.zeros(child_pos.shape[0], dtype=child_pos.dtype, device=child_pos.device)

        axis_bounds = [
            (relation.x_min, relation.x_max),
            (relation.y_min, relation.y_max),
            (relation.z_min, relation.z_max),
        ]
        for axis_index, (lower_bound, upper_bound) in enumerate(axis_bounds):
            if lower_bound is not None and upper_bound is not None:
                # Both bounds: zero inside [lower, upper], linear growth outside
                total_loss = total_loss + linear_band_loss(
                    child_pos[:, axis_index], lower_bound, upper_bound, slope=self.slope
                )
            elif lower_bound is not None:
                # Only lower bound: penalize positions below it
                total_loss = total_loss + single_boundary_linear_loss(
                    child_pos[:, axis_index], lower_bound, slope=self.slope, penalty_side="less"
                )
            elif upper_bound is not None:
                # Only upper bound: penalize positions above it
                total_loss = total_loss + single_boundary_linear_loss(
                    child_pos[:, axis_index], upper_bound, slope=self.slope, penalty_side="greater"
                )
            # Neither bound set: axis is unconstrained, no loss

        result = relation.relation_loss_weight * total_loss
        return result.squeeze(0) if single_input else result

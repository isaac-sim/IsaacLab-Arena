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
    from isaaclab_arena.assets.object_base import ObjectBase
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
        relation: "Relation",
        child_pos: torch.Tensor,
        child_bbox: AxisAlignedBoundingBox,
        parent_world_bbox: AxisAlignedBoundingBox,
        child_obj: "ObjectBase | None" = None,
    ) -> torch.Tensor:
        """Compute the loss for a relation constraint.

        Args:
            relation: The relation object containing relationship metadata.
            child_pos: Child object position tensor. Accepts (3,) for single-env
                backward compat or (N, 3) for batched.
            child_bbox: Child object local bounding box (N=1).
            parent_world_bbox: Parent bounding box in world coordinates.
            child_obj: The child ObjectBase instance (optional, used by mesh-based
                strategies that need to look up collision meshes).

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
        relation: "NextTo",
        child_pos: torch.Tensor,
        child_bbox: AxisAlignedBoundingBox,
        parent_world_bbox: AxisAlignedBoundingBox,
        child_obj: "ObjectBase | None" = None,
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

        # Parent world extents from the world bounding box
        if cfg.direction == Direction.POSITIVE:
            parent_edge = parent_world_bbox.max_point[:, cfg.primary_axis]
            child_offset = child_bbox.min_point[:, cfg.primary_axis]
            penalty_side = "less"
        else:
            parent_edge = parent_world_bbox.min_point[:, cfg.primary_axis]
            child_offset = child_bbox.max_point[:, cfg.primary_axis]
            penalty_side = "greater"

        # 1. Half-plane loss: child must be on correct side of parent edge
        half_plane_loss = single_boundary_linear_loss(
            child_pos[:, cfg.primary_axis],
            parent_edge,
            slope=self.slope,
            penalty_side=penalty_side,
        )

        # 2. Band position loss: child placed at target position within parent's perpendicular extent
        parent_band_min = parent_world_bbox.min_point[:, cfg.band_axis]
        parent_band_max = parent_world_bbox.max_point[:, cfg.band_axis]
        valid_band_min = parent_band_min - child_bbox.min_point[:, cfg.band_axis]
        valid_band_max = parent_band_max - child_bbox.max_point[:, cfg.band_axis]
        # Convert cross_position_ratio [-1, 1] to interpolation factor [0, 1]: -1 = min, 0 = center, 1 = max
        t = (relation.cross_position_ratio + 1.0) / 2.0
        target_band_pos = valid_band_min + t * (valid_band_max - valid_band_min)
        band_loss = single_point_linear_loss(
            child_pos[:, cfg.band_axis],
            target_band_pos,
            slope=self.slope,
        )

        # 3. Distance loss: child edge at target distance from parent edge
        # For direction +1: target = parent_max + distance - child_min
        # For direction -1: target = parent_min - distance - child_max
        target_pos = parent_edge + cfg.direction * distance - child_offset
        distance_loss = single_point_linear_loss(child_pos[:, cfg.primary_axis], target_pos, slope=self.slope)

        if self.debug and child_pos.shape[0] == 1:
            axis_name = cfg.primary_axis.name
            band_axis_name = cfg.band_axis.name
            print(
                f"    [NextTo] {relation.side.value}: child_{axis_name.lower()}="
                f"{child_pos[0, cfg.primary_axis].item():.4f}, parent_edge={parent_edge[0].item():.4f},"
                f" loss={half_plane_loss[0].item():.6f}"
            )
            print(
                f"    [NextTo] {band_axis_name} band: child_{band_axis_name.lower()}="
                f"{child_pos[0, cfg.band_axis].item():.4f}, target={target_band_pos[0].item():.4f}"
                f" (cross_position_ratio={relation.cross_position_ratio:.2f},"
                f" range=[{valid_band_min[0].item():.4f}, {valid_band_max[0].item():.4f}]),"
                f" loss={band_loss[0].item():.6f}"
            )
            print(
                f"    [NextTo] Distance: child_{axis_name.lower()}="
                f"{child_pos[0, cfg.primary_axis].item():.4f}, target={target_pos[0].item():.4f},"
                f" loss={distance_loss[0].item():.6f}"
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
        relation: "On",
        child_pos: torch.Tensor,
        child_bbox: AxisAlignedBoundingBox,
        parent_world_bbox: AxisAlignedBoundingBox,
        child_obj: "ObjectBase | None" = None,
    ) -> torch.Tensor:
        """Compute loss for On relation.

        Args:
            relation: On relation with clearance_m attribute.
            child_pos: Child object position (N, 3) in world coords.
            child_bbox: Child object local bounding box (N=1).
            parent_world_bbox: Parent bounding box in world coordinates.
            child_obj: Unused; accepted for interface compatibility.

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

        # Compute valid position ranges such that child's entire footprint is within parent
        valid_x_min = parent_x_min - child_bbox.min_point[:, 0]  # child's left at parent's left
        valid_x_max = parent_x_max - child_bbox.max_point[:, 0]  # child's right at parent's right
        valid_y_min = parent_y_min - child_bbox.min_point[:, 1]
        valid_y_max = parent_y_max - child_bbox.max_point[:, 1]

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


class NoCollisionLossStrategy(RelationLossStrategy):
    """Loss strategy for NoCollision relations.

    Computes loss based on:
    1. X overlap: zero when child and parent are separated along X; else overlap length
    2. Y overlap: zero when separated along Y; else overlap length
    3. Z overlap: zero when separated along Z; else overlap length
    4. Volume loss: slope * (overlap_x * overlap_y * overlap_z)
    """

    def __init__(self, slope: float = 10.0, debug: bool = False, bbox_scale: float = 1.0):
        """
        Args:
            slope: Gradient magnitude for overlap volume loss (default: 10.0).
                   Loss scales with slope times overlap volume.
            debug: If True, print detailed loss component breakdown.
            bbox_scale: Scale factor for bounding boxes used in the loss (default: 1.0).
                Values < 1.0 shrink both child and parent AABBs around their centers,
                letting the optimizer pack objects tighter.  Useful when a downstream
                mesh-level validation will catch real collisions.
        """
        self.slope = slope
        self.debug = debug
        self.bbox_scale = bbox_scale

    @staticmethod
    def _shrink_bbox(bbox: AxisAlignedBoundingBox, scale: float) -> AxisAlignedBoundingBox:
        center = (bbox.min_point + bbox.max_point) * 0.5
        half = (bbox.max_point - bbox.min_point) * 0.5 * scale
        return AxisAlignedBoundingBox(center - half, center + half)

    def compute_loss(
        self,
        relation: "NoCollision",
        child_pos: torch.Tensor,
        child_bbox: AxisAlignedBoundingBox,
        parent_world_bbox: AxisAlignedBoundingBox,
        child_obj: "ObjectBase | None" = None,
    ) -> torch.Tensor:
        """Compute loss for NoCollision relation.

        Args:
            relation: NoCollision relation with relation_loss_weight.
            child_pos: Child object position (N, 3) in world coords.
            child_bbox: Child object local bounding box (N=1).
            parent_world_bbox: Parent bounding box in world coordinates.
            child_obj: Unused; accepted for interface compatibility.

        Returns:
            Weighted loss tensor of shape (N,).
        """
        single_input = child_pos.dim() == 1
        if single_input:
            child_pos = child_pos.unsqueeze(0)

        if self.bbox_scale != 1.0:
            child_bbox = self._shrink_bbox(child_bbox, self.bbox_scale)
            parent_world_bbox = self._shrink_bbox(parent_world_bbox, self.bbox_scale)

        # Parent world extents from the world bounding box, expanded by clearance_m
        c = relation.clearance_m
        parent_x_min = parent_world_bbox.min_point[:, 0] - c
        parent_x_max = parent_world_bbox.max_point[:, 0] + c
        parent_y_min = parent_world_bbox.min_point[:, 1] - c
        parent_y_max = parent_world_bbox.max_point[:, 1] + c
        parent_z_min = parent_world_bbox.min_point[:, 2] - c
        parent_z_max = parent_world_bbox.max_point[:, 2] + c

        # Child world extents
        child_world_min = child_pos + child_bbox.min_point
        child_world_max = child_pos + child_bbox.max_point

        # 1. Per-axis overlap: zero when separated; else overlap length (default slope 1.0 gives length in m)
        overlap_x = interval_overlap_axis_loss(child_world_min[:, 0], child_world_max[:, 0], parent_x_min, parent_x_max)
        overlap_y = interval_overlap_axis_loss(child_world_min[:, 1], child_world_max[:, 1], parent_y_min, parent_y_max)
        overlap_z = interval_overlap_axis_loss(child_world_min[:, 2], child_world_max[:, 2], parent_z_min, parent_z_max)

        # 2. Volume loss: slope * product of per-axis overlap lengths (overlap volume when slope 1.0)
        overlap_volume = overlap_x * overlap_y * overlap_z
        total_loss = self.slope * overlap_volume

        if self.debug and child_pos.shape[0] == 1:
            print(
                f"    [NoCollision] X: overlap={overlap_x[0].item():.6f} (child_x=[{child_world_min[0, 0].item():.4f},"
                f" {child_world_max[0, 0].item():.4f}], parent_x=[{parent_x_min[0].item():.4f},"
                f" {parent_x_max[0].item():.4f}])"
            )
            print(
                f"    [NoCollision] Y: overlap={overlap_y[0].item():.6f} (child_y=[{child_world_min[0, 1].item():.4f},"
                f" {child_world_max[0, 1].item():.4f}], parent_y=[{parent_y_min[0].item():.4f},"
                f" {parent_y_max[0].item():.4f}])"
            )
            print(
                f"    [NoCollision] Z: overlap={overlap_z[0].item():.6f} (child_z=[{child_world_min[0, 2].item():.4f},"
                f" {child_world_max[0, 2].item():.4f}], parent_z=[{parent_z_min[0].item():.4f},"
                f" {parent_z_max[0].item():.4f}])"
            )
            print(f"    [NoCollision] volume={overlap_volume[0].item():.6f}, loss={total_loss[0].item():.6f}")

        result = relation.relation_loss_weight * total_loss
        return result.squeeze(0) if single_input else result


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

        # X position constraint
        if relation.x is not None:
            x_loss = single_point_linear_loss(child_pos[:, 0], relation.x, slope=self.slope)
            total_loss = total_loss + x_loss

        # Y position constraint
        if relation.y is not None:
            y_loss = single_point_linear_loss(child_pos[:, 1], relation.y, slope=self.slope)
            total_loss = total_loss + y_loss

        # Z position constraint
        if relation.z is not None:
            z_loss = single_point_linear_loss(child_pos[:, 2], relation.z, slope=self.slope)
            total_loss = total_loss + z_loss

        result = relation.relation_loss_weight * total_loss
        return result.squeeze(0) if single_input else result


class MeshNoCollisionLossStrategy(RelationLossStrategy):
    """Differentiable mesh-based collision loss using NVIDIA Warp SDF queries.

    For each ``NoCollision`` pair, samples one object's mesh vertices,
    transforms them to the other object's local frame, and queries signed
    distances via ``wp.mesh_query_point``.  Penetrating vertices (negative
    SDF) contribute ``slope * ReLU(-sdf)`` to the loss.  The query is
    bidirectional (A's verts against B's mesh, and B's verts against A's
    mesh) so that partial overlaps in either direction are penalised.

    Falls back to AABB overlap loss for objects that lack a collision mesh.
    """

    def __init__(self, slope: float = 100.0, device: str = "cuda:0"):
        from isaaclab_arena.relations.warp_mesh_manager import WarpMeshManager

        self.slope = slope
        self._mesh_mgr = WarpMeshManager(device=device)
        self._fallback = NoCollisionLossStrategy(slope=slope)

    def compute_loss(
        self,
        relation: "NoCollision",
        child_pos: torch.Tensor,
        child_bbox: AxisAlignedBoundingBox,
        parent_world_bbox: AxisAlignedBoundingBox,
        child_obj: "ObjectBase | None" = None,
    ) -> torch.Tensor:
        """Compute mesh-based penetration loss for a NoCollision relation.

        If *child_obj* is ``None`` or either object lacks a collision mesh,
        delegates to the AABB fallback strategy.
        """
        parent_obj = relation.parent
        if child_obj is None or child_obj.get_collision_mesh() is None or parent_obj.get_collision_mesh() is None:
            return self._fallback.compute_loss(relation, child_pos, child_bbox, parent_world_bbox)

        from isaaclab_arena.relations.warp_sdf_kernels import mesh_sdf

        single_input = child_pos.dim() == 1
        if single_input:
            child_pos = child_pos.unsqueeze(0)

        device_str = self._mesh_mgr.device
        device = child_pos.device
        batch = child_pos.shape[0]

        parent_local_bbox = parent_obj.get_bounding_box().to(device)
        parent_pos = parent_world_bbox.center - parent_local_bbox.center

        child_verts_np = self._mesh_mgr.get_mesh_vertices(child_obj)
        parent_verts_np = self._mesh_mgr.get_mesh_vertices(parent_obj)
        child_verts_local = torch.tensor(child_verts_np, dtype=torch.float32, device=device)
        parent_verts_local = torch.tensor(parent_verts_np, dtype=torch.float32, device=device)

        child_wp_mesh = self._mesh_mgr.get_warp_mesh(child_obj)
        parent_wp_mesh = self._mesh_mgr.get_warp_mesh(parent_obj)

        total_loss = torch.zeros(batch, device=device, dtype=torch.float32)

        for env_idx in range(batch):
            c_pos = child_pos[env_idx]
            p_pos = parent_pos[env_idx] if parent_pos.dim() > 1 else parent_pos.squeeze(0)

            child_world = child_verts_local + c_pos
            query_in_parent = child_world - p_pos
            sdf_a = mesh_sdf(query_in_parent, parent_wp_mesh, device_str)
            pen_a = torch.nn.functional.relu(-sdf_a).mean()

            parent_world = parent_verts_local + p_pos
            query_in_child = parent_world - c_pos
            sdf_b = mesh_sdf(query_in_child, child_wp_mesh, device_str)
            pen_b = torch.nn.functional.relu(-sdf_b).mean()

            total_loss[env_idx] = self.slope * (pen_a + pen_b)

        result = relation.relation_loss_weight * total_loss
        return result.squeeze(0) if single_input else result

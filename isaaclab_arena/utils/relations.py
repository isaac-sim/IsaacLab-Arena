# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Spatial relationship classes for object placement constraints.
"""

import torch
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.utils.bounding_box import BoundingBox

# Avoid circular import by using TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab_arena.assets.dummy_object import DummyObject


# Global mapping of directional names to axis and side information
# This maps intuitive direction names to their corresponding world coordinate axes
DIRECTION_MAP = {
    "front": {"axis": 0, "side": "negative"},  # -X direction
    "back": {"axis": 0, "side": "positive"},   # +X direction
    "left": {"axis": 1, "side": "negative"},   # -Y direction
    "right": {"axis": 1, "side": "positive"},  # +Y direction
}


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
        self, 
        child: "DummyObject",
        child_pos: torch.Tensor | None = None,
        parent_pos: torch.Tensor | None = None
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
        exponential_k: float = 50.0,
        side: str = "right",
    ):
        """
        Args:
            parent: The parent asset that this object should be placed next to.
            relation_loss_weight: Weight for the relationship loss function.
            distance: Target distance from parent's boundary in meters (default: 5cm).
            tolerance: Tolerance band around target distance in meters (default: 1cm).
            exponential_k: Steepness of exponential penalty outside the band (default: 50.0).
            side: Which side to place object: "front" (-X), "back" (+X), "left" (+Y), or "right" (-Y).
        """
        super().__init__(parent, relation_loss_weight)
        self.distance = distance
        self.tolerance = tolerance
        self.exponential_k = exponential_k
        self.side = side
        print("NextTo side initialized", side)
    
    def compute_relation_loss(
        self, 
        child: "DummyObject",
        child_pos: torch.Tensor | None = None,
        parent_pos: torch.Tensor | None = None
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

        # Type assertions for type checker
        assert parent_pos is not None
        assert child_pos is not None
        
        # Get bounding boxes
        parent_bbox = self.parent.get_bounding_box()
        child_bbox = child.get_bounding_box()

        # Check if child is to the right of the parent. 
        # We set a half plane loss on the right side of the parent.
        parent_right_bound = parent_pos[0] + parent_bbox.size[0] / 2
        right_side_loss = single_boundary_exponential_loss(child_pos[0], parent_right_bound, penalty_side="less")

        # Add a loss on the top side of the parent.
        parent_top_bound = parent_pos[1] + parent_bbox.size[1] / 2
        top_side_loss = single_boundary_exponential_loss(child_pos[1], parent_top_bound, penalty_side="greater")

        parent_bottom_bound = parent_pos[1] - parent_bbox.size[1] / 2
        bottom_side_loss = single_boundary_exponential_loss(child_pos[1], parent_bottom_bound, penalty_side="less")

        
        total_side_loss = right_side_loss + top_side_loss + bottom_side_loss
        return total_side_loss / 3.0
    

def single_boundary_exponential_loss(
    value: torch.Tensor,
    boundary: torch.Tensor | float,
    exponential_k: float = 5000.0,
    penalty_side: str = "greater"
) -> torch.Tensor:
    """Compute exponential loss for violating a single boundary threshold.
    
    This is the primitive building block for band losses. It penalizes
    violations on ONE side of a boundary threshold.
    
    Args:
        value: Measured value (tensor for gradient flow).
        boundary: The boundary threshold value (can be tensor or float).
        exponential_k: Steepness of exponential penalty (default: 5000.0).
        penalty_side: Which side to penalize:
            - 'greater': Penalize if value > boundary (upper bound)
            - 'less': Penalize if value < boundary (lower bound)
    
    Returns:
        Loss value (scalar tensor) in range [0, 1].
        - 0 when constraint satisfied
        - Approaches 1 exponentially as violation increases
    
    Examples:
        >>> # Penalize if distance exceeds 0.05m
        >>> loss = single_boundary_exponential_loss(distance, 0.05, penalty_side='greater')
        
        >>> # Penalize if distance is below 0.02m
        >>> loss = single_boundary_exponential_loss(distance, 0.02, penalty_side='less')
    """
    # Ensure value is a tensor
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value, dtype=torch.float32)
    
    # Ensure boundary is a tensor for proper gradient flow
    if not isinstance(boundary, torch.Tensor):
        boundary = torch.tensor(boundary, dtype=value.dtype, device=value.device)
    
    zero = torch.tensor(0.0, dtype=value.dtype, device=value.device)
    
    if penalty_side == "greater":
        # Penalize if value > boundary
        violation = torch.maximum(zero, value - boundary)
    elif penalty_side == "less":
        # Penalize if value < boundary
        violation = torch.maximum(zero, boundary - value)
    else:
        raise ValueError(f"penalty_side must be 'greater' or 'less', got '{penalty_side}'")
    
    # Exponential penalty: 1 - exp(-k * violation)
    loss = 1.0 - torch.exp(-exponential_k * violation)
    
    return loss


def exponential_band_loss(
    value: torch.Tensor, 
    lower_bound: torch.Tensor | float, 
    upper_bound: torch.Tensor | float,
    exponential_k: float = 5000.0
) -> torch.Tensor:
    """Compute exponential band loss function.
    
    Loss is zero within [lower_bound, upper_bound]
    and increases exponentially outside this band, approaching 1.0 asymptotically.
    
    This is implemented as the sum of two mirrored single-boundary losses:
    one for the lower bound and one for the upper bound.
    
    Args:
        value: Measured value (tensor for gradient flow).
        lower_bound: Lower bound threshold (can be tensor or float).
        upper_bound: Upper bound threshold (can be tensor or float).
        exponential_k: Steepness of exponential penalty (default: 5000.0).
        
    Returns:
        Loss value (scalar tensor) in range [0, 1].
        - 0 when value is within [lower_bound, upper_bound]
        - Approaches 1 when violating both bounds
    """
    # Compose two single-boundary losses (mirrored)
    loss_lower = single_boundary_exponential_loss(value, lower_bound, exponential_k, penalty_side="less")
    loss_upper = single_boundary_exponential_loss(value, upper_bound, exponential_k, penalty_side="greater")
    
    return (loss_lower + loss_upper) / 2.0


#    """Represents an 'on top of' relationship between objects."""
#
#    def __init__(
#        self,
#        parent: Asset,
#        relation_loss_weight: float = 1.0,
#        xy_range: float = 0.05,
#        z_range: float = 0.02,
#        xy_weight: float = 1.0,
#        z_weight: float = 1.0,
#    ):
#        """
#        Args:
#            parent: The parent asset that this object should be placed on.
#            relation_loss_weight: Weight for the relationship loss function.
#            xy_range: Transition range for XY bound cost in meters (default: 0.05 = 5cm).
#            z_range: Transition range for Z height cost in meters (default: 0.02 = 2cm).
#            xy_weight: Weight for XY bound cost [0,1] (default: 1.0).
#            z_weight: Weight for Z height cost [0,1] (default: 1.0).
#        """
#        super().__init__(parent, relation_loss_weight)
#        self.xy_range = xy_range
#        self.z_range = z_range
#        self.xy_weight = xy_weight
#        self.z_weight = z_weight
#
#    def compute_xy_bound_cost(
#        self,
#        child_x: torch.Tensor,
#        child_y: torch.Tensor,
#        parent_x_min: torch.Tensor,
#        parent_x_max: torch.Tensor,
#        parent_y_min: torch.Tensor,
#        parent_y_max: torch.Tensor,
#    ) -> torch.Tensor:
#        """Compute normalized XY bound cost [0, 1] using exponential transition.
#        
#        Args:
#            child_x: X coordinate of child center
#            child_y: Y coordinate of child center
#            parent_x_min: Minimum X bound of parent
#            parent_x_max: Maximum X bound of parent
#            parent_y_min: Minimum Y bound of parent
#            parent_y_max: Maximum Y bound of parent
#            
#        Returns:
#            XY bound cost in range [0, 1]:
#                - 0.0 when child center is inside parent bounds
#                - Increases exponentially as child moves outside
#                - Approaches 1.0 far outside bounds
#        """
#        # Compute violations (zero if inside bounds)
#        zero = torch.tensor(0.0, dtype=child_x.dtype, device=child_x.device)
#        
#        x_violation = torch.maximum(zero, parent_x_min - child_x) + torch.maximum(
#            zero, child_x - parent_x_max
#        )
#        
#        y_violation = torch.maximum(zero, parent_y_min - child_y) + torch.maximum(
#            zero, child_y - parent_y_max
#        )
#        
#        # Manhattan distance from valid region
#        manhattan_distance = x_violation + y_violation
#        
#        # Exponential transition: cost = 1 - exp(-distance / range)
#        xy_cost = 1.0 - torch.exp(-manhattan_distance / self.xy_range)
#        
#        return xy_cost
#
#    def compute_loss(self, child: "Asset", child_pos: torch.Tensor, parent_pos: torch.Tensor | None = None) -> torch.Tensor:
#        """Compute the loss for the 'on top of' relationship constraint.
#
#        This loss function penalizes:
#        1. XY overlap violations: Heavy penalty when child extends outside parent's XY bounds
#        2. Z alignment violations: Penalty when child penetrates parent or is misaligned
#
#        Args:
#            child: The child object that should be placed on the parent.
#            child_pos: Proposed position tensor (x, y, z) of shape (3,) for the child object.
#            parent_pos: Position tensor (x, y, z) of shape (3,) for the parent object.
#                       If None, uses parent's initial_pose.
#
#        Returns:
#            Scalar loss tensor representing the constraint violation.
#        """
#        # Import here to avoid circular imports
#        from isaaclab_arena.assets.object import Object
#
#        # Ensure both parent and child are Object instances with bounding boxes
#        if not isinstance(self.parent, Object):
#            raise TypeError(f"Parent must be an Object instance, got {type(self.parent)}")
#        if not isinstance(child, Object):
#            raise TypeError(f"Child must be an Object instance, got {type(child)}")
#
#        # Get bounding boxes
#        parent_bbox = self.parent.get_bounding_box()
#        child_bbox = child.get_bounding_box()
#
#        # Get parent position
#        if parent_pos is None:
#            # Try to get from initial pose
#            parent_pose = self.parent.get_initial_pose()
#            if parent_pose is not None:
#                parent_pos = torch.tensor(parent_pose.position_xyz, dtype=child_pos.dtype, device=child_pos.device)
#            else:
#                # Fallback to bounding box center (not ideal)
#                parent_pos = torch.tensor(parent_bbox.center, dtype=child_pos.dtype, device=child_pos.device)
#        
#        # Ensure parent_pos is not None
#        assert parent_pos is not None, "Parent position must be provided or derivable from parent object"
#
#        # ========== XY Overlap Loss ==========
#        # Get the 8 corners of the child at the proposed position
#        child_corners = child.get_corners_aabb_axis_aligned(child_pos)  # Shape: (8, 3)
#
#        # Get parent's actual XY bounds at its position
#        parent_width, parent_depth, _ = parent_bbox.size
#        parent_x_min = parent_pos[0] - parent_width / 2
#        parent_x_max = parent_pos[0] + parent_width / 2
#        parent_y_min = parent_pos[1] - parent_depth / 2
#        parent_y_max = parent_pos[1] + parent_depth / 2
#
#        # For each corner, compute how far it extends outside the parent's XY bounds
#        xy_loss = torch.tensor(0.0, dtype=child_pos.dtype, device=child_pos.device)
#
#        for corner in child_corners:
#            x, y = corner[0], corner[1]
#
#            # Compute violation in X direction
#            x_violation = torch.maximum(
#                torch.tensor(0.0, dtype=child_pos.dtype, device=child_pos.device),
#                parent_x_min - x,
#            ) + torch.maximum(
#                torch.tensor(0.0, dtype=child_pos.dtype, device=child_pos.device),
#                x - parent_x_max,
#            )
#
#            # Compute violation in Y direction
#            y_violation = torch.maximum(
#                torch.tensor(0.0, dtype=child_pos.dtype, device=child_pos.device),
#                parent_y_min - y,
#            ) + torch.maximum(
#                torch.tensor(0.0, dtype=child_pos.dtype, device=child_pos.device),
#                y - parent_y_max,
#            )
#
#            # Add squared violations
#            xy_loss = xy_loss + x_violation**2 + y_violation**2
#
#        # ========== Z Alignment Loss ==========
#        # Calculate child's bottom surface
#        child_height = child_bbox.size[2]
#        child_z_bottom = child_pos[2] - child_height / 2
#
#        # Get parent's top surface (at its actual position)
#        parent_height = parent_bbox.size[2]
#        parent_z_top = parent_pos[2] + parent_height / 2
#
#        # Compute Z gap (positive means child is above parent, negative means penetration)
#        z_gap = child_z_bottom - parent_z_top
#
#        # Penalize penetration (when z_gap < 0)
#        z_loss = torch.maximum(
#            torch.tensor(0.0, dtype=child_pos.dtype, device=child_pos.device), -z_gap
#        ) ** 2
#
#        # ========== Combine Losses ==========
#        total_loss = self.relation_loss_weight * (self.xy_weight * xy_loss + self.z_weight * z_loss)
#
#        return total_loss
    
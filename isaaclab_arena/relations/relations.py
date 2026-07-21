# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from enum import Enum
from typing import TYPE_CHECKING, TypeVar

from isaaclab.utils.math import euler_xyz_from_quat

from isaaclab_arena.assets.register import register_object_relation
from isaaclab_arena.utils.pose import PoseRange  # runtime: constructed in to_pose_range_centered_at()

if TYPE_CHECKING:
    from isaaclab_arena.relations.placement_asset import PlacementAsset

RelationT = TypeVar("RelationT", bound="RelationBase")

# Default inward inset (meters) from each X/Y edge of an ``On`` support surface, keeping
# the placed object's footprint off the rim.
DEFAULT_ON_EDGE_MARGIN_M = 0.05


class Side(str, Enum):
    """Axis direction for spatial relationships."""

    POSITIVE_X = "positive_x"  # +X
    NEGATIVE_X = "negative_x"  # -X
    POSITIVE_Y = "positive_y"  # +Y
    NEGATIVE_Y = "negative_y"  # -Y


class RelationBase:
    """Base for all relation-like concepts on objects.

    This is the common base class for both spatial relations (On, NextTo, etc.)
    and markers (IsAnchor). It allows the Object class to store both types
    in its relations list.
    """

    def validate_placement_configuration(self, subject: PlacementAsset, objects: set[PlacementAsset]) -> None:
        """Validate this relation for placement among the participating objects."""


class UnaryRelation(RelationBase):
    """Base class for unary spatial relations (no parent object).

    Unary relations constrain an object's position in world coordinates
    without referencing another object (e.g., AtPosition, PositionLimits).
    """

    @staticmethod
    def is_unary() -> bool:
        """Return whether the relation constrains a single object."""
        return True


class Relation(RelationBase):
    """Base class for binary spatial relationships between objects."""

    @staticmethod
    def is_unary() -> bool:
        """Return whether the relation constrains a single object."""
        return False

    def __init__(self, parent: PlacementAsset, relation_loss_weight: float = 1.0):
        """
        Args:
            parent: The parent asset in the relationship.
            relation_loss_weight: Weight for the relationship loss function.
        """
        self.parent = parent
        self.relation_loss_weight = relation_loss_weight


@register_object_relation
class FaceTo(RelationBase):
    """Relates an object's local +X heading to a target object's world-XY direction.

    This orientation relation does not constrain either object's position. The
    subject's XY position must differ from the target's.
    """

    name = "face_to"

    def __init__(self, parent: PlacementAsset):
        """
        Args:
            parent: Target object that defines the facing direction.
        """
        self.parent = parent

    @staticmethod
    def is_unary() -> bool:
        """Return whether the relation constrains a single object."""
        return False

    def validate_placement_configuration(self, subject: PlacementAsset, objects: set[PlacementAsset]) -> None:
        """Validate the facing relation for its subject and target."""
        face_to_count = sum(isinstance(relation, FaceTo) for relation in subject.get_relations())
        assert face_to_count == 1, f"Object '{subject.name}' has more than one FaceTo relation."
        assert not subject.is_anchor, f"Anchor object '{subject.name}' cannot have a FaceTo relation."
        assert self.parent is not subject, f"Object '{subject.name}' cannot face itself."
        assert self.parent in objects, f"FaceTo parent '{self.parent.name}' must participate in placement."
        assert not any(
            isinstance(relation, RotateAroundSolution) for relation in subject.get_relations()
        ), f"Object '{subject.name}' cannot combine FaceTo with RotateAroundSolution."

        subject_randomization = next(
            (relation for relation in subject.get_relations() if isinstance(relation, RandomAroundSolution)), None
        )
        if subject_randomization is not None:
            assert (
                subject_randomization.x_half_m == 0.0 and subject_randomization.y_half_m == 0.0
            ), f"Object '{subject.name}' cannot randomize XY while using FaceTo."

        parent_randomization = next(
            (relation for relation in self.parent.get_relations() if isinstance(relation, RandomAroundSolution)), None
        )
        if parent_randomization is not None:
            assert (
                parent_randomization.x_half_m == 0.0 and parent_randomization.y_half_m == 0.0
            ), f"FaceTo parent '{self.parent.name}' cannot randomize XY."


@register_object_relation
class NextTo(Relation):
    """Represents a 'next to' relationship between objects.

    This relation specifies that a child object should be placed adjacent to
    the parent object on a specified side, at a given distance.

    Note: Loss computation is handled by NextToLossStrategy in relation_loss_strategies.py.
    """

    name = "next_to"

    def __init__(
        self,
        parent: PlacementAsset,
        relation_loss_weight: float = 1.0,
        distance_m: float = 0.05,
        side: Side | str = Side.POSITIVE_X,
        cross_position_ratio: float = 0.0,
        tolerance_m: float = 1e-2,
    ):
        """
        Args:
            parent: The parent asset that this object should be placed next to.
            relation_loss_weight: Weight for the relationship loss function.
            distance_m: Target distance from parent's boundary in meters (default: 5cm).
            side: Which axis direction to place object (default: Side.POSITIVE_X).
            cross_position_ratio: Where to place the child along the parent's perpendicular
                (cross) axis, from -1.0 (min edge) through 0.0 (centered) to 1.0 (max edge).
                The cross axis depends on the side: for POSITIVE_X / NEGATIVE_X
                the cross axis is Y; for POSITIVE_Y / NEGATIVE_Y it is X.
                Default: 0.0 (centered).
            tolerance_m: Slack (meters) for placement validation: the child passes when its side and
                distance violations are each within this value. Raise it to accept the side but care
                less about the exact distance. cross_position_ratio is a soft preference, not gated.
        """
        super().__init__(parent, relation_loss_weight)
        assert distance_m > 0.0, f"Distance must be positive, got {distance_m}"
        assert (
            -1.0 <= cross_position_ratio <= 1.0
        ), f"cross_position_ratio must be in [-1, 1], got {cross_position_ratio}"
        assert tolerance_m >= 0.0, f"tolerance_m must be non-negative, got {tolerance_m}"
        self.distance_m = distance_m
        self.side = Side(side)
        self.cross_position_ratio = cross_position_ratio
        self.tolerance_m = tolerance_m


@register_object_relation
class On(Relation):
    """Represents an 'on top of' relationship between objects.

    This relation specifies that a child object should be placed on top of
    the parent object, with X/Y bounded within the parent's extent (optionally
    inset by ``edge_margin_m`` so the child stays off the rim) and Z positioned
    on the parent's top surface.

    Note: Loss computation is handled by OnLossStrategy in relation_loss_strategies.py.
    """

    name = "on"

    def __init__(
        self,
        parent: PlacementAsset,
        relation_loss_weight: float = 1.0,
        clearance_m: float = 0.01,
        edge_margin_m: float = DEFAULT_ON_EDGE_MARGIN_M,
    ):
        """
        Args:
            parent: The parent asset that this object should be placed on top of.
            relation_loss_weight: Weight for the relationship loss function.
            clearance_m: Safety clearance above parent's surface in meters (default: 1cm).
            edge_margin_m: Inward inset from each X/Y edge of the parent's surface in
                meters (default: 5cm). The child's whole footprint is kept at least this
                far from the rim. The solver rejects a margin too large for the surface
                to honor (``2 * edge_margin_m`` wider than ``parent_extent - child_extent``
                on either axis).
        """
        super().__init__(parent, relation_loss_weight)
        assert clearance_m >= 0.0, f"Clearance must be non-negative, got {clearance_m}"
        assert edge_margin_m >= 0.0, f"edge_margin_m must be non-negative, got {edge_margin_m}"
        self.clearance_m = clearance_m
        self.edge_margin_m = edge_margin_m


@register_object_relation
class NotNextTo(Relation):
    """Forbids placing the child next to the parent on the given side.

    The inverse of ``NextTo``. Blocks the whole half-plane past the parent's edge
    on that side, within the parent's perpendicular footprint (for ``side=+Y``:
    all of ``+Y`` past the ``+Y`` edge, clipped to the parent's ``X`` extent).
    Anywhere off to either side of that footprint stays free, at any distance.

    Note: Loss computation is handled by NotNextToLossStrategy in relation_loss_strategies.py.
    """

    name = "not_next_to"

    def __init__(
        self,
        parent: PlacementAsset,
        relation_loss_weight: float = 1.0,
        side: Side | str = Side.POSITIVE_X,
        tolerance_m: float = 1e-2,
    ):
        """
        Args:
            parent: The parent asset whose adjacent half-plane is forbidden.
            relation_loss_weight: Weight for the relationship loss function.
            side: Which side of the parent is blocked (default: Side.POSITIVE_X).
            tolerance_m: Slack (meters) for placement validation: the child passes when it has cleared
                the keep-out zone to within this value.
        """
        super().__init__(parent, relation_loss_weight)
        assert tolerance_m >= 0.0, f"tolerance_m must be non-negative, got {tolerance_m}"
        self.side = Side(side)
        self.tolerance_m = tolerance_m


@register_object_relation
class IsAnchor(RelationBase):
    """Marker indicating this object is an anchor for relation solving.

    Anchor objects are fixed references that won't be optimized during
    relation solving. Multiple objects can be marked as anchors.
    Each anchor must have an initial_pose set before calling ObjectPlacer.place().

    Usage:
        table.set_initial_pose(Pose(position_xyz=(1.0, 0.0, 0.0), ...))
        table.add_relation(IsAnchor())  # Mark as anchor

        chair.set_initial_pose(Pose(position_xyz=(2.0, 0.0, 0.0), ...))
        chair.add_relation(IsAnchor())  # Another anchor

        mug.add_relation(On(table))
        bin.add_relation(NextTo(chair))
    """

    name = "is_anchor"

    @staticmethod
    def is_unary() -> bool:
        """Return whether the relation constrains a single object."""
        return True


@register_object_relation
class RandomAroundSolution(RelationBase):
    """Marker indicating the solver solution should be used as center of a PoseRange.

    When ObjectPlacer applies positions, objects with this marker will get a PoseRange
    (enabling randomization at environment reset) instead of a fixed Pose.

    The half extents define a box centered on the solved position. At each environment
    reset, a random position within this box will be sampled.

    Note: This is NOT a spatial relation - the RelationSolver ignores it. It only
    affects how ObjectPlacer applies the solved position to the object.

    Usage:
        box.add_relation(On(desk))
        box.add_relation(RandomAroundSolution(x_half_m=0.1, y_half_m=0.1))
        # -> ObjectPlacer sets a PoseRange spanning ±0.1m in X and Y around solved position
    """

    name = "random_around_solution"

    @staticmethod
    def is_unary() -> bool:
        """Return whether the relation constrains a single object."""
        return True

    def __init__(
        self,
        x_half_m: float = 0.0,
        y_half_m: float = 0.0,
        z_half_m: float = 0.0,
        roll_half_rad: float = 0.0,
        pitch_half_rad: float = 0.0,
        yaw_half_rad: float = 0.0,
    ):
        """
        Args:
            x_half_m: Half-extent in X direction (meters). Position will be randomized ±x_half_m.
            y_half_m: Half-extent in Y direction (meters). Position will be randomized ±y_half_m.
            z_half_m: Half-extent in Z direction (meters). Position will be randomized ±z_half_m.
            roll_half_rad: Half-extent for roll (radians). Rotation will be randomized ±roll_half_rad.
            pitch_half_rad: Half-extent for pitch (radians). Rotation will be randomized ±pitch_half_rad.
            yaw_half_rad: Half-extent for yaw (radians). Rotation will be randomized ±yaw_half_rad.
        """
        self.x_half_m = x_half_m
        self.y_half_m = y_half_m
        self.z_half_m = z_half_m
        self.roll_half_rad = roll_half_rad
        self.pitch_half_rad = pitch_half_rad
        self.yaw_half_rad = yaw_half_rad

    def to_pose_range_centered_at(
        self,
        position: tuple[float, float, float],
        rotation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    ) -> PoseRange:
        """Create a PoseRange centered on the given position and rotation.

        Args:
            position: Center position (x, y, z) for the range.
            rotation_xyzw: Center rotation as quaternion (x, y, z, w) for the range.
                Defaults to identity quaternion.

        Returns:
            PoseRange spanning ± half-extents around the position and rotation.
        """
        # Convert quaternion to euler angles (roll, pitch, yaw)
        quat_tensor = torch.tensor([rotation_xyzw])
        roll, pitch, yaw = euler_xyz_from_quat(quat_tensor)
        center_roll = float(roll[0])
        center_pitch = float(pitch[0])
        center_yaw = float(yaw[0])

        return PoseRange(
            position_xyz_min=(
                position[0] - self.x_half_m,
                position[1] - self.y_half_m,
                position[2] - self.z_half_m,
            ),
            position_xyz_max=(
                position[0] + self.x_half_m,
                position[1] + self.y_half_m,
                position[2] + self.z_half_m,
            ),
            rpy_min=(
                center_roll - self.roll_half_rad,
                center_pitch - self.pitch_half_rad,
                center_yaw - self.yaw_half_rad,
            ),
            rpy_max=(
                center_roll + self.roll_half_rad,
                center_pitch + self.pitch_half_rad,
                center_yaw + self.yaw_half_rad,
            ),
        )


@register_object_relation
class RotateAroundSolution(RelationBase):
    """Marker specifying an explicit rotation to apply on top of the solver solution.

    When ObjectPlacer applies positions, objects with this marker will have the
    specified rotation applied on top of the solved position to create a fixed Pose.

    Note: This is NOT a spatial relation - the RelationSolver ignores it. It only
    affects how ObjectPlacer applies the solved position to the object.

    Usage:
        import math
        box.add_relation(On(desk))
        box.add_relation(RotateAroundSolution(yaw_rad=math.pi / 4))
        # -> ObjectPlacer sets a Pose with solved position and 45° yaw rotation
    """

    name = "rotate_around_solution"

    @staticmethod
    def is_unary() -> bool:
        """Return whether the relation constrains a single object."""
        return True

    def __init__(
        self,
        roll_rad: float = 0.0,
        pitch_rad: float = 0.0,
        yaw_rad: float = 0.0,
    ):
        """
        Args:
            roll_rad: Roll rotation in radians.
            pitch_rad: Pitch rotation in radians.
            yaw_rad: Yaw rotation in radians.
        """
        self.roll_rad = roll_rad
        self.pitch_rad = pitch_rad
        self.yaw_rad = yaw_rad

    def get_rotation_xyzw(self) -> tuple[float, float, float, float]:
        """Get the rotation as a quaternion (x, y, z, w).

        Returns:
            Quaternion rotation converted from roll/pitch/yaw.
        """
        import torch

        from isaaclab.utils.math import quat_from_euler_xyz

        roll = torch.tensor(self.roll_rad)
        pitch = torch.tensor(self.pitch_rad)
        yaw = torch.tensor(self.yaw_rad)
        quat = quat_from_euler_xyz(roll, pitch, yaw)
        return tuple(quat.tolist())


@register_object_relation
class AtPosition(UnaryRelation):
    """Constrains object to specific world coordinates.

    This is a unary relation (no parent) that pins an object's position to
    specific x, y, and/or z world coordinates. Any axis set to None is
    unconstrained by this relation (allowing other relations like On to
    control that axis).

    Note: Loss computation is handled by AtPositionLossStrategy in relation_loss_strategies.py.

    Usage:
        # Pin object to x=0.5, y=1.0 in world coords (z controlled by On relation)
        mug.add_relation(On(table))
        mug.add_relation(AtPosition(x=0.5, y=1.0))
    """

    name = "at_position"

    def __init__(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        relation_loss_weight: float = 1.0,
    ):
        """
        Args:
            x: Target x world coordinate, or None to leave unconstrained.
            y: Target y world coordinate, or None to leave unconstrained.
            z: Target z world coordinate, or None to leave unconstrained.
            relation_loss_weight: Weight for the relationship loss function.
        """
        assert (
            x is not None or y is not None or z is not None
        ), "At least one of x, y, or z must be specified for AtPosition"
        self.x = x
        self.y = y
        self.z = z
        self.relation_loss_weight = relation_loss_weight


@register_object_relation
class PositionLimits(UnaryRelation):
    """Constrains object position to a world-coordinate axis-aligned box.

    Each axis is independently optional (None = unconstrained).

    Usage:
        mug.add_relation(PositionLimits(x_min=-0.5, x_max=0.5, y_min=-0.5, y_max=0.5))
        mug.add_relation(PositionLimits(z_min=0.8))  # only constrain Z
    """

    name = "position_limits"

    def __init__(
        self,
        x_min: float | None = None,
        x_max: float | None = None,
        y_min: float | None = None,
        y_max: float | None = None,
        z_min: float | None = None,
        z_max: float | None = None,
        relation_loss_weight: float = 1.0,
    ):
        assert (
            x_min is not None
            or x_max is not None
            or y_min is not None
            or y_max is not None
            or z_min is not None
            or z_max is not None
        ), "At least one bound (x_min, x_max, y_min, y_max, z_min, or z_max) must be specified for PositionLimits"
        if x_min is not None and x_max is not None:
            assert x_min < x_max, f"x_min must be less than x_max, got x_min={x_min}, x_max={x_max}"
        if y_min is not None and y_max is not None:
            assert y_min < y_max, f"y_min must be less than y_max, got y_min={y_min}, y_max={y_max}"
        if z_min is not None and z_max is not None:
            assert z_min < z_max, f"z_min must be less than z_max, got z_min={z_min}, z_max={z_max}"
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
        self.relation_loss_weight = relation_loss_weight


def get_anchor_objects(objects: list[PlacementAsset]) -> list[PlacementAsset]:
    """Get all anchor objects from a list of objects.

    Anchor objects are marked with IsAnchor() relation and serve as
    fixed reference points for relation solving.

    Args:
        objects: List of objects to filter.

    Returns:
        List of anchor objects (may be empty if no anchors found).
    """
    return [obj for obj in objects if any(isinstance(r, IsAnchor) for r in obj.get_relations())]


def get_relation(obj: PlacementAsset, relation_type: type[RelationT]) -> RelationT | None:
    """Return obj's first relation of the given type, or None if it has none."""
    return next((relation for relation in obj.get_relations() if isinstance(relation, relation_type)), None)

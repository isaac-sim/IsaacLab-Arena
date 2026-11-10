# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import Enum
from typing import TYPE_CHECKING, TypeVar

from isaaclab_arena.assets.register import agent_ready, register_object_relation
from isaaclab_arena.assets.registries import ObjectRelationLibraryRegistry
from isaaclab_arena.relations.axis import Axis
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox, quaternion_to_90_deg_z_quarters

if TYPE_CHECKING:
    from isaaclab_arena.relations.placement_asset import PlaceableAsset

RelationT = TypeVar("RelationT", bound="RelationBase")

# Default inward inset (meters) from each X/Y edge of an ``On`` support surface, keeping
# the placed object's footprint off the rim.
DEFAULT_ON_EDGE_MARGIN_M = 0.05

CLUTTER_RELEASE_HEIGHT_TOLERANCE_M = 1e-6
"""Numerical slack on release clearance in metres; never permits support penetration."""


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

    def validate_placement_configuration(self, subject: PlaceableAsset, objects: set[PlaceableAsset]) -> None:
        """Validate this relation for placement among the participating objects."""


class UnaryRelation(RelationBase):
    """Base class for unary spatial relations (no parent object).

    Unary relations constrain an object's position in world coordinates
    without referencing another object (e.g., AtPosition, PositionLimitsBox).
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

    def __init__(self, parent: PlaceableAsset, relation_loss_weight: float = 1.0):
        """
        Args:
            parent: The parent asset in the relationship.
            relation_loss_weight: Weight for the relationship loss function.
        """
        self.parent = parent
        self.relation_loss_weight = relation_loss_weight


@register_object_relation
class FaceTo(RelationBase):
    """Turns the subject so its local +X axis points at the reference in world XY.

    Sets the heading only; it does not constrain either object's position. The heading is computed
    after positions are solved, from the subject's position towards the reference's, and replaces
    any random yaw from ``random_yaw_init``.

    Validation:
        The ``face_to`` check rejects layouts where the subject and reference share an XY
        position, so that no facing direction is defined.

    Restrictions:
        A subject can have only one FaceTo and cannot combine it with
        RotateAroundSolution. An anchor cannot use FaceTo, and the reference must take part in placement.
        The turn is about the subject's origin, so if the subject's placement bounds are not centred on
        its origin, such as a robot on an offset stand, the turn moves the bounds' centre. A horizontal
        AlignWith that uses those bounds, as subject or reference, would then no longer hold, so
        placement raises an error for that combination. Use a fixed RotateAroundSolution heading or
        centred placement bounds instead.
    """

    name = "face_to"

    def __init__(self, parent: PlaceableAsset):
        """
        Args:
            parent: Target object that defines the facing direction.
        """
        self.parent = parent

    @staticmethod
    def is_unary() -> bool:
        """Return whether the relation constrains a single object."""
        return False

    def validate_placement_configuration(self, subject: PlaceableAsset, objects: set[PlaceableAsset]) -> None:
        """Validate the facing relation for its subject and target."""
        face_to_count = sum(isinstance(relation, FaceTo) for relation in subject.get_relations())
        assert face_to_count == 1, f"Object '{subject.name}' has more than one FaceTo relation."
        assert not subject.is_anchor, f"Anchor object '{subject.name}' cannot have a FaceTo relation."
        assert self.parent is not subject, f"Object '{subject.name}' cannot face itself."
        assert self.parent in objects, f"FaceTo parent '{self.parent.name}' must participate in placement."
        assert not any(
            isinstance(relation, RotateAroundSolution) for relation in subject.get_relations()
        ), f"Object '{subject.name}' cannot combine FaceTo with RotateAroundSolution."


@agent_ready
@register_object_relation
class AlignWith(Relation):
    """Aligns the subject's placement-box centre with the reference's along one world axis.

    The subject's centre is placed at the reference's centre plus ``offset_m`` along ``axis``; the
    other axes stay free. Combine it with NextTo to set a gap on one horizontal axis and alignment on
    the other, or use two AlignWith relations for a target region relative to the reference's centre.

    Validation:
        The ``align_with`` check rejects layouts whose centre error along ``axis`` exceeds
        ``tolerance_m``.

    Restrictions:
        The subject cannot be an anchor, and the reference must be a different object that
        takes part in placement. Horizontal alignment cannot use an object that FaceTo turns if that
        object's placement bounds are off-centre; see FaceTo.
    """

    name = "align_with"

    def __init__(
        self,
        parent: PlaceableAsset,
        axis: Axis | str,
        offset_m: float = 0.0,
        tolerance_m: float = 0.02,
        relation_loss_weight: float = 1.0,
    ):
        """
        Args:
            parent: Reference asset participating in placement.
            axis: World axis to align along.
            offset_m: Signed subject-minus-reference centre displacement along that axis.
            tolerance_m: Zero-penalty alignment band and maximum error accepted by placement validation.
            relation_loss_weight: Weight of the alignment objective during solving.
        """
        super().__init__(parent, relation_loss_weight)
        assert math.isfinite(offset_m), "offset_m must be finite"
        assert math.isfinite(tolerance_m) and tolerance_m >= 0, "tolerance_m must be finite and non-negative"
        assert math.isfinite(relation_loss_weight) and relation_loss_weight > 0, "Alignment weight must be positive"
        self.axis = Axis(axis)
        self.offset_m = offset_m
        self.tolerance_m = tolerance_m

    def validate_placement_configuration(self, subject: PlaceableAsset, objects: set[PlaceableAsset]) -> None:
        """Require a movable subject and a distinct reference that participates in placement."""
        assert self.parent is not subject, "AlignWith requires a different reference asset"
        assert not subject.is_anchor, f"Anchor object '{subject.name}' cannot have an AlignWith relation"
        assert self.parent in objects, f"AlignWith reference '{self.parent.name}' must participate in placement"


@agent_ready
@register_object_relation
class NextTo(Relation):
    """Places the subject beside the reference on one side, with a set gap between their placement boxes.

    ``side`` picks the reference's side. The gap between the reference's edge on that side and the
    subject's facing edge targets ``distance_m``. Along the other horizontal axis the subject may sit
    anywhere within the reference's extent, unless ``cross_position_ratio`` sets a soft preference.
    The solver treats any gap within ``tolerance_m`` of ``distance_m`` as satisfied, and the default
    initializer starts the subject on the requested side within ``tolerance_m``.

    Validation:
        The ``next_to`` check rejects layouts where the subject is on the wrong side, or its
        gap differs from ``distance_m`` by more than ``tolerance_m``, including when it is closer than
        requested. ``cross_position_ratio`` is not checked.
    """

    name = "next_to"

    def __init__(
        self,
        parent: PlaceableAsset,
        side: Side | str,
        relation_loss_weight: float = 1.0,
        distance_m: float = 0.05,
        cross_position_ratio: float | None = None,
        tolerance_m: float = 1e-2,
    ):
        """
        Args:
            parent: The parent asset that this object should be placed next to.
            side: Side of the parent, in world axes, to place this object on.
            relation_loss_weight: Weight for the relationship loss function.
            distance_m: Target gap in metres between the parent's edge and this object's facing edge.
            cross_position_ratio: Soft preference for where to place this object along the parent's
                other horizontal axis (Y for an X side, X for a Y side), from -1.0 (min edge) through
                0.0 (centred) to 1.0 (max edge). None allows any position within the parent's extent.
            tolerance_m: Allowed gap error in metres. The solver treats any gap within this value of
                distance_m as satisfied, and validation passes when the side and gap errors are each
                within it.
        """
        super().__init__(parent, relation_loss_weight)
        assert distance_m > 0.0, f"Distance must be positive, got {distance_m}"
        assert (
            cross_position_ratio is None or -1.0 <= cross_position_ratio <= 1.0
        ), f"cross_position_ratio must be None or in [-1, 1], got {cross_position_ratio}"
        assert tolerance_m >= 0.0, f"tolerance_m must be non-negative, got {tolerance_m}"
        self.distance_m = distance_m
        self.side = Side(side)
        self.cross_position_ratio = cross_position_ratio
        self.tolerance_m = tolerance_m


@agent_ready
@register_object_relation
class On(Relation):
    """Places the subject on top of the reference, with its footprint inside the reference's top surface.

    Uses the top and XY footprint of the reference's axis-aligned placement box, inset by
    ``edge_margin_m`` so the subject stays off the rim. With ``overlap=True`` the footprints only need
    to overlap on both horizontal axes and the margin is ignored; the subject may then tip or fall.
    For L-shaped, hollow or concave supports, anchor an ObjectReference to the actual support surface.

    Validation:
        The ``on_relation`` check rejects layouts where the subject's bottom is not within
        ``clearance_m`` above the reference's top, give or take ``ObjectPlacerParams.on_relation_z_tolerance_m``,
        or where its footprint leaves the inset support (or, with ``overlap=True``, does not overlap it).
    """

    name = "on"

    def __init__(
        self,
        parent: PlaceableAsset,
        relation_loss_weight: float = 1.0,
        clearance_m: float = 0.01,
        edge_margin_m: float = DEFAULT_ON_EDGE_MARGIN_M,
        overlap: bool = False,
    ):
        """
        Args:
            parent: The parent asset that this object should be placed on top of.
            relation_loss_weight: Weight for the relationship loss function.
            clearance_m: Allowed gap in metres between the parent's top and this object's bottom.
            edge_margin_m: Inward inset in metres from each X/Y edge of the parent's surface, applied
                only for containment. Ignored when overlap is True.
            overlap: Require only footprint overlap on both X and Y instead of full containment.
        """
        super().__init__(parent, relation_loss_weight)
        assert clearance_m >= 0.0, f"Clearance must be non-negative, got {clearance_m}"
        assert edge_margin_m >= 0.0, f"edge_margin_m must be non-negative, got {edge_margin_m}"
        self.clearance_m = clearance_m
        self.edge_margin_m = edge_margin_m
        self.overlap = overlap

    def is_child_bottom_z_within_placement_height_range(
        self, bottom: float, support_top: float, tolerance_m: float
    ) -> bool:
        """Whether the child bottom lies in the support's contact band, in metres."""
        return support_top - tolerance_m < bottom <= support_top + self.clearance_m + tolerance_m


@register_object_relation
class ClutterOn(On):
    """Releases the subject above a support, inside a central release region, for physics to settle.

    Defines a release pose before physics, not a resting pose. The release region is the central
    ``spread`` fraction of the support's width and depth, and follows the support if the solver moves
    it. The subject's bottom must clear the support by ``clearance_m``; there is no upper height
    limit. The support may be an anchor or positioned through its own relations.

    Validation:
        The ``clutter_on_relation`` check rejects layouts whose footprint leaves the release
        region or whose bottom is below the minimum clearance. It does not check the settled pile.

    Restrictions:
        ClutterOn must be the subject's only spatial relation. The support must take part
        in placement, cannot itself use ClutterOn, and must stay upright at a fixed yaw that is a multiple
        of 90 degrees; a movable support cannot use FaceTo. Roll or pitch on the subject requires
        ``collision_mode="bbox"``. Not offered to agentic environment generation.
    """

    name = "clutter_on"
    agent_ready = False
    """Excluded from agentic generation; overrides On's inherited opt-in."""

    def __init__(
        self,
        parent: PlaceableAsset,
        spread: float = 0.2,
        clearance_m: float = 0.01,
        gap_m: float = 0.03,
        random_yaw: bool = True,
        edge_margin_m: float = 0.0,
        relation_loss_weight: float = 1.0,
    ):
        """
        Args:
            parent: Support fixed with IsAnchor or positioned through its own relations.
            spread: Fraction of support width and depth available for release, centered on the support.
                For example, 0.2 restricts each axis to its central 20%; settled objects may use the full support.
            clearance_m: Release clearance above the support in metres.
            gap_m: Initial clearance from neighboring release bounds in metres, at least the solver clearance.
            random_yaw: Sample world-Z yaw in addition to RotateAroundSolution.
            edge_margin_m: Inward margin within the release region in metres.
            relation_loss_weight: Weight for the relation loss.
        """
        super().__init__(parent, relation_loss_weight, clearance_m=clearance_m, edge_margin_m=edge_margin_m)
        assert 0 < spread <= 1, "spread must be in (0, 1]"
        assert isinstance(random_yaw, bool), "random_yaw must be a boolean"
        assert (
            math.isfinite(relation_loss_weight) and relation_loss_weight >= 0
        ), "relation_loss_weight must be finite and non-negative"
        for name, value in (("gap_m", gap_m), ("clearance_m", clearance_m), ("edge_margin_m", edge_margin_m)):
            assert math.isfinite(value) and value >= 0, f"{name} must be finite and non-negative"
        self.spread = spread
        self.gap_m = gap_m
        self.random_yaw = random_yaw

    def get_release_region_bbox(self, bbox: AxisAlignedBoundingBox) -> AxisAlignedBoundingBox:
        """Return the support bounds with its XY footprint scaled by spread."""
        lower, upper = bbox.min_point.clone(), bbox.max_point.clone()
        center = (lower[:, :2] + upper[:, :2]) * 0.5
        half_size = (upper[:, :2] - lower[:, :2]) * (0.5 * self.spread)
        lower[:, :2], upper[:, :2] = center - half_size, center + half_size
        return AxisAlignedBoundingBox(lower, upper)

    def is_child_bottom_z_within_placement_height_range(
        self, bottom: float, support_top: float, tolerance_m: float
    ) -> bool:
        """Require nonpenetrating clearance, with numerical slack and no upper height limit.

        The ordinary On contact tolerance does not apply to release poses.
        """
        minimum_bottom = support_top + max(0.0, self.clearance_m - CLUTTER_RELEASE_HEIGHT_TOLERANCE_M)
        return bottom >= minimum_bottom

    def validate_placement_configuration(self, subject: PlaceableAsset, objects: set[PlaceableAsset]) -> None:
        """Require a participating support and preserve validated release poses."""
        assert self.parent in objects, "ClutterOn support must participate in placement"
        assert get_relation(self.parent, ClutterOn) is None, "Nested ClutterOn supports are unsupported"
        assert not subject.is_anchor, "ClutterOn cannot be an anchor"
        assert subject.get_spatial_relations() == [self], "ClutterOn must be the object's only spatial relation"

        support = self.parent
        if not support.is_anchor:
            assert get_relation(support, FaceTo) is None, f"ClutterOn support '{support.name}' cannot use FaceTo"
            rotation = get_relation(support, RotateAroundSolution)
            if rotation is not None:
                # Release regions require axis-aligned support bounds.
                try:
                    quaternion_to_90_deg_z_quarters(rotation.get_rotation_xyzw())
                except AssertionError as error:
                    raise AssertionError(f"ClutterOn support '{support.name}': {error}") from error


@agent_ready
@register_object_relation
class NotNextTo(Relation):
    """Keeps the subject out of the region beside the reference on one side.

    The inverse of NextTo. The keep-out region starts at the reference's edge on ``side``, extends
    outward without limit, and spans the reference's footprint along the other horizontal axis: for
    ``side=positive_y``, everything past the reference's +Y edge within its X extent. Positions off
    either end of that footprint stay free at any distance. The subject must clear the region by a
    margin, 0.1 m by default, set by ``NotNextToLossStrategy.margin_m`` in
    ``RelationSolverParams.strategies``.

    Validation:
        The ``not_next_to`` check rejects layouts where the subject has not cleared the
        keep-out region, to within ``tolerance_m``.
    """

    name = "not_next_to"

    def __init__(
        self,
        parent: PlaceableAsset,
        relation_loss_weight: float = 1.0,
        side: Side | str = Side.POSITIVE_X,
        tolerance_m: float = 1e-2,
    ):
        """
        Args:
            parent: The parent asset whose adjacent region is forbidden.
            relation_loss_weight: Weight for the relationship loss function.
            side: Side of the parent, in world axes, that is blocked.
            tolerance_m: Slack (meters) for placement validation: the child passes when it has cleared
                the keep-out zone to within this value.
        """
        super().__init__(parent, relation_loss_weight)
        assert tolerance_m >= 0.0, f"tolerance_m must be non-negative, got {tolerance_m}"
        self.side = Side(side)
        self.tolerance_m = tolerance_m


@agent_ready
@register_object_relation
class IsAnchor(RelationBase):
    """Marks the subject as a fixed reference that the solver does not move.

    An anchor keeps its initial pose, which must be set before placement; in a graph YAML an omitted
    pose defaults to identity, and an ObjectReference takes its pose from the referenced prim. Several
    objects can be anchors. Anchoring does not make an object static or kinematic in physics.

    Validation:
        None; anchors are not moved.

    Restrictions:
        An anchor's root rotation must be a multiple of 90 degrees about world Z, with no
        tilt.

    Example:
        >>> table.set_initial_pose(Pose(position_xyz=(1.0, 0.0, 0.0), ...))
        >>> table.add_relation(IsAnchor())  # Mark as anchor

        >>> chair.set_initial_pose(Pose(position_xyz=(2.0, 0.0, 0.0), ...))
        >>> chair.add_relation(IsAnchor())  # Another anchor

        >>> mug.add_relation(On(table))
        >>> bin.add_relation(NextTo(chair, side=Side.POSITIVE_X))
    """

    name = "is_anchor"

    @staticmethod
    def is_unary() -> bool:
        """Return whether the relation constrains a single object."""
        return True


@register_object_relation
class RequiresReachability(RelationBase):
    """Requires the robot to reach a top-down grasp at the subject.

    Adds no solver objective; it only rejects layouts. Tasks such as pick and place add it to the
    objects they need the robot to grasp.

    Validation:
        The ``ik_reachable`` check solves cuRobo IK for a top-down grasp at the subject from
        the robot's placed base, and rejects layouts where it fails. The check runs only when
        ``ObjectPlacerParams.reachability_config.embodiment`` has a cuRobo configuration.

    Example:
        >>> banana.add_relation(RequiresReachability())  # IK-check reachability of the banana
    """

    name = "requires_reachability"

    @staticmethod
    def is_unary() -> bool:
        """Return whether the relation constrains a single object."""
        return True


@agent_ready
@register_object_relation
class RotateAroundSolution(RelationBase):
    """Applies a fixed rotation to the subject on top of its solved position.

    The solver does not optimise it; placement composes this rotation with the solved position. Use
    it for a fixed heading, such as a robot facing a counter. Because the heading is known before
    solving, it works with AlignWith on objects whose placement bounds are off-centre, unlike FaceTo.
    A ClutterOn subject keeps it as its base rotation.

    Validation:
        None.

    Restrictions:
        Cannot be combined with FaceTo on the same subject.

    Example:
        >>> import math
        >>> box.add_relation(On(desk))
        >>> box.add_relation(RotateAroundSolution(yaw_rad=math.pi / 4))
        >>> # -> ObjectPlacer sets a Pose with solved position and 45° yaw rotation
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
    """Pulls the subject towards fixed world coordinates on the selected axes.

    Axes left as None stay free, so another relation such as On can set them.

    Validation:
        None; this is a solver objective only, so a valid layout can miss the target slightly.

    Example:
        >>> # Pin object to x=0.5, y=1.0 in world coords (z controlled by On relation)
        >>> mug.add_relation(On(table))
        >>> mug.add_relation(AtPosition(x=0.5, y=1.0))
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
class PositionLimitsBox(UnaryRelation):
    """Keeps the subject's position within an axis-aligned box in world coordinates.

    Give each axis as a (minimum, maximum) pair; None leaves that bound open, and omitted axes are
    free.

    Validation:
        None; this is a solver objective only, so a valid layout can exceed the limits
        slightly.

    Example:
        >>> mug.add_relation(PositionLimitsBox(x_limits=(-0.5, 0.5), y_limits=(-0.5, 0.5)))
        >>> mug.add_relation(PositionLimitsBox(z_limits=(0.8, None)))  # only constrain Z
    """

    name = "position_limits_box"

    def __init__(
        self,
        x_limits: Sequence[float | None] = (None, None),
        y_limits: Sequence[float | None] = (None, None),
        z_limits: Sequence[float | None] = (None, None),
        relation_loss_weight: float = 1.0,
    ):
        """
        Args:
            x_limits: Minimum and maximum world X in metres; None leaves a bound open.
            y_limits: Minimum and maximum world Y in metres; None leaves a bound open.
            z_limits: Minimum and maximum world Z in metres; None leaves a bound open.
            relation_loss_weight: Weight for the relationship loss function.
        """
        for axis, limits in (("x", x_limits), ("y", y_limits), ("z", z_limits)):
            assert len(limits) == 2, f"{axis}_limits must contain a minimum and maximum"
            lower, upper = limits
            assert lower is None or upper is None or lower < upper, f"{axis}_limits minimum must be less than maximum"
        assert any(
            bound is not None for bound in (*x_limits, *y_limits, *z_limits)
        ), "At least one axis bound must be specified for PositionLimitsBox"
        self.x_min, self.x_max = x_limits
        self.y_min, self.y_max = y_limits
        self.z_min, self.z_max = z_limits
        self.relation_loss_weight = relation_loss_weight


# Keep the original Python and YAML names as aliases for box limits.
PositionLimits = PositionLimitsBox
_relation_registry = ObjectRelationLibraryRegistry()
if not _relation_registry.is_registered("position_limits", ensure_loaded=False):
    _relation_registry.register(PositionLimitsBox, "position_limits")


@register_object_relation
class PositionLimitsCylindrical(UnaryRelation):
    """Keeps the subject's world-XY distance from a fixed centre within a radius range.

    ``radius_min`` alone keeps the subject outside a cylinder, ``radius_max`` alone keeps it inside
    one, and both give an annulus. It does not constrain Z; use PositionLimitsBox for that or for
    axis-aligned XY limits.

    Validation:
        None; this is a solver objective only, so a valid layout can exceed the limits
        slightly.

    Example:
        >>> mug.add_relation(PositionLimitsCylindrical(center_x=0.0, center_y=0.0, radius_max=0.5))
    """

    name = "position_limits_cylindrical"

    def __init__(
        self,
        center_x: float,
        center_y: float,
        radius_min: float | None = None,
        radius_max: float | None = None,
        relation_loss_weight: float = 1.0,
    ):
        """
        Args:
            center_x: World X of the cylinder's axis in metres.
            center_y: World Y of the cylinder's axis in metres.
            radius_min: Smallest allowed XY distance from the centre in metres, or None for no minimum.
            radius_max: Largest allowed XY distance from the centre in metres, or None for no maximum.
            relation_loss_weight: Weight for the relationship loss function.
        """
        assert (
            radius_min is not None or radius_max is not None
        ), "At least one radial bound must be specified for PositionLimitsCylindrical"
        assert radius_min is None or radius_min >= 0.0, "radius_min must be non-negative"
        assert radius_max is None or radius_max >= 0.0, "radius_max must be non-negative"
        assert (
            radius_min is None or radius_max is None or radius_min < radius_max
        ), "radius_min must be less than radius_max"
        self.center_x = center_x
        self.center_y = center_y
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.relation_loss_weight = relation_loss_weight


def get_anchor_objects(objects: list[PlaceableAsset]) -> list[PlaceableAsset]:
    """Get all anchor objects from a list of objects.

    Anchor objects are marked with IsAnchor() relation and serve as
    fixed reference points for relation solving.

    Args:
        objects: List of objects to filter.

    Returns:
        List of anchor objects (may be empty if no anchors found).
    """
    return [obj for obj in objects if any(isinstance(r, IsAnchor) for r in obj.get_relations())]


def get_relation(obj: PlaceableAsset, relation_type: type[RelationT]) -> RelationT | None:
    """Return obj's first relation of the given type, or None if it has none."""
    return next((relation for relation in obj.get_relations() if isinstance(relation, relation_type)), None)

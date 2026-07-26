# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pure geometry and state-machine guard for the CAP grocery close transition."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from numbers import Real
from typing import TypeAlias

from .gripper_linkage_override import (
    _PROXY_BOX_SPECS,
    _PROXY_CONTACT_OFFSET_M,
    _PROXY_REST_OFFSET_M,
)
from .grocery_bin_collision_override import (
    _BIN_PROXY_BOX_SPECS,
    _BIN_PROXY_CONTACT_OFFSET_M,
    _BIN_PROXY_REST_OFFSET_M,
    _BIN_ROOT_SCALE,
)
from .grocery_object_collision_override import (
    _CAN_PROXY_CONTACT_OFFSET_M,
    _CAN_PROXY_HEIGHT_M,
    _CAN_PROXY_RADIUS_M,
    _CAN_PROXY_REST_OFFSET_M,
)
from .grocery_scene_spec import (
    CAP_GROCERY_GROUND_CONTACT_OFFSET_M,
    CAP_GROCERY_GROUND_REST_OFFSET_M,
    CAP_GROCERY_SUPPORT_CONTACT_OFFSET_M,
    CAP_GROCERY_SUPPORT_REST_OFFSET_M,
    CAP_GROCERY_SUPPORT_SIZE,
)
from .joint_mapping import (
    DROID_GRIPPER_ENDPOINT_TOLERANCE_RAD,
    DROID_GRIPPER_OPEN_POSITION_RAD,
    droid_binary_gripper_action,
)

_Vector3: TypeAlias = tuple[float, float, float]
_QuaternionXyzw: TypeAlias = tuple[float, float, float, float]
_Interval: TypeAlias = tuple[float, float]

_ARM_JOINT_COUNT = 7
_ARM_STATIONARY_BOUND_RAD = 1.0e-3
_ARM_STATIONARY_BOUND_RAD_S = 1.0e-3
_QUATERNION_NORM_TOLERANCE = 1.0e-6
_EXACT_OFFSET_TOLERANCE_M = 1.0e-9
# A cross-product norm squared at or below one representable unit around 1.0
# carries no reliable separating direction. Skipping it is fail-closed because
# the other 14 SAT axes still must prove strict separation.
_SAT_DEGENERATE_CROSS_AXIS_NORM_SQUARED = math.ulp(1.0)


class GroceryCloseAuthorizationError(RuntimeError):
    """A close transition could not be proven safe from the supplied observation."""


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise GroceryCloseAuthorizationError(
            f"{label} must be a real number, not {value!r}"
        )
    result = float(value)
    if not math.isfinite(result):
        raise GroceryCloseAuthorizationError(f"{label} must be finite, got {value!r}")
    return result


def _finite_tuple(
    values: object,
    *,
    length: int,
    label: str,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise GroceryCloseAuthorizationError(
            f"{label} must contain {length} finite numbers"
        )
    try:
        result = tuple(
            _finite_float(value, label=f"{label}[{index}]")
            for index, value in enumerate(values)  # type: ignore[arg-type]
        )
    except TypeError as exc:
        raise GroceryCloseAuthorizationError(
            f"{label} must contain {length} finite numbers"
        ) from exc
    if len(result) != length:
        raise GroceryCloseAuthorizationError(
            f"{label} must contain exactly {length} values, got {len(result)}"
        )
    return result


def _dot(left: _Vector3, right: _Vector3) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _add(left: _Vector3, right: _Vector3) -> _Vector3:
    return tuple(a + b for a, b in zip(left, right, strict=True))  # type: ignore[return-value]


def _subtract(left: _Vector3, right: _Vector3) -> _Vector3:
    return tuple(a - b for a, b in zip(left, right, strict=True))  # type: ignore[return-value]


def _scale(vector: _Vector3, scalar: float) -> _Vector3:
    return tuple(value * scalar for value in vector)  # type: ignore[return-value]


def _cross(left: _Vector3, right: _Vector3) -> _Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _rotation_columns(quaternion_xyzw: _QuaternionXyzw) -> tuple[_Vector3, ...]:
    x, y, z, w = quaternion_xyzw
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return (
        (
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy + wz),
            2.0 * (xz - wy),
        ),
        (
            2.0 * (xy - wz),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz + wx),
        ),
        (
            2.0 * (xz + wy),
            2.0 * (yz - wx),
            1.0 - 2.0 * (xx + yy),
        ),
    )


@dataclass(frozen=True)
class Pose:
    """A finite rigid pose whose quaternion is ordered XYZW."""

    position_m: _Vector3
    orientation_xyzw: _QuaternionXyzw

    def __post_init__(self) -> None:
        position = _finite_tuple(
            self.position_m,
            length=3,
            label="pose.position_m",
        )
        orientation = _finite_tuple(
            self.orientation_xyzw,
            length=4,
            label="pose.orientation_xyzw",
        )
        norm = math.sqrt(sum(value * value for value in orientation))
        if not math.isclose(
            norm,
            1.0,
            rel_tol=0.0,
            abs_tol=_QUATERNION_NORM_TOLERANCE,
        ):
            raise GroceryCloseAuthorizationError(
                f"pose.orientation_xyzw must be normalized; norm={norm!r}"
            )
        orientation = tuple(value / norm for value in orientation)
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "orientation_xyzw", orientation)

    @property
    def rotation_columns(self) -> tuple[_Vector3, ...]:
        """Return local X/Y/Z axes expressed in world coordinates."""
        return _rotation_columns(self.orientation_xyzw)

    def transform_point(self, point_local_m: _Vector3) -> _Vector3:
        """Transform one local point into world coordinates."""
        point = _finite_tuple(point_local_m, length=3, label="point_local_m")
        columns = self.rotation_columns
        rotated = tuple(
            sum(columns[column][row] * point[column] for column in range(3))
            for row in range(3)
        )
        return _add(self.position_m, rotated)  # type: ignore[arg-type]


@dataclass(frozen=True)
class Box:
    """An axis-aligned box in its owning body's local frame."""

    lower_m: _Vector3
    upper_m: _Vector3

    def __post_init__(self) -> None:
        lower = _finite_tuple(self.lower_m, length=3, label="box.lower_m")
        upper = _finite_tuple(self.upper_m, length=3, label="box.upper_m")
        if any(
            lower_value >= upper_value
            for lower_value, upper_value in zip(lower, upper, strict=True)
        ):
            raise GroceryCloseAuthorizationError(
                f"box bounds must be strictly ordered, got {lower!r}..{upper!r}"
            )
        object.__setattr__(self, "lower_m", lower)
        object.__setattr__(self, "upper_m", upper)

    @property
    def center_m(self) -> _Vector3:
        return tuple(
            0.5 * (lower + upper)
            for lower, upper in zip(self.lower_m, self.upper_m, strict=True)
        )  # type: ignore[return-value]

    @property
    def half_extent_m(self) -> _Vector3:
        return tuple(
            0.5 * (upper - lower)
            for lower, upper in zip(self.lower_m, self.upper_m, strict=True)
        )  # type: ignore[return-value]

    def directional_support(
        self,
        pose: Pose,
        direction_world: _Vector3,
    ) -> _Interval:
        """Return the exact projected interval along one unit world direction."""
        direction = _unit_direction(direction_world)
        center_world = pose.transform_point(self.center_m)
        radius = sum(
            abs(_dot(axis_world, direction)) * half_extent
            for axis_world, half_extent in zip(
                pose.rotation_columns,
                self.half_extent_m,
                strict=True,
            )
        )
        center_support = _dot(center_world, direction)
        return center_support - radius, center_support + radius


@dataclass(frozen=True)
class Cylinder:
    """A cylinder aligned with local Z in its owning body's local frame."""

    radius_m: float
    height_m: float

    def __post_init__(self) -> None:
        radius = _finite_float(self.radius_m, label="cylinder.radius_m")
        height = _finite_float(self.height_m, label="cylinder.height_m")
        if radius <= 0.0 or height <= 0.0:
            raise GroceryCloseAuthorizationError(
                "cylinder radius and height must be positive"
            )
        object.__setattr__(self, "radius_m", radius)
        object.__setattr__(self, "height_m", height)

    def directional_support(
        self,
        pose: Pose,
        direction_world: _Vector3,
    ) -> _Interval:
        """Return the exact projected interval along one unit world direction."""
        direction = _unit_direction(direction_world)
        cylinder_axis_world = pose.rotation_columns[2]
        axial_projection = abs(_dot(cylinder_axis_world, direction))
        radial_projection = math.sqrt(
            max(0.0, 1.0 - axial_projection * axial_projection)
        )
        radius = (
            0.5 * self.height_m * axial_projection + self.radius_m * radial_projection
        )
        center_support = _dot(pose.position_m, direction)
        return center_support - radius, center_support + radius


@dataclass(frozen=True)
class CollisionOffsets:
    """Observed PhysX contact/rest offsets for one analytic proxy."""

    contact_m: float
    rest_m: float

    def __post_init__(self) -> None:
        contact = _finite_float(self.contact_m, label="collision contact_m")
        rest = _finite_float(self.rest_m, label="collision rest_m")
        if contact < 0.0 or rest > contact:
            raise GroceryCloseAuthorizationError(
                "collision offsets must satisfy contact_m >= 0 and "
                f"rest_m <= contact_m, got ({contact!r}, {rest!r})"
            )
        object.__setattr__(self, "contact_m", contact)
        object.__setattr__(self, "rest_m", rest)


@dataclass(frozen=True)
class GroceryCollisionOffsets:
    """Observed offsets for every analytic proxy used by the proof."""

    palm: CollisionOffsets | None
    left_finger4: CollisionOffsets | None
    left_fingertip: CollisionOffsets | None
    right_finger4: CollisionOffsets | None
    right_fingertip: CollisionOffsets | None
    can: CollisionOffsets | None
    bin: CollisionOffsets | None
    support: CollisionOffsets | None
    ground: CollisionOffsets | None


@dataclass(frozen=True)
class GroceryCloseObservation:
    """One same-frame physical observation supplied to the close guard."""

    gripper_base_pose: Pose
    left_inner_finger_pose: Pose
    right_inner_finger_pose: Pose
    can_pose: Pose
    bin_pose: Pose
    support_pose: Pose
    driver_position_rad: float
    arm_current_position_rad: tuple[float, ...]
    arm_target_position_rad: tuple[float, ...]
    arm_derived_rate_rad_s: tuple[float, ...]
    collision_offsets: GroceryCollisionOffsets | None

    def __post_init__(self) -> None:
        if not isinstance(self.gripper_base_pose, Pose):
            raise GroceryCloseAuthorizationError("gripper_base_pose must be a Pose")
        if not isinstance(self.left_inner_finger_pose, Pose):
            raise GroceryCloseAuthorizationError(
                "left_inner_finger_pose must be a Pose"
            )
        if not isinstance(self.right_inner_finger_pose, Pose):
            raise GroceryCloseAuthorizationError(
                "right_inner_finger_pose must be a Pose"
            )
        if not isinstance(self.can_pose, Pose):
            raise GroceryCloseAuthorizationError("can_pose must be a Pose")
        if not isinstance(self.bin_pose, Pose):
            raise GroceryCloseAuthorizationError("bin_pose must be a Pose")
        if not isinstance(self.support_pose, Pose):
            raise GroceryCloseAuthorizationError("support_pose must be a Pose")
        driver_position = _finite_float(
            self.driver_position_rad,
            label="driver_position_rad",
        )
        current = _finite_tuple(
            self.arm_current_position_rad,
            length=_ARM_JOINT_COUNT,
            label="arm_current_position_rad",
        )
        target = _finite_tuple(
            self.arm_target_position_rad,
            length=_ARM_JOINT_COUNT,
            label="arm_target_position_rad",
        )
        derived_rate = _finite_tuple(
            self.arm_derived_rate_rad_s,
            length=_ARM_JOINT_COUNT,
            label="arm_derived_rate_rad_s",
        )
        if self.collision_offsets is not None and not isinstance(
            self.collision_offsets, GroceryCollisionOffsets
        ):
            raise GroceryCloseAuthorizationError(
                "collision_offsets must be GroceryCollisionOffsets or None"
            )
        object.__setattr__(self, "driver_position_rad", driver_position)
        object.__setattr__(self, "arm_current_position_rad", current)
        object.__setattr__(self, "arm_target_position_rad", target)
        object.__setattr__(self, "arm_derived_rate_rad_s", derived_rate)


@dataclass(frozen=True)
class GroceryCloseEvidence:
    """Evidence captured when a close target is admitted."""

    newly_latched: bool
    jaw_axis_world: _Vector3
    left_inner_support_m: float
    right_inner_support_m: float
    can_jaw_lower_m: float
    can_jaw_upper_m: float
    left_clearance_m: float
    right_clearance_m: float
    left_x_overlap_m: float
    left_z_overlap_m: float
    right_x_overlap_m: float
    right_z_overlap_m: float
    palm_clearance_m: float
    minimum_fixture_clearance_m: float
    max_arm_target_error_rad: float
    max_arm_derived_rate_rad_s: float


def _unit_direction(direction: object) -> _Vector3:
    vector = _finite_tuple(direction, length=3, label="direction")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0.0:
        raise GroceryCloseAuthorizationError("direction must be nonzero")
    return tuple(value / norm for value in vector)  # type: ignore[return-value]


def _proxy_box(*, body_suffix: str, name: str) -> Box:
    matches = tuple(
        spec
        for spec in _PROXY_BOX_SPECS
        if spec.body_subpath.endswith(body_suffix) and spec.name == name
    )
    if len(matches) != 1:
        raise RuntimeError(
            "CAP grocery collision proxy table does not uniquely define "
            f"{body_suffix}/{name}"
        )
    return Box(lower_m=matches[0].lower_m, upper_m=matches[0].upper_m)


_PALM_BOX = _proxy_box(
    body_suffix="/base_link",
    name="cap_collision_proxy_palm",
)
_LEFT_FINGER_BOXES = (
    _proxy_box(
        body_suffix="/left_inner_finger",
        name="cap_collision_proxy_finger4",
    ),
    _proxy_box(
        body_suffix="/left_inner_finger",
        name="cap_collision_proxy_fingertip",
    ),
)
_RIGHT_FINGER_BOXES = (
    _proxy_box(
        body_suffix="/right_inner_finger",
        name="cap_collision_proxy_finger4",
    ),
    _proxy_box(
        body_suffix="/right_inner_finger",
        name="cap_collision_proxy_fingertip",
    ),
)
_CAN_CYLINDER = Cylinder(
    radius_m=_CAN_PROXY_RADIUS_M,
    height_m=_CAN_PROXY_HEIGHT_M,
)
_BIN_BOXES = tuple(
    Box(
        lower_m=tuple(
            value * _BIN_ROOT_SCALE[index] for index, value in enumerate(spec.lower_raw)
        ),
        upper_m=tuple(
            value * _BIN_ROOT_SCALE[index] for index, value in enumerate(spec.upper_raw)
        ),
    )
    for spec in _BIN_PROXY_BOX_SPECS
)
_SUPPORT_BOX = Box(
    lower_m=tuple(-0.5 * value for value in CAP_GROCERY_SUPPORT_SIZE),
    upper_m=tuple(0.5 * value for value in CAP_GROCERY_SUPPORT_SIZE),
)


def _require_exact_offset(
    actual: CollisionOffsets | None,
    *,
    expected_contact_m: float,
    expected_rest_m: float,
    label: str,
) -> CollisionOffsets:
    if actual is None:
        raise GroceryCloseAuthorizationError(f"{label} collision offsets are missing")
    if not isinstance(actual, CollisionOffsets):
        raise GroceryCloseAuthorizationError(
            f"{label} collision offsets have the wrong type"
        )
    if not (
        math.isclose(
            actual.contact_m,
            expected_contact_m,
            rel_tol=0.0,
            abs_tol=_EXACT_OFFSET_TOLERANCE_M,
        )
        and math.isclose(
            actual.rest_m,
            expected_rest_m,
            rel_tol=0.0,
            abs_tol=_EXACT_OFFSET_TOLERANCE_M,
        )
    ):
        raise GroceryCloseAuthorizationError(
            f"{label} collision offsets drifted: "
            f"expected ({expected_contact_m!r}, {expected_rest_m!r}), "
            f"got ({actual.contact_m!r}, {actual.rest_m!r})"
        )
    return actual


def _require_all_offsets(
    offsets: GroceryCollisionOffsets | None,
) -> GroceryCollisionOffsets:
    if offsets is None:
        raise GroceryCloseAuthorizationError("grocery collision offsets are missing")
    if not isinstance(offsets, GroceryCollisionOffsets):
        raise GroceryCloseAuthorizationError(
            "grocery collision offsets have the wrong type"
        )
    for label, actual in (
        ("palm", offsets.palm),
        ("left finger4", offsets.left_finger4),
        ("left fingertip", offsets.left_fingertip),
        ("right finger4", offsets.right_finger4),
        ("right fingertip", offsets.right_fingertip),
    ):
        _require_exact_offset(
            actual,
            expected_contact_m=_PROXY_CONTACT_OFFSET_M,
            expected_rest_m=_PROXY_REST_OFFSET_M,
            label=label,
        )
    _require_exact_offset(
        offsets.can,
        expected_contact_m=_CAN_PROXY_CONTACT_OFFSET_M,
        expected_rest_m=_CAN_PROXY_REST_OFFSET_M,
        label="can",
    )
    _require_exact_offset(
        offsets.bin,
        expected_contact_m=_BIN_PROXY_CONTACT_OFFSET_M,
        expected_rest_m=_BIN_PROXY_REST_OFFSET_M,
        label="bin",
    )
    _require_exact_offset(
        offsets.support,
        expected_contact_m=CAP_GROCERY_SUPPORT_CONTACT_OFFSET_M,
        expected_rest_m=CAP_GROCERY_SUPPORT_REST_OFFSET_M,
        label="support",
    )
    _require_exact_offset(
        offsets.ground,
        expected_contact_m=CAP_GROCERY_GROUND_CONTACT_OFFSET_M,
        expected_rest_m=CAP_GROCERY_GROUND_REST_OFFSET_M,
        label="ground",
    )
    return offsets


def _require_arm_stationary(
    observation: GroceryCloseObservation,
) -> tuple[float, float]:
    target_error = max(
        abs(target - current)
        for current, target in zip(
            observation.arm_current_position_rad,
            observation.arm_target_position_rad,
            strict=True,
        )
    )
    derived_rate = max(abs(value) for value in observation.arm_derived_rate_rad_s)
    if target_error >= _ARM_STATIONARY_BOUND_RAD:
        raise GroceryCloseAuthorizationError(
            "arm target-current mismatch is not below its close bound: "
            f"max_error_rad={target_error!r}, "
            f"bound_rad={_ARM_STATIONARY_BOUND_RAD!r}"
        )
    if derived_rate >= _ARM_STATIONARY_BOUND_RAD_S:
        raise GroceryCloseAuthorizationError(
            "arm derived rate is not below its close bound: "
            f"max_rate_rad_s={derived_rate!r}, "
            f"bound_rad_s={_ARM_STATIONARY_BOUND_RAD_S!r}"
        )
    return target_error, derived_rate


def _union_interval(intervals: tuple[_Interval, ...]) -> _Interval:
    return min(item[0] for item in intervals), max(item[1] for item in intervals)


def _strict_overlap(first: _Interval, second: _Interval) -> float:
    return min(first[1], second[1]) - max(first[0], second[0])


def _contact_expanded_obb_separation(
    first_box: Box,
    first_pose: Pose,
    first_contact_offset_m: float,
    second_box: Box,
    second_pose: Pose,
    second_contact_offset_m: float,
) -> float | None:
    """Return a strict separating-axis gap, or ``None`` when OBBs touch/overlap."""
    first_contact = _finite_float(
        first_contact_offset_m,
        label="first_contact_offset_m",
    )
    second_contact = _finite_float(
        second_contact_offset_m,
        label="second_contact_offset_m",
    )
    if first_contact < 0.0 or second_contact < 0.0:
        raise GroceryCloseAuthorizationError(
            "contact offsets used by fixture SAT must be nonnegative"
        )

    first_axes = first_pose.rotation_columns
    second_axes = second_pose.rotation_columns
    principal_axes = (*first_axes, *second_axes)
    cross_axes = tuple(
        _cross(first_axis, second_axis)
        for first_axis in first_axes
        for second_axis in second_axes
    )
    center_delta = _subtract(
        second_pose.transform_point(second_box.center_m),
        first_pose.transform_point(first_box.center_m),
    )

    def gap_on_axis(candidate_axis: _Vector3) -> float:
        axis_norm_squared = _dot(candidate_axis, candidate_axis)
        axis = _scale(candidate_axis, 1.0 / math.sqrt(axis_norm_squared))
        first_radius = sum(
            abs(_dot(axis, box_axis)) * half_extent
            for box_axis, half_extent in zip(
                first_axes,
                first_box.half_extent_m,
                strict=True,
            )
        )
        second_radius = sum(
            abs(_dot(axis, box_axis)) * half_extent
            for box_axis, half_extent in zip(
                second_axes,
                second_box.half_extent_m,
                strict=True,
            )
        )
        return (
            abs(_dot(center_delta, axis))
            - first_radius
            - second_radius
            - first_contact
            - second_contact
        )

    best_gap: float | None = None
    for candidate_axis in principal_axes:
        gap = gap_on_axis(candidate_axis)
        if gap > 0.0 and (best_gap is None or gap > best_gap):
            best_gap = gap
    for candidate_axis in cross_axes:
        axis_norm_squared = _dot(candidate_axis, candidate_axis)
        if axis_norm_squared <= _SAT_DEGENERATE_CROSS_AXIS_NORM_SQUARED:
            continue
        gap = gap_on_axis(candidate_axis)
        if gap > 0.0 and (best_gap is None or gap > best_gap):
            best_gap = gap
    return best_gap


def _require_fixture_clearance(
    observation: GroceryCloseObservation,
    offsets: GroceryCollisionOffsets,
) -> float:
    """Prove all five gripper boxes clear the bin, support, and ground."""
    gripper_proxies = (
        ("palm", _PALM_BOX, observation.gripper_base_pose, offsets.palm),
        (
            "left finger4",
            _LEFT_FINGER_BOXES[0],
            observation.left_inner_finger_pose,
            offsets.left_finger4,
        ),
        (
            "left fingertip",
            _LEFT_FINGER_BOXES[1],
            observation.left_inner_finger_pose,
            offsets.left_fingertip,
        ),
        (
            "right finger4",
            _RIGHT_FINGER_BOXES[0],
            observation.right_inner_finger_pose,
            offsets.right_finger4,
        ),
        (
            "right fingertip",
            _RIGHT_FINGER_BOXES[1],
            observation.right_inner_finger_pose,
            offsets.right_fingertip,
        ),
    )
    bin_offset = offsets.bin
    support_offset = offsets.support
    ground_offset = offsets.ground
    assert bin_offset is not None
    assert support_offset is not None
    assert ground_offset is not None
    strict_gaps: list[float] = []
    for label, box, pose, collision_offset in gripper_proxies:
        assert collision_offset is not None
        for bin_index, bin_box in enumerate(_BIN_BOXES):
            gap = _contact_expanded_obb_separation(
                box,
                pose,
                collision_offset.contact_m,
                bin_box,
                observation.bin_pose,
                bin_offset.contact_m,
            )
            if gap is None:
                raise GroceryCloseAuthorizationError(
                    f"{label} is not strictly contact-expanded separated from "
                    f"bin proxy {bin_index}"
                )
            strict_gaps.append(gap)

        support_gap = _contact_expanded_obb_separation(
            box,
            pose,
            collision_offset.contact_m,
            _SUPPORT_BOX,
            observation.support_pose,
            support_offset.contact_m,
        )
        if support_gap is None:
            raise GroceryCloseAuthorizationError(
                f"{label} is not strictly contact-expanded separated from "
                "the procedural support"
            )
        strict_gaps.append(support_gap)

        ground_interval = box.directional_support(pose, (0.0, 0.0, 1.0))
        ground_gap = (
            ground_interval[0] - collision_offset.contact_m - ground_offset.contact_m
        )
        if ground_gap <= 0.0:
            raise GroceryCloseAuthorizationError(
                f"{label} is not strictly contact-expanded separated from "
                f"ground z=0: clearance_m={ground_gap!r}"
            )
        strict_gaps.append(ground_gap)
    return min(strict_gaps)


def prove_initial_grocery_close(
    observation: GroceryCloseObservation,
) -> GroceryCloseEvidence:
    """Prove same-frame geometry, offsets, and stationarity for an initial close."""
    if not isinstance(observation, GroceryCloseObservation):
        raise GroceryCloseAuthorizationError(
            "initial close requires a GroceryCloseObservation"
        )
    max_target_error, max_derived_rate = _require_arm_stationary(observation)
    if not (
        DROID_GRIPPER_OPEN_POSITION_RAD - DROID_GRIPPER_ENDPOINT_TOLERANCE_RAD
        <= observation.driver_position_rad
        <= DROID_GRIPPER_OPEN_POSITION_RAD + DROID_GRIPPER_ENDPOINT_TOLERANCE_RAD
    ):
        raise GroceryCloseAuthorizationError(
            "initial close requires the physical driver in its open band: "
            f"position_rad={observation.driver_position_rad!r}"
        )
    offsets = _require_all_offsets(observation.collision_offsets)
    minimum_fixture_clearance = _require_fixture_clearance(observation, offsets)

    base_x, jaw_axis, base_z = observation.gripper_base_pose.rotation_columns
    can_y = _CAN_CYLINDER.directional_support(observation.can_pose, jaw_axis)
    can_x = _CAN_CYLINDER.directional_support(observation.can_pose, base_x)
    can_z = _CAN_CYLINDER.directional_support(observation.can_pose, base_z)

    left_y_by_box = tuple(
        box.directional_support(
            observation.left_inner_finger_pose,
            jaw_axis,
        )
        for box in _LEFT_FINGER_BOXES
    )
    right_y_by_box = tuple(
        box.directional_support(
            observation.right_inner_finger_pose,
            jaw_axis,
        )
        for box in _RIGHT_FINGER_BOXES
    )
    left_offsets = (offsets.left_finger4, offsets.left_fingertip)
    right_offsets = (offsets.right_finger4, offsets.right_fingertip)
    left_index = min(
        range(len(left_y_by_box)),
        key=lambda index: left_y_by_box[index][0],
    )
    right_index = max(
        range(len(right_y_by_box)),
        key=lambda index: right_y_by_box[index][1],
    )
    left_inner_support = left_y_by_box[left_index][0]
    right_inner_support = right_y_by_box[right_index][1]
    if right_inner_support >= left_inner_support:
        raise GroceryCloseAuthorizationError(
            "left/right finger ordering is invalid along gripper-base +Y: "
            f"right={right_inner_support!r}, left={left_inner_support!r}"
        )

    can_offset = offsets.can
    assert can_offset is not None
    left_offset = left_offsets[left_index]
    right_offset = right_offsets[right_index]
    assert left_offset is not None
    assert right_offset is not None
    left_clearance = (
        left_inner_support - can_y[1] - left_offset.contact_m - can_offset.contact_m
    )
    right_clearance = (
        can_y[0] - right_inner_support - right_offset.contact_m - can_offset.contact_m
    )
    if left_clearance <= 0.0 or right_clearance <= 0.0:
        raise GroceryCloseAuthorizationError(
            "can is not strictly clear of both contact-expanded inner fingers: "
            f"left_clearance_m={left_clearance!r}, "
            f"right_clearance_m={right_clearance!r}"
        )

    left_x = _union_interval(
        tuple(
            box.directional_support(
                observation.left_inner_finger_pose,
                base_x,
            )
            for box in _LEFT_FINGER_BOXES
        )
    )
    left_z = _union_interval(
        tuple(
            box.directional_support(
                observation.left_inner_finger_pose,
                base_z,
            )
            for box in _LEFT_FINGER_BOXES
        )
    )
    right_x = _union_interval(
        tuple(
            box.directional_support(
                observation.right_inner_finger_pose,
                base_x,
            )
            for box in _RIGHT_FINGER_BOXES
        )
    )
    right_z = _union_interval(
        tuple(
            box.directional_support(
                observation.right_inner_finger_pose,
                base_z,
            )
            for box in _RIGHT_FINGER_BOXES
        )
    )
    overlaps = (
        _strict_overlap(can_x, left_x),
        _strict_overlap(can_z, left_z),
        _strict_overlap(can_x, right_x),
        _strict_overlap(can_z, right_z),
    )
    if any(overlap <= 0.0 for overlap in overlaps):
        raise GroceryCloseAuthorizationError(
            "can does not overlap both finger working envelopes in gripper-base "
            f"X/Z: left_x={overlaps[0]!r}, left_z={overlaps[1]!r}, "
            f"right_x={overlaps[2]!r}, right_z={overlaps[3]!r}"
        )

    palm_x = _PALM_BOX.directional_support(
        observation.gripper_base_pose,
        base_x,
    )
    palm_offset = offsets.palm
    assert palm_offset is not None
    palm_clearance = can_x[0] - palm_x[1] - palm_offset.contact_m - can_offset.contact_m
    if palm_clearance <= 0.0:
        raise GroceryCloseAuthorizationError(
            "can is not strictly separated from the contact-expanded palm "
            f"along gripper-base +X: clearance_m={palm_clearance!r}"
        )

    return GroceryCloseEvidence(
        newly_latched=True,
        jaw_axis_world=jaw_axis,
        left_inner_support_m=left_inner_support,
        right_inner_support_m=right_inner_support,
        can_jaw_lower_m=can_y[0],
        can_jaw_upper_m=can_y[1],
        left_clearance_m=left_clearance,
        right_clearance_m=right_clearance,
        left_x_overlap_m=overlaps[0],
        left_z_overlap_m=overlaps[1],
        right_x_overlap_m=overlaps[2],
        right_z_overlap_m=overlaps[3],
        palm_clearance_m=palm_clearance,
        minimum_fixture_clearance_m=minimum_fixture_clearance,
        max_arm_target_error_rad=max_target_error,
        max_arm_derived_rate_rad_s=max_derived_rate,
    )


class GroceryCloseGuard:
    """Latch a proven initial close while rechecking arm stationarity each frame."""

    def __init__(self) -> None:
        self._latched_evidence: GroceryCloseEvidence | None = None

    @property
    def close_authorized(self) -> bool:
        """Whether the current contiguous target run has an initial-close proof."""
        return self._latched_evidence is not None

    def reset(self) -> None:
        """Clear authorization at an episode/generation reset."""
        self._latched_evidence = None

    def evaluate_target(
        self,
        target_position_rad: float,
        observation: GroceryCloseObservation | None = None,
    ) -> GroceryCloseEvidence | None:
        """Evaluate one binary endpoint target and update the close latch."""
        target = _finite_float(
            target_position_rad,
            label="target_position_rad",
        )
        try:
            closedness = droid_binary_gripper_action(target)
        except ValueError as exc:
            raise GroceryCloseAuthorizationError(str(exc)) from exc

        if closedness == 0.0:
            self._latched_evidence = None
            return None
        if closedness != 1.0:
            raise GroceryCloseAuthorizationError(
                f"unsupported gripper closedness {closedness!r}"
            )
        if observation is None:
            raise GroceryCloseAuthorizationError(
                "close target requires a same-frame physical observation"
            )
        if not isinstance(observation, GroceryCloseObservation):
            raise GroceryCloseAuthorizationError(
                "close target observation has the wrong type"
            )

        max_target_error, max_derived_rate = _require_arm_stationary(observation)
        if self._latched_evidence is None:
            evidence = prove_initial_grocery_close(observation)
        else:
            evidence = replace(
                self._latched_evidence,
                newly_latched=False,
                max_arm_target_error_rad=max_target_error,
                max_arm_derived_rate_rad_s=max_derived_rate,
            )
        self._latched_evidence = evidence
        return evidence

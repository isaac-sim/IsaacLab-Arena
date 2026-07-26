# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from isaaclab_arena.integrations.cap_barrier.grocery_close_guard import (
    Box,
    CollisionOffsets,
    Cylinder,
    GroceryCloseAuthorizationError,
    GroceryCloseGuard,
    GroceryCloseObservation,
    GroceryCollisionOffsets,
    Pose,
    _contact_expanded_obb_separation,
    _require_fixture_clearance,
    prove_initial_grocery_close,
)
from isaaclab_arena.integrations.cap_barrier.grocery_bin_collision_override import (
    _BIN_PROXY_CONTACT_OFFSET_M,
    _BIN_PROXY_REST_OFFSET_M,
)
from isaaclab_arena.integrations.cap_barrier.grocery_object_collision_override import (
    _CAN_PROXY_CONTACT_OFFSET_M,
    _CAN_PROXY_REST_OFFSET_M,
)
from isaaclab_arena.integrations.cap_barrier.grocery_scene_spec import (
    CAP_GROCERY_BIN_POSE,
    CAP_GROCERY_GROUND_CONTACT_OFFSET_M,
    CAP_GROCERY_GROUND_REST_OFFSET_M,
    CAP_GROCERY_SUPPORT_CONTACT_OFFSET_M,
    CAP_GROCERY_SUPPORT_POSE,
    CAP_GROCERY_SUPPORT_REST_OFFSET_M,
)
from isaaclab_arena.integrations.cap_barrier.gripper_linkage_override import (
    _PROXY_CONTACT_OFFSET_M,
    _PROXY_REST_OFFSET_M,
)
from isaaclab_arena.integrations.cap_barrier.joint_mapping import (
    DROID_GRIPPER_CLOSED_POSITION_RAD,
    DROID_GRIPPER_OPEN_POSITION_RAD,
)

_IDENTITY = (0.0, 0.0, 0.0, 1.0)


def _pose(
    position_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation_xyzw: tuple[float, float, float, float] = _IDENTITY,
) -> Pose:
    return Pose(position_m=position_m, orientation_xyzw=orientation_xyzw)


def _offsets() -> GroceryCollisionOffsets:
    proxy = CollisionOffsets(
        contact_m=_PROXY_CONTACT_OFFSET_M,
        rest_m=_PROXY_REST_OFFSET_M,
    )
    can = CollisionOffsets(
        contact_m=_CAN_PROXY_CONTACT_OFFSET_M,
        rest_m=_CAN_PROXY_REST_OFFSET_M,
    )
    return GroceryCollisionOffsets(
        palm=proxy,
        left_finger4=proxy,
        left_fingertip=proxy,
        right_finger4=proxy,
        right_fingertip=proxy,
        can=can,
        bin=CollisionOffsets(
            contact_m=_BIN_PROXY_CONTACT_OFFSET_M,
            rest_m=_BIN_PROXY_REST_OFFSET_M,
        ),
        support=CollisionOffsets(
            contact_m=CAP_GROCERY_SUPPORT_CONTACT_OFFSET_M,
            rest_m=CAP_GROCERY_SUPPORT_REST_OFFSET_M,
        ),
        ground=CollisionOffsets(
            contact_m=CAP_GROCERY_GROUND_CONTACT_OFFSET_M,
            rest_m=CAP_GROCERY_GROUND_REST_OFFSET_M,
        ),
    )


def _observation(**changes: object) -> GroceryCloseObservation:
    values: dict[str, object] = {
        "gripper_base_pose": _pose((0.0, 0.0, 0.3)),
        "left_inner_finger_pose": _pose((0.0, 0.0, 0.3)),
        "right_inner_finger_pose": _pose((0.0, 0.0, 0.3)),
        "can_pose": _pose((0.13, 0.0, 0.3)),
        "bin_pose": _pose(*CAP_GROCERY_BIN_POSE),
        "support_pose": _pose(*CAP_GROCERY_SUPPORT_POSE),
        "driver_position_rad": DROID_GRIPPER_OPEN_POSITION_RAD,
        "arm_current_position_rad": (0.0,) * 7,
        "arm_target_position_rad": (0.0,) * 7,
        "arm_derived_rate_rad_s": (0.0,) * 7,
        "collision_offsets": _offsets(),
    }
    values.update(changes)
    return GroceryCloseObservation(**values)  # type: ignore[arg-type]


def _transform(
    local_position_m: tuple[float, float, float],
    *,
    yaw_rad: float,
    translation_m: tuple[float, float, float],
) -> tuple[float, float, float]:
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    x, y, z = local_position_m
    tx, ty, tz = translation_m
    return (
        tx + cosine * x - sine * y,
        ty + sine * x + cosine * y,
        tz + z,
    )


def test_initial_close_proves_exact_geometry_and_stationarity() -> None:
    evidence = prove_initial_grocery_close(_observation())

    assert evidence.newly_latched
    assert evidence.jaw_axis_world == pytest.approx((0.0, 1.0, 0.0))
    assert evidence.left_clearance_m > 0.006
    assert evidence.right_clearance_m > 0.006
    assert evidence.left_x_overlap_m > 0.0
    assert evidence.left_z_overlap_m > 0.0
    assert evidence.right_x_overlap_m > 0.0
    assert evidence.right_z_overlap_m > 0.0
    assert evidence.palm_clearance_m > 0.003
    assert evidence.minimum_fixture_clearance_m > 0.0
    assert evidence.max_arm_target_error_rad == 0.0
    assert evidence.max_arm_derived_rate_rad_s == 0.0


@pytest.mark.parametrize("can_y_m", (-0.007, 0.007))
def test_initial_close_rejects_a_can_without_both_side_clearances(
    can_y_m: float,
) -> None:
    observation = _observation(can_pose=_pose((0.13, can_y_m, 0.3)))

    with pytest.raises(
        GroceryCloseAuthorizationError,
        match="strictly clear of both.*inner fingers",
    ):
        prove_initial_grocery_close(observation)


def test_initial_close_rejects_can_outside_finger_working_envelopes() -> None:
    observation = _observation(can_pose=_pose((0.30, 0.0, 0.3)))

    with pytest.raises(
        GroceryCloseAuthorizationError,
        match="does not overlap both finger working envelopes",
    ):
        prove_initial_grocery_close(observation)


def test_initial_close_rejects_contact_expanded_palm_overlap() -> None:
    observation = _observation(can_pose=_pose((0.124, 0.0, 0.3)))

    with pytest.raises(
        GroceryCloseAuthorizationError,
        match="not strictly separated.*palm",
    ):
        prove_initial_grocery_close(observation)


def test_initial_close_rejects_reversed_left_right_fingers() -> None:
    observation = _observation(
        left_inner_finger_pose=_pose((0.0, -0.1, 0.3)),
        right_inner_finger_pose=_pose((0.0, 0.1, 0.3)),
    )

    with pytest.raises(
        GroceryCloseAuthorizationError,
        match="left/right finger ordering is invalid",
    ):
        prove_initial_grocery_close(observation)


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (
            lambda: Pose((math.nan, 0.0, 0.0), _IDENTITY),
            "position_m.*finite",
        ),
        (
            lambda: Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)),
            "must be normalized",
        ),
        (
            lambda: Pose((False, 0.0, 0.0), _IDENTITY),
            "must be a real number",
        ),
        (
            lambda: Box((0.0, 0.0, 0.0), (1.0, 0.0, 1.0)),
            "strictly ordered",
        ),
        (
            lambda: Cylinder(True, 1.0),
            "must be a real number",
        ),
    ],
)
def test_geometry_types_reject_invalid_or_boolean_numerics(
    factory: object,
    match: str,
) -> None:
    with pytest.raises(GroceryCloseAuthorizationError, match=match):
        factory()  # type: ignore[operator]


def test_obb_sat_requires_strict_clearance_after_both_contact_skins() -> None:
    unit_box = Box((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))
    first_pose = _pose()

    assert (
        _contact_expanded_obb_separation(
            unit_box,
            first_pose,
            0.125,
            unit_box,
            _pose((1.25, 0.0, 0.0)),
            0.125,
        )
        is None
    )
    assert _contact_expanded_obb_separation(
        unit_box,
        first_pose,
        0.125,
        unit_box,
        _pose((1.251, 0.0, 0.0)),
        0.125,
    ) == pytest.approx(0.001)


def test_obb_sat_normalizes_an_accepted_near_unit_quaternion() -> None:
    unit_box = Box((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))
    yaw_rad = math.pi / 6.0
    exact_orientation = (
        0.0,
        0.0,
        math.sin(0.5 * yaw_rad),
        math.cos(0.5 * yaw_rad),
    )
    scale = 0.99999901
    near_unit_orientation = tuple(value * scale for value in exact_orientation)
    direction = (math.cos(yaw_rad), math.sin(yaw_rad), 0.0)
    center_distance_m = 1.0 - 1.0e-7
    second_position = tuple(center_distance_m * value for value in direction)

    first_pose = _pose(orientation_xyzw=near_unit_orientation)
    second_pose = _pose(
        second_position,
        near_unit_orientation,
    )

    assert first_pose.orientation_xyzw != near_unit_orientation
    assert first_pose.orientation_xyzw == pytest.approx(
        exact_orientation,
        rel=0.0,
        abs=math.ulp(1.0),
    )
    assert (
        _contact_expanded_obb_separation(
            unit_box,
            first_pose,
            0.0,
            unit_box,
            second_pose,
            0.0,
        )
        is None
    )


def test_initial_close_rejects_missing_or_drifted_offsets() -> None:
    missing = replace(_offsets(), left_fingertip=None)
    with pytest.raises(
        GroceryCloseAuthorizationError,
        match="left fingertip collision offsets are missing",
    ):
        prove_initial_grocery_close(_observation(collision_offsets=missing))

    drifted = replace(
        _offsets(),
        can=CollisionOffsets(
            contact_m=_CAN_PROXY_CONTACT_OFFSET_M + 1.0e-4,
            rest_m=_CAN_PROXY_REST_OFFSET_M,
        ),
    )
    with pytest.raises(
        GroceryCloseAuthorizationError,
        match="can collision offsets drifted",
    ):
        prove_initial_grocery_close(_observation(collision_offsets=drifted))

    for field, match in (
        ("bin", "bin collision offsets are missing"),
        ("support", "support collision offsets are missing"),
        ("ground", "ground collision offsets are missing"),
    ):
        missing_fixture = replace(_offsets(), **{field: None})
        with pytest.raises(
            GroceryCloseAuthorizationError,
            match=match,
        ):
            prove_initial_grocery_close(_observation(collision_offsets=missing_fixture))

    with pytest.raises(
        GroceryCloseAuthorizationError,
        match="rest_m <= contact_m",
    ):
        CollisionOffsets(contact_m=0.0, rest_m=0.001)


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        (
            {"arm_target_position_rad": (0.001,) + (0.0,) * 6},
            "target-current mismatch",
        ),
        (
            {"arm_derived_rate_rad_s": (0.0,) * 6 + (0.001,)},
            "derived rate",
        ),
        (
            {"arm_current_position_rad": (0.0,) * 6},
            "exactly 7",
        ),
        (
            {"arm_derived_rate_rad_s": (False,) + (0.0,) * 6},
            "must be a real number",
        ),
    ],
)
def test_initial_close_requires_all_seven_arm_joints_stationary(
    changes: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(GroceryCloseAuthorizationError, match=match):
        observation = _observation(**changes)
        prove_initial_grocery_close(observation)


def test_initial_close_requires_driver_open_proof() -> None:
    observation = _observation(driver_position_rad=0.02)

    with pytest.raises(
        GroceryCloseAuthorizationError,
        match="physical driver in its open band",
    ):
        prove_initial_grocery_close(observation)


def test_guard_latches_close_rechecks_arm_and_clears_on_open_and_reset() -> None:
    guard = GroceryCloseGuard()
    initial = guard.evaluate_target(
        DROID_GRIPPER_CLOSED_POSITION_RAD,
        _observation(),
    )

    assert initial is not None
    assert initial.newly_latched
    assert guard.close_authorized

    contact_geometry = _observation(
        driver_position_rad=0.4,
        can_pose=_pose((0.13, 0.02, 0.3)),
        arm_target_position_rad=(0.0002,) + (0.0,) * 6,
        arm_derived_rate_rad_s=(0.0003,) + (0.0,) * 6,
    )
    repeated = guard.evaluate_target(
        DROID_GRIPPER_CLOSED_POSITION_RAD,
        contact_geometry,
    )
    assert repeated is not None
    assert not repeated.newly_latched
    assert repeated.left_clearance_m == initial.left_clearance_m
    assert repeated.max_arm_target_error_rad == pytest.approx(0.0002)
    assert repeated.max_arm_derived_rate_rad_s == pytest.approx(0.0003)

    moving = replace(
        contact_geometry,
        arm_derived_rate_rad_s=(0.002,) + (0.0,) * 6,
    )
    with pytest.raises(GroceryCloseAuthorizationError, match="derived rate"):
        guard.evaluate_target(DROID_GRIPPER_CLOSED_POSITION_RAD, moving)
    assert guard.close_authorized

    assert (
        guard.evaluate_target(DROID_GRIPPER_OPEN_POSITION_RAD, observation=None) is None
    )
    assert not guard.close_authorized
    with pytest.raises(
        GroceryCloseAuthorizationError,
        match="physical driver in its open band",
    ):
        guard.evaluate_target(
            DROID_GRIPPER_CLOSED_POSITION_RAD,
            contact_geometry,
        )

    guard.evaluate_target(DROID_GRIPPER_CLOSED_POSITION_RAD, _observation())
    guard.reset()
    assert not guard.close_authorized
    with pytest.raises(
        GroceryCloseAuthorizationError,
        match="physical driver in its open band",
    ):
        guard.evaluate_target(
            DROID_GRIPPER_CLOSED_POSITION_RAD,
            contact_geometry,
        )


def test_guard_rejects_missing_observation_intermediate_and_boolean_targets() -> None:
    guard = GroceryCloseGuard()
    with pytest.raises(
        GroceryCloseAuthorizationError,
        match="same-frame physical observation",
    ):
        guard.evaluate_target(DROID_GRIPPER_CLOSED_POSITION_RAD)
    with pytest.raises(
        GroceryCloseAuthorizationError,
        match="outside the supported endpoint bands",
    ):
        guard.evaluate_target(0.4, _observation())
    with pytest.raises(
        GroceryCloseAuthorizationError,
        match="must be a real number",
    ):
        guard.evaluate_target(True, _observation())


def test_fixture_proof_is_invariant_when_every_fixture_is_transformed() -> None:
    yaw = math.pi / 2.0
    translation = (1.2, -0.7, 0.0)
    quaternion = (0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))
    base = _observation()
    base_clearance = _require_fixture_clearance(base, _offsets())

    def transformed_pose(pose: Pose) -> Pose:
        return _pose(
            _transform(
                pose.position_m,
                yaw_rad=yaw,
                translation_m=translation,
            ),
            quaternion,
        )

    can_pose = _pose(
        _transform(
            base.can_pose.position_m,
            yaw_rad=yaw,
            translation_m=translation,
        ),
        quaternion,
    )
    observation = _observation(
        gripper_base_pose=transformed_pose(base.gripper_base_pose),
        left_inner_finger_pose=transformed_pose(base.left_inner_finger_pose),
        right_inner_finger_pose=transformed_pose(base.right_inner_finger_pose),
        can_pose=can_pose,
        bin_pose=transformed_pose(base.bin_pose),
        support_pose=transformed_pose(base.support_pose),
    )

    transformed_clearance = _require_fixture_clearance(observation, _offsets())

    assert transformed_clearance == pytest.approx(base_clearance)


def _translated_grasp(
    translation_m: tuple[float, float, float],
) -> GroceryCloseObservation:
    tx, ty, tz = translation_m
    return _observation(
        gripper_base_pose=_pose(translation_m),
        left_inner_finger_pose=_pose(translation_m),
        right_inner_finger_pose=_pose(translation_m),
        can_pose=_pose((tx + 0.13, ty, tz)),
    )


def test_initial_close_rejects_gripper_inside_bin_fixture() -> None:
    bin_x, bin_y, _ = CAP_GROCERY_BIN_POSE[0]
    observation = _translated_grasp((bin_x, bin_y, 0.08))

    with pytest.raises(
        GroceryCloseAuthorizationError,
        match="not strictly contact-expanded separated from bin proxy",
    ):
        prove_initial_grocery_close(observation)


def test_initial_close_rejects_gripper_inside_support_fixture() -> None:
    support_x, support_y, _ = CAP_GROCERY_SUPPORT_POSE[0]
    observation = _translated_grasp((support_x, support_y + 0.3, 0.04))

    with pytest.raises(
        GroceryCloseAuthorizationError,
        match="not strictly contact-expanded separated from the procedural support",
    ):
        prove_initial_grocery_close(observation)


def test_initial_close_rejects_gripper_at_ground_contact_skin() -> None:
    observation = _translated_grasp((0.0, 0.8, 0.0376))

    with pytest.raises(
        GroceryCloseAuthorizationError,
        match="not strictly contact-expanded separated from ground z=0",
    ):
        prove_initial_grocery_close(observation)

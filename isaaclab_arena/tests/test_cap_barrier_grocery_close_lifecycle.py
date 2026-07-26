# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import math

import pytest

from isaaclab_arena.integrations.cap_barrier.grocery_close_lifecycle import (
    CLOSURE_SETTLED_RATE_BOUND_RAD_S,
    DROID_DRIVER_RATE_BOUND_RAD_S,
    PINNED_SIMULATION_DT_S,
    PINNED_SIMULATION_DT_TOLERANCE_S,
    GroceryCloseLifecycle,
    GroceryCloseLifecycleError,
    GroceryCloseLifecycleSample,
    GroceryCloseLifecycleState,
    GroceryGripperTarget,
)
from isaaclab_arena.integrations.cap_barrier.joint_mapping import (
    DROID_GRIPPER_CLOSED_POSITION_RAD,
    DROID_GRIPPER_ENDPOINT_TOLERANCE_RAD,
    DROID_GRIPPER_OPEN_POSITION_RAD,
    DROID_PHYSICAL_GRIPPER_JOINTS,
)


def _sample(
    sequence: int,
    position_rad: float,
    target: GroceryGripperTarget = GroceryGripperTarget.CLOSE,
    *,
    timestamp_s: float | None = None,
    physical_sample_safe: bool = True,
    gripper_joint_positions_rad: tuple[float, ...] | None = None,
) -> GroceryCloseLifecycleSample:
    return GroceryCloseLifecycleSample(
        sequence=sequence,
        simulation_timestamp_s=(
            sequence * PINNED_SIMULATION_DT_S if timestamp_s is None else timestamp_s
        ),
        gripper_joint_positions_rad=(
            (position_rad,) * len(DROID_PHYSICAL_GRIPPER_JOINTS)
            if gripper_joint_positions_rad is None
            else gripper_joint_positions_rad
        ),
        requested_target=target,
        physical_sample_safe=physical_sample_safe,
    )


def _begin_close(
    lifecycle: GroceryCloseLifecycle,
    *,
    open_position_rad: float = DROID_GRIPPER_OPEN_POSITION_RAD,
) -> tuple[int, float]:
    lifecycle.observe_pre_step(_sample(0, open_position_rad, GroceryGripperTarget.OPEN))
    position_rad = open_position_rad + 0.004
    evidence = lifecycle.observe_pre_step(_sample(1, position_rad))
    assert evidence.state is GroceryCloseLifecycleState.CLOSE_TRANSITION
    return 1, position_rad


def _drive_to_closed_hold(
    lifecycle: GroceryCloseLifecycle,
    *,
    terminal_position_rad: float = DROID_GRIPPER_CLOSED_POSITION_RAD,
) -> tuple[int, GroceryCloseLifecycleSample]:
    sequence, position_rad = _begin_close(lifecycle)
    while position_rad < terminal_position_rad:
        sequence += 1
        position_rad = min(position_rad + 0.004, terminal_position_rad)
        lifecycle.observe_pre_step(_sample(sequence, position_rad))
    sequence += 1
    first_settled = lifecycle.observe_pre_step(_sample(sequence, position_rad))
    assert first_settled.closure_settled_pair_count == 1
    assert not first_settled.closure_settled
    sequence += 1
    settled_sample = _sample(sequence, position_rad)
    settled = lifecycle.observe_pre_step(settled_sample)
    assert settled.closure_settled
    assert settled.newly_closure_settled
    return sequence, settled_sample


def test_constants_pin_the_droid_timing_and_rate_contract() -> None:
    assert PINNED_SIMULATION_DT_S == 0.005
    assert PINNED_SIMULATION_DT_TOLERANCE_S == 1.0e-9
    assert CLOSURE_SETTLED_RATE_BOUND_RAD_S == 1.0e-3
    assert DROID_DRIVER_RATE_BOUND_RAD_S == 1.0
    assert len(DROID_PHYSICAL_GRIPPER_JOINTS) == 6


def test_open_target_clears_without_using_geometry_or_driver_rate() -> None:
    lifecycle = GroceryCloseLifecycle()
    _begin_close(lifecycle)

    evidence = lifecycle.observe_pre_step(
        _sample(
            10,
            0.4,
            GroceryGripperTarget.OPEN,
            timestamp_s=-2.0,
            physical_sample_safe=False,
        )
    )

    assert evidence.state is GroceryCloseLifecycleState.OPEN
    assert evidence.gripper_joint_positions_rad == (0.4,) * 6
    assert evidence.derived_gripper_joint_rates_rad_s is None
    assert evidence.max_abs_derived_gripper_joint_rate_rad_s is None
    assert evidence.derived_driver_rate_rad_s is None
    assert evidence.closure_settled_pair_count == 0
    assert not lifecycle.closure_settled


def test_initial_close_requires_prior_adjacent_physical_open_sample() -> None:
    lifecycle = GroceryCloseLifecycle()
    with pytest.raises(
        GroceryCloseLifecycleError,
        match="prior sequence-adjacent physically-open",
    ):
        lifecycle.observe_pre_step(_sample(0, DROID_GRIPPER_OPEN_POSITION_RAD))

    assert lifecycle.state is GroceryCloseLifecycleState.OPEN
    lifecycle.observe_pre_step(_sample(3, 0.02, GroceryGripperTarget.OPEN))
    with pytest.raises(GroceryCloseLifecycleError, match="open endpoint band"):
        lifecycle.observe_pre_step(_sample(4, 0.021))
    assert lifecycle.state is GroceryCloseLifecycleState.OPEN


@pytest.mark.parametrize(
    "open_position_rad",
    (
        DROID_GRIPPER_OPEN_POSITION_RAD - DROID_GRIPPER_ENDPOINT_TOLERANCE_RAD,
        DROID_GRIPPER_OPEN_POSITION_RAD + DROID_GRIPPER_ENDPOINT_TOLERANCE_RAD,
    ),
)
def test_initial_close_accepts_both_exact_open_band_boundaries(
    open_position_rad: float,
) -> None:
    lifecycle = GroceryCloseLifecycle()
    sequence, position_rad = _begin_close(
        lifecycle,
        open_position_rad=open_position_rad,
    )

    assert sequence == 1
    assert position_rad == pytest.approx(open_position_rad + 0.004)
    assert lifecycle.state is GroceryCloseLifecycleState.CLOSE_TRANSITION


def test_initial_and_continuing_close_require_physical_sample_safe() -> None:
    lifecycle = GroceryCloseLifecycle()
    lifecycle.observe_pre_step(_sample(0, 0.0, GroceryGripperTarget.OPEN))
    with pytest.raises(GroceryCloseLifecycleError, match="geometry safety proof"):
        lifecycle.observe_pre_step(_sample(1, 0.004, physical_sample_safe=False))
    assert lifecycle.state is GroceryCloseLifecycleState.OPEN

    sequence, position_rad = _begin_close(lifecycle)
    with pytest.raises(GroceryCloseLifecycleError, match="geometry safety proof"):
        lifecycle.observe_pre_step(
            _sample(
                sequence + 1,
                position_rad + 0.004,
                physical_sample_safe=False,
            )
        )
    assert lifecycle.state is GroceryCloseLifecycleState.OPEN


@pytest.mark.parametrize(
    ("sequence", "timestamp_s", "match"),
    [
        (2, PINNED_SIMULATION_DT_S, "strictly increase"),
        (1, 2 * PINNED_SIMULATION_DT_S, "strictly increase"),
        (4, 4 * PINNED_SIMULATION_DT_S, "sequence-adjacent"),
        (3, 2 * PINNED_SIMULATION_DT_S, "positive simulation timestamp delta"),
        (3, PINNED_SIMULATION_DT_S, "positive simulation timestamp delta"),
        (
            3,
            3 * PINNED_SIMULATION_DT_S + 2 * PINNED_SIMULATION_DT_TOLERANCE_S,
            "sample delta drifted",
        ),
    ],
)
def test_close_rejects_skipped_or_nonmonotonic_sample_pairs(
    sequence: int,
    timestamp_s: float,
    match: str,
) -> None:
    lifecycle = GroceryCloseLifecycle()
    last_sequence, position_rad = _begin_close(lifecycle)
    assert last_sequence == 1
    lifecycle.observe_pre_step(_sample(2, position_rad + 0.004))

    with pytest.raises(GroceryCloseLifecycleError, match=match):
        lifecycle.observe_pre_step(
            _sample(
                sequence,
                position_rad + 0.008,
                timestamp_s=timestamp_s,
            )
        )
    assert lifecycle.state is GroceryCloseLifecycleState.OPEN


@pytest.mark.parametrize(
    "delta_s",
    (
        PINNED_SIMULATION_DT_S - PINNED_SIMULATION_DT_TOLERANCE_S,
        PINNED_SIMULATION_DT_S,
        PINNED_SIMULATION_DT_S + PINNED_SIMULATION_DT_TOLERANCE_S,
    ),
)
def test_close_accepts_exact_timestamp_tolerance_boundaries(
    delta_s: float,
) -> None:
    lifecycle = GroceryCloseLifecycle()
    lifecycle.observe_pre_step(_sample(0, 0.0, GroceryGripperTarget.OPEN))
    evidence = lifecycle.observe_pre_step(
        _sample(
            1,
            0.004,
            timestamp_s=delta_s,
        )
    )

    assert evidence.derived_driver_rate_rad_s == pytest.approx(0.004 / delta_s)


def test_close_requires_monotonic_bounded_driver_motion() -> None:
    lifecycle = GroceryCloseLifecycle()
    sequence, position_rad = _begin_close(lifecycle)
    with pytest.raises(GroceryCloseLifecycleError, match="reversed"):
        lifecycle.observe_pre_step(_sample(sequence + 1, position_rad - 0.001))
    assert lifecycle.state is GroceryCloseLifecycleState.OPEN

    lifecycle.observe_pre_step(_sample(10, 0.0, GroceryGripperTarget.OPEN))
    with pytest.raises(GroceryCloseLifecycleError, match="actuator bound"):
        lifecycle.observe_pre_step(
            _sample(
                11,
                (DROID_DRIVER_RATE_BOUND_RAD_S + 0.01) * PINNED_SIMULATION_DT_S,
            )
        )
    assert lifecycle.state is GroceryCloseLifecycleState.OPEN


def test_close_accepts_the_exact_pinned_driver_rate_bound() -> None:
    lifecycle = GroceryCloseLifecycle()
    lifecycle.observe_pre_step(_sample(0, 0.0, GroceryGripperTarget.OPEN))

    evidence = lifecycle.observe_pre_step(
        _sample(
            1,
            DROID_DRIVER_RATE_BOUND_RAD_S * PINNED_SIMULATION_DT_S,
        )
    )

    assert evidence.derived_driver_rate_rad_s == pytest.approx(
        DROID_DRIVER_RATE_BOUND_RAD_S
    )
    assert evidence.state is GroceryCloseLifecycleState.CLOSE_TRANSITION


def test_transition_checks_direction_and_rate_only_on_the_driver() -> None:
    lifecycle = GroceryCloseLifecycle()
    lifecycle.observe_pre_step(
        _sample(
            0,
            0.0,
            GroceryGripperTarget.OPEN,
            gripper_joint_positions_rad=(0.0,) * 6,
        )
    )
    positions = (0.004, -0.004, 0.003, -0.002, 0.001, -0.001)
    evidence = lifecycle.observe_pre_step(
        _sample(
            1,
            positions[0],
            gripper_joint_positions_rad=positions,
        )
    )

    assert evidence.state is GroceryCloseLifecycleState.CLOSE_TRANSITION
    assert evidence.gripper_joint_positions_rad == positions
    assert evidence.derived_gripper_joint_rates_rad_s == pytest.approx(
        (0.8, -0.8, 0.6, -0.4, 0.2, -0.2)
    )
    assert evidence.max_abs_derived_gripper_joint_rate_rad_s == pytest.approx(0.8)
    assert evidence.derived_driver_rate_rad_s == pytest.approx(0.8)


@pytest.mark.parametrize(
    "position_rad",
    (
        DROID_GRIPPER_OPEN_POSITION_RAD - DROID_GRIPPER_ENDPOINT_TOLERANCE_RAD - 1.0e-9,
        DROID_GRIPPER_CLOSED_POSITION_RAD
        + DROID_GRIPPER_ENDPOINT_TOLERANCE_RAD
        + 1.0e-9,
    ),
)
def test_transition_rejects_positions_outside_the_endpoint_envelope(
    position_rad: float,
) -> None:
    lifecycle = GroceryCloseLifecycle()
    lifecycle.observe_pre_step(_sample(0, 0.0, GroceryGripperTarget.OPEN))
    with pytest.raises(GroceryCloseLifecycleError, match="endpoint envelope"):
        lifecycle.observe_pre_step(_sample(1, position_rad))
    assert lifecycle.state is GroceryCloseLifecycleState.OPEN


@pytest.mark.parametrize(
    "terminal_position_rad",
    (
        DROID_GRIPPER_CLOSED_POSITION_RAD - DROID_GRIPPER_ENDPOINT_TOLERANCE_RAD,
        DROID_GRIPPER_CLOSED_POSITION_RAD,
        DROID_GRIPPER_CLOSED_POSITION_RAD + DROID_GRIPPER_ENDPOINT_TOLERANCE_RAD,
    ),
)
def test_two_consecutive_strictly_settled_pairs_enter_closed_hold(
    terminal_position_rad: float,
) -> None:
    lifecycle = GroceryCloseLifecycle()
    sequence, _ = _drive_to_closed_hold(
        lifecycle,
        terminal_position_rad=terminal_position_rad,
    )

    assert lifecycle.state is GroceryCloseLifecycleState.CLOSED_HOLD
    assert lifecycle.closure_settled
    evidence = lifecycle.observe_pre_step(
        _sample(
            sequence + 1,
            terminal_position_rad,
        )
    )
    assert evidence.state is GroceryCloseLifecycleState.CLOSED_HOLD
    assert evidence.closure_settled
    assert not evidence.newly_closure_settled
    assert evidence.closure_settled_marker is None


def test_settled_rate_threshold_is_strict() -> None:
    lifecycle = GroceryCloseLifecycle()
    sequence, position_rad = _begin_close(lifecycle)
    while position_rad < DROID_GRIPPER_CLOSED_POSITION_RAD - 5.0e-6:
        sequence += 1
        position_rad = min(
            position_rad + 0.004,
            DROID_GRIPPER_CLOSED_POSITION_RAD - 5.0e-6,
        )
        lifecycle.observe_pre_step(_sample(sequence, position_rad))

    sequence += 1
    position_rad += CLOSURE_SETTLED_RATE_BOUND_RAD_S * PINNED_SIMULATION_DT_S
    boundary = lifecycle.observe_pre_step(_sample(sequence, position_rad))
    assert boundary.derived_driver_rate_rad_s == pytest.approx(
        CLOSURE_SETTLED_RATE_BOUND_RAD_S
    )
    assert boundary.closure_settled_pair_count == 0
    assert not boundary.closure_settled

    sequence += 1
    first_strict = lifecycle.observe_pre_step(_sample(sequence, position_rad))
    assert first_strict.closure_settled_pair_count == 1
    sequence += 1
    second_strict = lifecycle.observe_pre_step(_sample(sequence, position_rad))
    assert second_strict.closure_settled_pair_count == 2
    assert second_strict.closure_settled


@pytest.mark.parametrize(
    "passive_rate_rad_s",
    (
        CLOSURE_SETTLED_RATE_BOUND_RAD_S,
        CLOSURE_SETTLED_RATE_BOUND_RAD_S + 1.0e-6,
    ),
)
def test_stopped_driver_cannot_hide_a_moving_passive_joint(
    passive_rate_rad_s: float,
) -> None:
    lifecycle = GroceryCloseLifecycle()
    sequence, position_rad = _begin_close(lifecycle)
    while position_rad < DROID_GRIPPER_CLOSED_POSITION_RAD:
        sequence += 1
        position_rad = min(
            position_rad + 0.004,
            DROID_GRIPPER_CLOSED_POSITION_RAD,
        )
        lifecycle.observe_pre_step(_sample(sequence, position_rad))

    prior_positions = (position_rad,) * 6
    moving_positions = list(prior_positions)
    moving_positions[4] += passive_rate_rad_s * PINNED_SIMULATION_DT_S
    sequence += 1
    evidence = lifecycle.observe_pre_step(
        _sample(
            sequence,
            position_rad,
            gripper_joint_positions_rad=tuple(moving_positions),
        )
    )

    assert evidence.derived_driver_rate_rad_s == 0.0
    assert evidence.derived_gripper_joint_rates_rad_s is not None
    assert evidence.derived_gripper_joint_rates_rad_s[4] == pytest.approx(
        passive_rate_rad_s
    )
    assert evidence.max_abs_derived_gripper_joint_rate_rad_s == pytest.approx(
        passive_rate_rad_s
    )
    assert evidence.closure_settled_pair_count == 0
    assert not evidence.closure_settled


def test_settled_pair_count_resets_when_driver_motion_resumes() -> None:
    lifecycle = GroceryCloseLifecycle()
    sequence, position_rad = _begin_close(lifecycle)
    while position_rad < DROID_GRIPPER_CLOSED_POSITION_RAD - 0.001:
        sequence += 1
        position_rad = min(
            position_rad + 0.004,
            DROID_GRIPPER_CLOSED_POSITION_RAD - 0.001,
        )
        lifecycle.observe_pre_step(_sample(sequence, position_rad))

    sequence += 1
    first_settled = lifecycle.observe_pre_step(_sample(sequence, position_rad))
    assert first_settled.closure_settled_pair_count == 1

    sequence += 1
    position_rad += 0.0005
    resumed = lifecycle.observe_pre_step(_sample(sequence, position_rad))
    assert resumed.derived_driver_rate_rad_s == pytest.approx(0.1)
    assert resumed.closure_settled_pair_count == 0

    sequence += 1
    first_new_pair = lifecycle.observe_pre_step(_sample(sequence, position_rad))
    assert first_new_pair.closure_settled_pair_count == 1
    assert not first_new_pair.closure_settled
    sequence += 1
    second_new_pair = lifecycle.observe_pre_step(_sample(sequence, position_rad))
    assert second_new_pair.closure_settled_pair_count == 2
    assert second_new_pair.closure_settled


def test_closure_does_not_latch_without_the_final_physical_sample_proof() -> None:
    lifecycle = GroceryCloseLifecycle()
    sequence, position_rad = _begin_close(lifecycle)
    while position_rad < DROID_GRIPPER_CLOSED_POSITION_RAD:
        sequence += 1
        position_rad = min(
            position_rad + 0.004,
            DROID_GRIPPER_CLOSED_POSITION_RAD,
        )
        lifecycle.observe_pre_step(_sample(sequence, position_rad))
    sequence += 1
    first_settled = lifecycle.observe_pre_step(_sample(sequence, position_rad))
    assert first_settled.closure_settled_pair_count == 1

    with pytest.raises(GroceryCloseLifecycleError, match="geometry safety proof"):
        lifecycle.observe_pre_step(
            _sample(
                sequence + 1,
                position_rad,
                physical_sample_safe=False,
            )
        )
    assert lifecycle.state is GroceryCloseLifecycleState.OPEN
    assert not lifecycle.closure_settled


def test_closure_settled_marker_states_only_the_proven_fact() -> None:
    fresh = GroceryCloseLifecycle()
    _, _ = _begin_close(fresh)
    sequence = 1
    position_rad = 0.004
    while position_rad < DROID_GRIPPER_CLOSED_POSITION_RAD:
        sequence += 1
        position_rad = min(
            position_rad + 0.004,
            DROID_GRIPPER_CLOSED_POSITION_RAD,
        )
        fresh.observe_pre_step(_sample(sequence, position_rad))
    sequence += 1
    fresh.observe_pre_step(_sample(sequence, position_rad))
    sequence += 1
    transition = fresh.observe_pre_step(_sample(sequence, position_rad))
    marker = transition.closure_settled_marker
    assert marker is not None
    assert marker.startswith("CAP_GROCERY_CLOSURE_SETTLED ")
    assert "closure_settled_pairs=2" in marker
    assert "grasp" not in marker.lower()
    assert "retention" not in marker.lower()


@pytest.mark.parametrize(
    ("loss", "match"),
    [
        ("rate", "strictly-settled all-joint rate"),
        ("position", "full-close endpoint band"),
        ("physical", "physical sample proof"),
        ("sequence", "sequence-adjacent"),
        ("timestamp", "positive simulation timestamp delta"),
    ],
)
def test_closed_hold_loss_raises_and_resets_fail_closed(
    loss: str,
    match: str,
) -> None:
    lifecycle = GroceryCloseLifecycle()
    sequence, settled_sample = _drive_to_closed_hold(lifecycle)
    next_sequence = sequence + 1
    next_position = settled_sample.driver_position_rad
    next_timestamp = next_sequence * PINNED_SIMULATION_DT_S
    physical_sample_safe = True
    if loss == "rate":
        next_position += CLOSURE_SETTLED_RATE_BOUND_RAD_S * PINNED_SIMULATION_DT_S
    elif loss == "position":
        next_position = (
            DROID_GRIPPER_CLOSED_POSITION_RAD
            + DROID_GRIPPER_ENDPOINT_TOLERANCE_RAD
            + 1.0e-6
        )
    elif loss == "physical":
        physical_sample_safe = False
    elif loss == "sequence":
        next_sequence += 1
        next_timestamp = next_sequence * PINNED_SIMULATION_DT_S
    elif loss == "timestamp":
        next_timestamp = settled_sample.simulation_timestamp_s

    with pytest.raises(GroceryCloseLifecycleError, match=match):
        lifecycle.observe_pre_step(
            _sample(
                next_sequence,
                next_position,
                timestamp_s=next_timestamp,
                physical_sample_safe=physical_sample_safe,
            )
        )
    assert lifecycle.state is GroceryCloseLifecycleState.OPEN
    assert not lifecycle.closure_settled
    with pytest.raises(GroceryCloseLifecycleError, match="prior sequence-adjacent"):
        lifecycle.observe_pre_step(_sample(next_sequence + 1, next_position))


def test_closed_hold_rejects_one_passive_joint_at_the_exact_rate_bound() -> None:
    lifecycle = GroceryCloseLifecycle()
    sequence, settled_sample = _drive_to_closed_hold(lifecycle)
    positions = list(settled_sample.gripper_joint_positions_rad)
    positions[-1] += CLOSURE_SETTLED_RATE_BOUND_RAD_S * PINNED_SIMULATION_DT_S

    with pytest.raises(
        GroceryCloseLifecycleError,
        match="strictly-settled all-joint rate",
    ):
        lifecycle.observe_pre_step(
            _sample(
                sequence + 1,
                settled_sample.driver_position_rad,
                gripper_joint_positions_rad=tuple(positions),
            )
        )

    assert lifecycle.state is GroceryCloseLifecycleState.OPEN
    assert not lifecycle.closure_settled


def test_open_target_clears_closed_hold_instead_of_reporting_loss() -> None:
    lifecycle = GroceryCloseLifecycle()
    sequence, settled_sample = _drive_to_closed_hold(lifecycle)

    evidence = lifecycle.observe_pre_step(
        _sample(
            sequence + 4,
            settled_sample.driver_position_rad,
            GroceryGripperTarget.OPEN,
            physical_sample_safe=False,
        )
    )

    assert evidence.state is GroceryCloseLifecycleState.OPEN
    assert not evidence.closure_settled


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"sequence": True}, "sequence must be an integer"),
        ({"sequence": -1}, "sequence must be nonnegative"),
        ({"simulation_timestamp_s": math.nan}, "timestamp.*finite"),
        (
            {
                "gripper_joint_positions_rad": (
                    0.0,
                    0.0,
                    0.0,
                    math.inf,
                    0.0,
                    0.0,
                )
            },
            "gripper_joint_positions_rad.*finite",
        ),
        (
            {"gripper_joint_positions_rad": (0.0,) * 5},
            "exactly 6 numbers",
        ),
        ({"requested_target": "CLOSE"}, "requested_target"),
        ({"requested_target": True}, "requested_target"),
        (
            {"physical_sample_safe": 1},
            "physical_sample_safe must be boolean",
        ),
    ],
)
def test_invalid_samples_reject_and_clear_prior_proof(
    changes: dict[str, object],
    match: str,
) -> None:
    lifecycle = GroceryCloseLifecycle()
    lifecycle.observe_pre_step(_sample(0, 0.0, GroceryGripperTarget.OPEN))
    values: dict[str, object] = {
        "sequence": 1,
        "simulation_timestamp_s": PINNED_SIMULATION_DT_S,
        "gripper_joint_positions_rad": (0.004,) * 6,
        "requested_target": GroceryGripperTarget.CLOSE,
        "physical_sample_safe": True,
    }
    values.update(changes)

    with pytest.raises(GroceryCloseLifecycleError, match=match):
        lifecycle.observe_pre_step(
            GroceryCloseLifecycleSample(**values)  # type: ignore[arg-type]
        )
    assert lifecycle.state is GroceryCloseLifecycleState.OPEN
    with pytest.raises(GroceryCloseLifecycleError, match="prior sequence-adjacent"):
        lifecycle.observe_pre_step(_sample(2, 0.008))


def test_fresh_close_schema_has_no_arm_raw_velocity_or_obstruction_inputs() -> None:
    lifecycle = GroceryCloseLifecycle()
    lifecycle.observe_pre_step(_sample(0, 0.0, GroceryGripperTarget.OPEN))
    evidence = lifecycle.observe_pre_step(_sample(1, 0.004))
    assert evidence.state is GroceryCloseLifecycleState.CLOSE_TRANSITION

    field_names = {
        field.name for field in dataclasses.fields(GroceryCloseLifecycleSample)
    }
    assert field_names == {
        "sequence",
        "simulation_timestamp_s",
        "gripper_joint_positions_rad",
        "requested_target",
        "physical_sample_safe",
    }
    assert all("velocity" not in name for name in field_names)
    assert all("arm" not in name for name in field_names)
    assert all("obstruction" not in name for name in field_names)

    with pytest.raises(TypeError, match="unexpected keyword"):
        GroceryCloseLifecycleSample(
            sequence=0,
            simulation_timestamp_s=0.0,
            gripper_joint_positions_rad=(0.0,) * 6,
            requested_target=GroceryGripperTarget.OPEN,
            physical_sample_safe=True,
            raw_joint_velocity_rad_s=0.0,  # type: ignore[call-arg]
        )

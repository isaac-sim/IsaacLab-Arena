# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import struct

import pytest

from isaaclab_arena.integrations.cap_barrier.joint_mapping import DROID_GRIPPER_CLOSED_POSITION_RAD
from isaaclab_arena.integrations.cap_barrier.lockstep_manager import FrameResult
from isaaclab_arena.integrations.cap_barrier.production_smoke import (
    CAP_PRODUCTION_KIT_ENV_READY_MARKER,
    CAP_PRODUCTION_STARTUP_RENDEZVOUS_TIMEOUT_S,
    DROID_GRIPPER_SMOKE_ARM_PHYSICAL_DRIFT_TRIGGER_RAD,
    DROID_GRIPPER_SMOKE_ARM_PHYSICAL_TOLERANCE_RAD,
    DROID_GRIPPER_SMOKE_CALIBRATED_ARM_PHYSICAL_MAX_RAD,
    DroidGripperTransitionProof,
    open_production_startup_rendezvous,
    run_physics_until_generation_transition,
)
from isaaclab_arena.integrations.cap_barrier.protocol import BarrierPhase, FrameKind, ServiceabilityState
from isaaclab_arena.integrations.cap_barrier.shared_memory import BarrierInterrupted, BarrierStatus


def test_production_startup_marks_environment_ready_before_barrier_open() -> None:
    events = []

    class _Environment:
        def close(self) -> None:
            events.append("environment-close")

    class _Client:
        pass

    def make_environment() -> _Environment:
        events.append("environment-created")
        return _Environment()

    def emit_marker(marker: str) -> None:
        events.append(marker)

    def open_barrier(deadline: float) -> _Client:
        events.append(f"barrier-open:{deadline}")
        return _Client()

    environment, client, deadline = open_production_startup_rendezvous(
        make_environment,
        open_barrier,
        marker_sink=emit_marker,
        monotonic=lambda: 100.0,
    )

    assert isinstance(environment, _Environment)
    assert isinstance(client, _Client)
    assert deadline == 100.0 + CAP_PRODUCTION_STARTUP_RENDEZVOUS_TIMEOUT_S
    assert events == [
        "environment-created",
        CAP_PRODUCTION_KIT_ENV_READY_MARKER,
        f"barrier-open:{deadline}",
    ]


def test_production_startup_closes_environment_when_barrier_open_fails() -> None:
    events = []

    class _Environment:
        def close(self) -> None:
            events.append("environment-close")

    def fail_barrier_open(deadline: float):
        events.append(f"barrier-open:{deadline}")
        raise RuntimeError("barrier unavailable")

    with pytest.raises(RuntimeError, match="barrier unavailable"):
        open_production_startup_rendezvous(
            lambda: _Environment(),
            fail_barrier_open,
            marker_sink=lambda marker: events.append(marker),
            monotonic=lambda: 20.0,
        )

    assert events == [
        CAP_PRODUCTION_KIT_ENV_READY_MARKER,
        f"barrier-open:{20.0 + CAP_PRODUCTION_STARTUP_RENDEZVOUS_TIMEOUT_S}",
        "environment-close",
    ]


class _TransitioningManager:
    def __init__(self, transition_after: int, transition_status: BarrierStatus):
        self.generation = 1
        self.transition_after = transition_after
        self.transition_status = transition_status
        self.published = 0

    @property
    def barrier_status(self) -> BarrierStatus:
        if self.published >= self.transition_after:
            return self.transition_status
        return BarrierStatus(
            active_generation=1,
            phase=BarrierPhase.AWAIT_STATE,
            serviceability=ServiceabilityState.SERVICEABLE,
        )

    def physics_step(self) -> FrameResult:
        assert self.barrier_status.permits_publish(1), "old generation published after reset became observable"
        result = FrameResult(
            frame_kind=FrameKind.PHYSICS,
            generation=1,
            sequence=self.published,
            physics_tick=self.published,
            commanded_positions=(0.1,),
        )
        self.published += 1
        return result


@pytest.mark.parametrize(
    "transition_status",
    [
        BarrierStatus(
            active_generation=1,
            phase=BarrierPhase.AWAIT_STATE,
            serviceability=ServiceabilityState.INTERRUPTED,
        ),
        BarrierStatus(
            active_generation=2,
            phase=BarrierPhase.AWAIT_STATE,
            serviceability=ServiceabilityState.SERVICEABLE,
        ),
    ],
)
def test_generation_transition_stops_before_another_old_generation_publish(
    transition_status: BarrierStatus,
) -> None:
    manager = _TransitioningManager(transition_after=137, transition_status=transition_status)
    observed_ticks = []

    observation = run_physics_until_generation_transition(
        manager,
        timeout_s=1.0,
        on_frame=lambda result: observed_ticks.append(result.physics_tick),
    )

    assert observation.previous_generation == 1
    assert observation.status == transition_status
    assert observation.physics_frames == 137
    assert manager.published == 137
    assert observed_ticks == list(range(137))


def test_generation_transition_rejects_non_quiescent_publish_boundary() -> None:
    manager = _TransitioningManager(
        transition_after=0,
        transition_status=BarrierStatus(
            active_generation=1,
            phase=BarrierPhase.COMMAND_READY,
            serviceability=ServiceabilityState.SERVICEABLE,
        ),
    )

    with pytest.raises(RuntimeError, match="not quiescent"):
        run_physics_until_generation_transition(manager, timeout_s=1.0)
    assert manager.published == 0


def test_generation_transition_handles_reset_winning_the_publish_reservation() -> None:
    class _InterruptingManager:
        def __init__(self):
            self.generation = 1
            self.interrupted = False

        @property
        def barrier_status(self) -> BarrierStatus:
            return BarrierStatus(
                active_generation=1,
                phase=BarrierPhase.AWAIT_STATE,
                serviceability=(
                    ServiceabilityState.INTERRUPTED if self.interrupted else ServiceabilityState.SERVICEABLE
                ),
            )

        def physics_step(self) -> FrameResult:
            self.interrupted = True
            raise BarrierInterrupted("reset won the reservation")

    manager = _InterruptingManager()
    observation = run_physics_until_generation_transition(manager, timeout_s=1.0)

    assert observation.physics_frames == 0
    assert observation.status.serviceability == ServiceabilityState.INTERRUPTED


def _droid_frame(sequence: int, gripper_position: float, arm_position: float = 0.0) -> FrameResult:
    return FrameResult(
        frame_kind=FrameKind.PHYSICS,
        generation=1,
        sequence=sequence,
        physics_tick=sequence,
        commanded_positions=(*([arm_position] * 7), gripper_position),
    )


def _physical(gripper_position: float, arm_position: float = 0.0) -> tuple[float, ...]:
    return (*([arm_position] * 7), gripper_position)


def _next_float32(value: float) -> float:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return struct.unpack("<f", struct.pack("<I", bits + 1))[0]


def test_gripper_transition_arm_reaction_thresholds_are_derived_from_calibration() -> None:
    assert DROID_GRIPPER_SMOKE_CALIBRATED_ARM_PHYSICAL_MAX_RAD == 0.00016736984252929688
    assert DROID_GRIPPER_SMOKE_ARM_PHYSICAL_DRIFT_TRIGGER_RAD == DROID_GRIPPER_SMOKE_CALIBRATED_ARM_PHYSICAL_MAX_RAD
    assert DROID_GRIPPER_SMOKE_ARM_PHYSICAL_TOLERANCE_RAD == 2.0 * DROID_GRIPPER_SMOKE_CALIBRATED_ARM_PHYSICAL_MAX_RAD
    assert _next_float32(DROID_GRIPPER_SMOKE_ARM_PHYSICAL_DRIFT_TRIGGER_RAD) == 0.0001673698570812121
    assert _next_float32(DROID_GRIPPER_SMOKE_ARM_PHYSICAL_TOLERANCE_RAD) == 0.0003347397141624242


def _observe_gripper_transition_before_final_open(
    proof: DroidGripperTransitionProof,
    arm_physical_peak_rad: float,
) -> None:
    half_closed = 0.5 * DROID_GRIPPER_CLOSED_POSITION_RAD
    proof.observe(
        _droid_frame(0, DROID_GRIPPER_CLOSED_POSITION_RAD),
        _physical(half_closed + 0.01),
        step_started_at_s=1.0,
        observed_at_s=1.1,
    )
    proof.observe(
        _droid_frame(1, DROID_GRIPPER_CLOSED_POSITION_RAD),
        _physical(DROID_GRIPPER_CLOSED_POSITION_RAD - 0.005),
        step_started_at_s=1.2,
        observed_at_s=1.4,
    )
    proof.observe(
        _droid_frame(2, 0.0),
        _physical(half_closed - 0.01, arm_physical_peak_rad),
        step_started_at_s=1.6,
        observed_at_s=1.8,
    )


def test_gripper_transition_proof_observes_bounded_physical_open_close_open() -> None:
    proof = DroidGripperTransitionProof(_physical(0.0))
    half_closed = 0.5 * DROID_GRIPPER_CLOSED_POSITION_RAD

    proof.observe(_droid_frame(0, 0.0), _physical(0.0), step_started_at_s=0.0, observed_at_s=0.01)
    proof.observe(
        _droid_frame(1, DROID_GRIPPER_CLOSED_POSITION_RAD),
        _physical(0.2),
        step_started_at_s=1.0,
        observed_at_s=1.1,
    )
    proof.observe(
        _droid_frame(2, DROID_GRIPPER_CLOSED_POSITION_RAD),
        _physical(half_closed + 0.01),
        step_started_at_s=1.2,
        observed_at_s=1.5,
    )
    proof.observe(
        _droid_frame(3, DROID_GRIPPER_CLOSED_POSITION_RAD),
        _physical(DROID_GRIPPER_CLOSED_POSITION_RAD - 0.005),
        step_started_at_s=1.6,
        observed_at_s=1.8,
    )
    proof.observe(
        _droid_frame(4, 0.0),
        _physical(0.6),
        step_started_at_s=2.0,
        observed_at_s=2.1,
    )
    proof.observe(
        _droid_frame(5, 0.0),
        _physical(half_closed - 0.01, DROID_GRIPPER_SMOKE_ARM_PHYSICAL_DRIFT_TRIGGER_RAD),
        step_started_at_s=2.2,
        observed_at_s=2.5,
    )
    proof.observe(
        _droid_frame(6, 0.0),
        _physical(0.005),
        step_started_at_s=2.6,
        observed_at_s=2.8,
    )

    observation = proof.observation()
    assert observation.close_elapsed_s == pytest.approx(0.8)
    assert observation.open_elapsed_s == pytest.approx(0.8)
    assert observation.closed_position_rad == pytest.approx(DROID_GRIPPER_CLOSED_POSITION_RAD - 0.005)
    assert observation.final_position_rad == pytest.approx(0.005)
    assert observation.maximum_arm_command_delta_rad == 0.0
    assert observation.maximum_arm_physical_delta_rad == DROID_GRIPPER_SMOKE_ARM_PHYSICAL_DRIFT_TRIGGER_RAD


def test_gripper_transition_proof_rejects_command_echo_without_physical_motion() -> None:
    proof = DroidGripperTransitionProof(_physical(0.0))
    proof.observe(
        _droid_frame(0, DROID_GRIPPER_CLOSED_POSITION_RAD),
        _physical(0.0),
        step_started_at_s=1.0,
        observed_at_s=1.1,
    )
    with pytest.raises(TimeoutError, match="close transition"):
        proof.observe(
            _droid_frame(1, DROID_GRIPPER_CLOSED_POSITION_RAD),
            _physical(0.0),
            step_started_at_s=3.0,
            observed_at_s=3.01,
        )


@pytest.mark.parametrize(
    "drift",
    [
        _next_float32(DROID_GRIPPER_SMOKE_ARM_PHYSICAL_DRIFT_TRIGGER_RAD),
        DROID_GRIPPER_SMOKE_ARM_PHYSICAL_TOLERANCE_RAD,
    ],
)
def test_gripper_transition_proof_defers_calibration_drift_until_transition_completion(drift: float) -> None:
    proof = DroidGripperTransitionProof(_physical(0.0))
    _observe_gripper_transition_before_final_open(proof, drift)
    with pytest.raises(RuntimeError, match="physical calibration drift") as error:
        proof.observe(
            _droid_frame(3, 0.0),
            _physical(0.005, 0.0),
            step_started_at_s=2.0,
            observed_at_s=2.2,
        )

    assert "command_delta=0.0" in str(error.value)
    assert f"physical_delta={drift}" in str(error.value)
    assert "new maximum exceeds calibrated envelope" in str(error.value)
    assert "extend calibration with additional runs and adopt the new max" in str(error.value)
    assert "full recalibration per config-scope rule" in str(error.value)
    assert "physical_peak_sample=3" in str(error.value)
    assert "physical_peak_phase=reopening" in str(error.value)


def test_gripper_transition_proof_distinguishes_physical_isolation_failure() -> None:
    proof = DroidGripperTransitionProof(_physical(0.0))
    physical_delta = _next_float32(DROID_GRIPPER_SMOKE_ARM_PHYSICAL_TOLERANCE_RAD)
    _observe_gripper_transition_before_final_open(proof, physical_delta)
    with pytest.raises(RuntimeError, match="physical isolation failure") as error:
        proof.observe(
            _droid_frame(3, 0.0),
            _physical(0.005),
            step_started_at_s=2.0,
            observed_at_s=2.2,
        )

    assert f"physical_delta={physical_delta}" in str(error.value)
    assert "calibration drift" not in str(error.value)
    assert "physical_peak_phase=reopening" in str(error.value)


def test_gripper_transition_proof_rejects_nonzero_arm_command_immediately() -> None:
    proof = DroidGripperTransitionProof(_physical(0.0))
    with pytest.raises(RuntimeError, match="command isolation failure") as error:
        proof.observe(
            _droid_frame(0, DROID_GRIPPER_CLOSED_POSITION_RAD, 1e-12),
            _physical(0.1, 0.0),
            step_started_at_s=1.0,
            observed_at_s=1.1,
        )
    assert "command_delta=1e-12" in str(error.value)
    assert "command_tolerance=0.0" in str(error.value)
    assert f"physical_hard_limit={DROID_GRIPPER_SMOKE_ARM_PHYSICAL_TOLERANCE_RAD}" in str(error.value)


@pytest.mark.parametrize(
    "tolerance",
    [-1.0, float("nan"), float("inf")],
)
@pytest.mark.parametrize(
    "parameter",
    ["arm_command_tolerance_rad", "arm_physical_tolerance_rad", "arm_physical_drift_trigger_rad"],
)
def test_gripper_transition_proof_rejects_invalid_arm_tolerances(parameter: str, tolerance: float) -> None:
    with pytest.raises(ValueError, match="arm .*"):
        DroidGripperTransitionProof(_physical(0.0), **{parameter: tolerance})


def test_gripper_transition_proof_rejects_drift_trigger_above_physical_tolerance() -> None:
    with pytest.raises(ValueError, match="drift trigger must not exceed"):
        DroidGripperTransitionProof(
            _physical(0.0),
            arm_physical_drift_trigger_rad=2.0,
            arm_physical_tolerance_rad=1.0,
        )


@pytest.mark.parametrize("initial", [0.1, float("nan"), float("inf")])
def test_gripper_transition_proof_rejects_unsupported_initial_state(initial: float) -> None:
    expected = ValueError if not math.isfinite(initial) else RuntimeError
    with pytest.raises(expected):
        DroidGripperTransitionProof(_physical(initial))

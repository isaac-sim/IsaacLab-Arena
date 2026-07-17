# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""ROS-free generation handoff used by the production control-plane smoke."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .joint_mapping import (
    DROID_GRIPPER_CLOSED_POSITION_RAD,
    DROID_GRIPPER_ENDPOINT_TOLERANCE_RAD,
    DROID_GRIPPER_OPEN_POSITION_RAD,
    droid_binary_gripper_action,
)
from .lockstep_manager import ArenaLockstepManager, FrameResult
from .protocol import FrameKind
from .shared_memory import BarrierInterrupted, BarrierStatus

DROID_GRIPPER_TRANSITION_TIMEOUT_S = 2.0
DROID_GRIPPER_HALF_CLOSED_POSITION_RAD = 0.5 * DROID_GRIPPER_CLOSED_POSITION_RAD
DROID_GRIPPER_SMOKE_ARM_COMMAND_TOLERANCE_RAD = 0.0
DROID_GRIPPER_SMOKE_ARM_PHYSICAL_TOLERANCE_RAD = 1e-4


@dataclass(frozen=True)
class GenerationTransitionObservation:
    """Generation transition observed before publishing another physics frame."""

    previous_generation: int
    status: BarrierStatus
    physics_frames: int


@dataclass(frozen=True)
class DroidGripperTransitionObservation:
    """Physical open-to-close-to-open evidence collected during PHYSICS frames."""

    close_elapsed_s: float
    open_elapsed_s: float
    closed_position_rad: float
    final_position_rad: float
    maximum_arm_command_delta_rad: float
    maximum_arm_physical_delta_rad: float


class DroidGripperTransitionProof:
    """Require a bounded physical open-to-close-to-open transition while the arm holds."""

    def __init__(
        self,
        initial_positions: Sequence[float],
        *,
        transition_timeout_s: float = DROID_GRIPPER_TRANSITION_TIMEOUT_S,
        endpoint_tolerance_rad: float = DROID_GRIPPER_ENDPOINT_TOLERANCE_RAD,
        arm_command_tolerance_rad: float = DROID_GRIPPER_SMOKE_ARM_COMMAND_TOLERANCE_RAD,
        arm_physical_tolerance_rad: float = DROID_GRIPPER_SMOKE_ARM_PHYSICAL_TOLERANCE_RAD,
    ):
        initial = self._validated_positions(initial_positions, "initial physical state")
        if transition_timeout_s <= 0 or not math.isfinite(transition_timeout_s):
            raise ValueError("gripper transition timeout must be finite and positive")
        if endpoint_tolerance_rad <= 0 or not math.isfinite(endpoint_tolerance_rad):
            raise ValueError("gripper endpoint tolerance must be finite and positive")
        if arm_command_tolerance_rad < 0 or not math.isfinite(arm_command_tolerance_rad):
            raise ValueError("gripper smoke arm command tolerance must be finite and nonnegative")
        if arm_physical_tolerance_rad < 0 or not math.isfinite(arm_physical_tolerance_rad):
            raise ValueError("gripper smoke arm physical tolerance must be finite and nonnegative")
        if not self._within_endpoint(initial[7], DROID_GRIPPER_OPEN_POSITION_RAD, endpoint_tolerance_rad):
            raise RuntimeError(
                "DROID gripper transition proof must start physically open: "
                f"position={initial[7]} tolerance={endpoint_tolerance_rad}"
            )

        self._arm_reference = initial[:7]
        self._transition_timeout_s = transition_timeout_s
        self._endpoint_tolerance_rad = endpoint_tolerance_rad
        self._arm_command_tolerance_rad = arm_command_tolerance_rad
        self._arm_physical_tolerance_rad = arm_physical_tolerance_rad
        self._phase = "await_close"
        self._close_started_at_s: float | None = None
        self._open_started_at_s: float | None = None
        self._close_elapsed_s: float | None = None
        self._open_elapsed_s: float | None = None
        self._closed_position_rad: float | None = None
        self._final_position_rad: float | None = None
        self._close_midpoint_crossed = False
        self._open_midpoint_crossed = False
        self._maximum_arm_command_delta_rad = 0.0
        self._maximum_arm_physical_delta_rad = 0.0
        self._arm_observation_count = 0
        self._maximum_arm_physical_delta_sample: int | None = None
        self._maximum_arm_physical_delta_phase: str | None = None

    @property
    def complete(self) -> bool:
        return self._phase == "complete"

    @staticmethod
    def _within_endpoint(position: float, endpoint: float, tolerance: float) -> bool:
        return abs(position - endpoint) <= tolerance

    @staticmethod
    def _validated_positions(values: Sequence[float], label: str) -> tuple[float, ...]:
        positions = tuple(float(value) for value in values)
        if len(positions) != 8:
            raise ValueError(f"{label} must contain eight DROID joints, got {len(positions)}")
        if not all(math.isfinite(value) for value in positions):
            raise ValueError(f"{label} must contain only finite joint positions")
        return positions

    @staticmethod
    def _maximum_delta(lhs: Sequence[float], rhs: Sequence[float]) -> float:
        return max(abs(left - right) for left, right in zip(lhs, rhs, strict=True))

    def _observe_held_arm(
        self,
        commanded_positions: tuple[float, ...],
        physical_positions: tuple[float, ...],
        transition_phase: str,
    ) -> None:
        self._arm_observation_count += 1
        command_delta = self._maximum_delta(commanded_positions[:7], self._arm_reference)
        physical_delta = self._maximum_delta(physical_positions[:7], self._arm_reference)
        self._maximum_arm_command_delta_rad = max(self._maximum_arm_command_delta_rad, command_delta)
        if physical_delta > self._maximum_arm_physical_delta_rad:
            self._maximum_arm_physical_delta_rad = physical_delta
            self._maximum_arm_physical_delta_sample = self._arm_observation_count
            self._maximum_arm_physical_delta_phase = transition_phase
        if command_delta > self._arm_command_tolerance_rad:
            self._raise_arm_movement()

    def _raise_arm_movement(self) -> None:
        raise RuntimeError(
            "DROID arm moved during the gripper transition proof: "
            f"command_delta={self._maximum_arm_command_delta_rad}, "
            f"command_tolerance={self._arm_command_tolerance_rad}, "
            f"physical_delta={self._maximum_arm_physical_delta_rad}, "
            f"physical_tolerance={self._arm_physical_tolerance_rad}, "
            f"physical_peak_sample={self._maximum_arm_physical_delta_sample}, "
            f"physical_peak_phase={self._maximum_arm_physical_delta_phase}"
        )

    def _validate_physical_arm_reaction(self) -> None:
        if self._maximum_arm_physical_delta_rad > self._arm_physical_tolerance_rad:
            self._raise_arm_movement()

    def _arm_transition_phase(self, command_closedness: float) -> str:
        if self._phase == "await_close":
            return "closing" if command_closedness == 1.0 else "open_settling"
        if self._phase == "await_open":
            return "reopening" if command_closedness == 0.0 else "closed_settling"
        if self._phase == "opening":
            return "reopening"
        return self._phase

    def _elapsed(self, started_at_s: float, observed_at_s: float, direction: str) -> float:
        elapsed_s = observed_at_s - started_at_s
        if elapsed_s > self._transition_timeout_s:
            raise TimeoutError(
                f"DROID gripper {direction} transition exceeded {self._transition_timeout_s}s: elapsed={elapsed_s}"
            )
        return elapsed_s

    def observe(
        self,
        frame: FrameResult,
        physical_positions: Sequence[float],
        *,
        step_started_at_s: float,
        observed_at_s: float,
    ) -> None:
        """Consume one synchronized post-step physical observation."""
        if self.complete:
            return
        if frame.frame_kind != FrameKind.PHYSICS:
            raise ValueError("gripper transition proof accepts only PHYSICS frames")
        commanded = self._validated_positions(frame.commanded_positions, "command frame")
        physical = self._validated_positions(physical_positions, "physical observation")
        if (
            not math.isfinite(step_started_at_s)
            or not math.isfinite(observed_at_s)
            or observed_at_s < step_started_at_s
        ):
            raise ValueError("gripper transition timestamps must be finite and ordered")
        command_closedness = droid_binary_gripper_action(commanded[7])
        self._observe_held_arm(commanded, physical, self._arm_transition_phase(command_closedness))
        physical_position = physical[7]
        if self._phase == "await_close":
            if command_closedness == 0.0:
                if not self._within_endpoint(
                    physical_position,
                    DROID_GRIPPER_OPEN_POSITION_RAD,
                    self._endpoint_tolerance_rad,
                ):
                    raise RuntimeError("DROID gripper left the open endpoint before a close command")
                return
            self._close_started_at_s = step_started_at_s
            self._phase = "closing"

        if self._phase == "closing":
            if command_closedness != 1.0:
                raise RuntimeError("DROID gripper close command reversed before physical completion")
            assert self._close_started_at_s is not None
            elapsed_s = self._elapsed(self._close_started_at_s, observed_at_s, "close")
            if physical_position >= DROID_GRIPPER_HALF_CLOSED_POSITION_RAD:
                self._close_midpoint_crossed = True
            if self._within_endpoint(
                physical_position,
                DROID_GRIPPER_CLOSED_POSITION_RAD,
                self._endpoint_tolerance_rad,
            ):
                if not self._close_midpoint_crossed:
                    raise RuntimeError("DROID gripper reached closed without an observed midpoint crossing")
                self._close_elapsed_s = elapsed_s
                self._closed_position_rad = physical_position
                self._phase = "await_open"
            return

        if self._phase == "await_open":
            if command_closedness == 1.0:
                if not self._within_endpoint(
                    physical_position,
                    DROID_GRIPPER_CLOSED_POSITION_RAD,
                    self._endpoint_tolerance_rad,
                ):
                    raise RuntimeError("DROID gripper left the closed endpoint before an open command")
                return
            self._open_started_at_s = step_started_at_s
            self._phase = "opening"

        if self._phase == "opening":
            if command_closedness != 0.0:
                raise RuntimeError("DROID gripper open command reversed before physical completion")
            assert self._open_started_at_s is not None
            elapsed_s = self._elapsed(self._open_started_at_s, observed_at_s, "open")
            if physical_position <= DROID_GRIPPER_HALF_CLOSED_POSITION_RAD:
                self._open_midpoint_crossed = True
            if self._within_endpoint(
                physical_position,
                DROID_GRIPPER_OPEN_POSITION_RAD,
                self._endpoint_tolerance_rad,
            ):
                if not self._open_midpoint_crossed:
                    raise RuntimeError("DROID gripper reached open without an observed midpoint crossing")
                self._open_elapsed_s = elapsed_s
                self._final_position_rad = physical_position
                self._validate_physical_arm_reaction()
                self._phase = "complete"

    def observation(self) -> DroidGripperTransitionObservation:
        if not self.complete:
            raise RuntimeError(f"DROID gripper transition proof is incomplete in phase {self._phase}")
        assert self._close_elapsed_s is not None
        assert self._open_elapsed_s is not None
        assert self._closed_position_rad is not None
        assert self._final_position_rad is not None
        return DroidGripperTransitionObservation(
            close_elapsed_s=self._close_elapsed_s,
            open_elapsed_s=self._open_elapsed_s,
            closed_position_rad=self._closed_position_rad,
            final_position_rad=self._final_position_rad,
            maximum_arm_command_delta_rad=self._maximum_arm_command_delta_rad,
            maximum_arm_physical_delta_rad=self._maximum_arm_physical_delta_rad,
        )


def run_physics_until_generation_transition(
    manager: ArenaLockstepManager,
    *,
    timeout_s: float,
    on_frame: Callable[[FrameResult], None] | None = None,
) -> GenerationTransitionObservation:
    """Run PHYSICS while the attached generation remains consumer-serviceable.

    Status is sampled only at the quiescent boundary after the previous exchange has
    completed and immediately before constructing the next state frame. A reset may first
    appear as an asserted serviceability latch or as the incremented generation; either one
    stops publication of the old generation.
    """
    generation = manager.generation
    if generation is None:
        raise RuntimeError("cannot observe a generation transition before initial attachment")
    if timeout_s <= 0:
        raise ValueError("generation transition timeout must be positive")

    deadline = time.monotonic() + timeout_s
    physics_frames = 0
    while True:
        status = manager.barrier_status
        if status.active_generation != generation or not status.consumer_serviceable:
            return GenerationTransitionObservation(
                previous_generation=generation,
                status=status,
                physics_frames=physics_frames,
            )
        if not status.permits_publish(generation):
            raise RuntimeError(f"barrier is not quiescent before PHYSICS publish: phase={status.phase.name}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for generation {generation + 1} reset fence")

        try:
            result = manager.physics_step()
        except BarrierInterrupted:
            status = manager.barrier_status
            return GenerationTransitionObservation(
                previous_generation=generation,
                status=status,
                physics_frames=physics_frames,
            )
        if physics_frames == 0 and result.physics_tick != 0:
            raise RuntimeError("first PHYSICS frame of the generation was not phase zero")
        physics_frames += 1
        if on_frame is not None:
            on_frame(result)

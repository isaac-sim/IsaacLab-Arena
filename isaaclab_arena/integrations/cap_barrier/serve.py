# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Serve-only DROID barrier half for the GaP-through-ROS open_gripper skeleton.

Unlike the production smoke, this producer drives no fixed client sequence and
asserts no arm-streaming or reset dance. It keeps the B=1 barrier alive in the
attached generation, applies whatever the CAP control plane commands each PHYSICS
frame, and instruments the gripper (commanded ABI endpoint + physical joint) so an
external policy -- here GaP issuing a single ``open_gripper`` through ROS -- can be
observed actuating the gripper without this producer choosing the action.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

from .joint_mapping import (
    DROID_GRIPPER_CLOSED_POSITION_RAD,
    DROID_GRIPPER_ENDPOINT_TOLERANCE_RAD,
    DROID_GRIPPER_OPEN_POSITION_RAD,
)
from .lockstep_manager import ArenaLockstepManager
from .shared_memory import BarrierInterrupted

# ABI slot of robotiq_85_left_knuckle_joint in the eight-joint DROID contract.
_GRIPPER_ABI_INDEX = 7
# "Substantially closed" band: half the closed endpoint. Used only to record that
# the gripper was observed closed before an open transition, so a transient pass
# through intermediate angles is never mistaken for the closed start state.
_SUBSTANTIALLY_CLOSED_RAD = 0.5 * DROID_GRIPPER_CLOSED_POSITION_RAD
_BASE_PERIOD_S = 0.005


class ServeExit:
    """Why the serve loop stopped, for the caller's success marker decision."""

    OPEN_TRANSITION = "open_transition_observed"
    GENERATION_ADVANCED = "generation_advanced"
    BARRIER_INTERRUPTED = "barrier_interrupted"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class GripperServeObservation:
    """Full gripper trace summary over the serve window."""

    exit_reason: str
    physics_frames: int
    initial_commanded_rad: float
    initial_physical_rad: float
    final_commanded_rad: float
    final_physical_rad: float
    min_physical_rad: float
    max_physical_rad: float
    observed_closed: bool
    observed_open_after_closed: bool

    @property
    def open_transition(self) -> bool:
        return self.observed_open_after_closed


def _is_open(position_rad: float, tolerance_rad: float) -> bool:
    return abs(position_rad - DROID_GRIPPER_OPEN_POSITION_RAD) <= tolerance_rad


def serve_generation_watching_gripper(
    manager: ArenaLockstepManager,
    gripper_position: Callable[[], float],
    *,
    deadline_monotonic_s: float,
    settle_frames: int,
    marker_sink: Callable[[str], None],
    declare_open_success: bool = True,
    gripper_endpoint_tolerance_rad: float = DROID_GRIPPER_ENDPOINT_TOLERANCE_RAD,
    on_physics_frame: Callable[[int], None] | None = None,
) -> GripperServeObservation:
    """Serve one generation's PHYSICS frames, watching for an open_gripper actuation.

    Drives PHYSICS frames at the 200 Hz base while the attached generation stays
    consumer-serviceable, recording the control plane's commanded gripper endpoint
    and the physical finger joint each frame. Success is a physical closed->open
    transition (the effect of the external open_gripper skill on a closed-start
    scene); after it is seen, ``settle_frames`` further frames confirm the gripper
    holds open before returning ``OPEN_TRANSITION``.

    The GaP connector calls ResetEpisode before open_gripper, so the first served
    generation ends with ``GENERATION_ADVANCED`` and the caller is expected to
    follow the reset (attach the next generation, re-bootstrap) and serve again --
    the actuation lands on the post-reset generation. Also stops on barrier
    interrupt or when ``deadline_monotonic_s`` passes; the caller inspects the
    trace and exit reason to decide the verdict.
    """
    generation = manager.generation
    if generation is None:
        raise RuntimeError("cannot serve before an initial generation attachment")
    if settle_frames < 0:
        raise ValueError("settle_frames must be non-negative")

    deadline = deadline_monotonic_s
    frames = 0
    initial_commanded = math.nan
    initial_physical = math.nan
    last_commanded = math.nan
    last_physical = math.nan
    min_physical = math.inf
    max_physical = -math.inf
    observed_closed = False
    observed_open_after_closed = False
    settle_remaining = -1  # activates (>= 0) once the open transition is first seen
    exit_reason = ServeExit.TIMEOUT
    next_tick = time.monotonic()

    while True:
        status = manager.barrier_status
        if status.active_generation != generation or not status.consumer_serviceable:
            exit_reason = ServeExit.GENERATION_ADVANCED
            break
        if not status.permits_publish(generation):
            raise RuntimeError(
                f"barrier is not quiescent before PHYSICS publish: phase={status.phase.name}"
            )
        if time.monotonic() >= deadline:
            exit_reason = ServeExit.TIMEOUT
            break

        try:
            result = manager.physics_step()
        except BarrierInterrupted:
            exit_reason = ServeExit.BARRIER_INTERRUPTED
            break

        # Optional per-frame observer for a co-resident perception producer. It
        # runs on this (main/Kit) thread right after the physics step, so a camera
        # render/sample stays thread-safe; it must be nonblocking and must never
        # raise, so it can never stall or break the barrier serve loop.
        if on_physics_frame is not None:
            on_physics_frame(frames)

        commanded = float(result.commanded_positions[_GRIPPER_ABI_INDEX])
        physical = float(gripper_position())
        if frames == 0:
            initial_commanded = commanded
            initial_physical = physical
            if physical >= _SUBSTANTIALLY_CLOSED_RAD:
                marker_sink(
                    f"CAP_SERVE_KIT_GRIPPER_CLOSED_AT_BOOTSTRAP physical_rad={physical:.6f} "
                    f"commanded_rad={commanded:.6f}"
                )
        last_commanded = commanded
        last_physical = physical
        min_physical = min(min_physical, physical)
        max_physical = max(max_physical, physical)
        frames += 1

        if physical >= _SUBSTANTIALLY_CLOSED_RAD:
            observed_closed = True
        if (
            observed_closed
            and not observed_open_after_closed
            and _is_open(physical, gripper_endpoint_tolerance_rad)
        ):
            observed_open_after_closed = True
            marker_sink(
                f"CAP_SERVE_KIT_OPEN_TRANSITION_OBSERVED frame={frames} "
                f"physical_rad={physical:.6f} commanded_rad={commanded:.6f} "
                f"counts_as_success={int(declare_open_success)}"
            )
            # The GaP open lands only after ResetEpisode; an open on the pre-reset
            # generation is not attributed to the skill. There we keep serving until
            # the reset advances the generation instead of stopping early.
            if declare_open_success:
                settle_remaining = settle_frames

        if declare_open_success:
            if settle_remaining == 0:
                exit_reason = ServeExit.OPEN_TRANSITION
                break
            if settle_remaining > 0:
                settle_remaining -= 1

        next_tick += _BASE_PERIOD_S
        remaining = next_tick - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)

    return GripperServeObservation(
        exit_reason=exit_reason,
        physics_frames=frames,
        initial_commanded_rad=initial_commanded,
        initial_physical_rad=initial_physical,
        final_commanded_rad=last_commanded,
        final_physical_rad=last_physical,
        min_physical_rad=min_physical if frames else math.nan,
        max_physical_rad=max_physical if frames else math.nan,
        observed_closed=observed_closed,
        observed_open_after_closed=observed_open_after_closed,
    )

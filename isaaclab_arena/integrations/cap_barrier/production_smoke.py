# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""ROS-free generation handoff used by the production control-plane smoke."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from .lockstep_manager import ArenaLockstepManager, FrameResult
from .shared_memory import BarrierInterrupted, BarrierStatus


@dataclass(frozen=True)
class GenerationTransitionObservation:
    """Generation transition observed before publishing another physics frame."""

    previous_generation: int
    status: BarrierStatus
    physics_frames: int


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

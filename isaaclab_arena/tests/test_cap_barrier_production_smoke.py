# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from isaaclab_arena.integrations.cap_barrier.lockstep_manager import FrameResult
from isaaclab_arena.integrations.cap_barrier.production_smoke import run_physics_until_generation_transition
from isaaclab_arena.integrations.cap_barrier.protocol import BarrierPhase, FrameKind, ServiceabilityState
from isaaclab_arena.integrations.cap_barrier.shared_memory import BarrierInterrupted, BarrierStatus


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

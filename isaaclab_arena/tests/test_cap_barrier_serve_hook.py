# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The serve loop's optional per-physics-frame hook fires exactly once per step."""

from __future__ import annotations

from isaaclab_arena.integrations.cap_barrier.protocol import (
    BarrierPhase,
    FrameKind,
    ServiceabilityState,
)
from isaaclab_arena.integrations.cap_barrier.lockstep_manager import FrameResult
from isaaclab_arena.integrations.cap_barrier.serve import (
    ServeExit,
    serve_generation_watching_gripper,
)
from isaaclab_arena.integrations.cap_barrier.shared_memory import BarrierStatus


class _ServingManager:
    """Serves a fixed number of PHYSICS frames, then advances the generation."""

    def __init__(self, frames_before_advance: int):
        self.generation = 1
        self._frames_before_advance = frames_before_advance
        self.published = 0

    @property
    def barrier_status(self) -> BarrierStatus:
        if self.published >= self._frames_before_advance:
            return BarrierStatus(
                active_generation=2,
                phase=BarrierPhase.AWAIT_STATE,
                serviceability=ServiceabilityState.SERVICEABLE,
            )
        return BarrierStatus(
            active_generation=1,
            phase=BarrierPhase.AWAIT_STATE,
            serviceability=ServiceabilityState.SERVICEABLE,
        )

    def physics_step(self) -> FrameResult:
        result = FrameResult(
            frame_kind=FrameKind.PHYSICS,
            generation=1,
            sequence=self.published,
            physics_tick=self.published,
            commanded_positions=(0.0,) * 8,
        )
        self.published += 1
        return result


def test_on_physics_frame_hook_fires_once_per_physics_step() -> None:
    manager = _ServingManager(frames_before_advance=5)
    seen: list[int] = []

    observation = serve_generation_watching_gripper(
        manager,
        lambda: 0.0,
        deadline_monotonic_s=float("inf"),
        settle_frames=0,
        declare_open_success=False,
        marker_sink=lambda _marker: None,
        on_physics_frame=seen.append,
    )

    assert observation.exit_reason == ServeExit.GENERATION_ADVANCED
    assert observation.physics_frames == 5
    # One hook call per served physics frame, in order, pre-increment counter.
    assert seen == [0, 1, 2, 3, 4]


def test_serve_loop_is_unchanged_without_a_hook() -> None:
    manager = _ServingManager(frames_before_advance=3)

    observation = serve_generation_watching_gripper(
        manager,
        lambda: 0.0,
        deadline_monotonic_s=float("inf"),
        settle_frames=0,
        declare_open_success=False,
        marker_sink=lambda _marker: None,
    )

    assert observation.exit_reason == ServeExit.GENERATION_ADVANCED
    assert observation.physics_frames == 3

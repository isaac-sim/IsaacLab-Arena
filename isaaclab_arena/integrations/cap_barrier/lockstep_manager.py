# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Frame-to-physics lockstep manager independent of Kit imports."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .joint_mapping import JointOrderMapping
from .protocol import ControllerTimingSpec, FaultCode, FrameKind, JointState, make_state_frame
from .shared_memory import ArenaBarrierClient, BarrierStatus


class SimulationAdapter(Protocol):
    """Minimal simulator surface needed by the lockstep protocol."""

    @property
    def joint_names(self) -> Sequence[str]:
        pass

    def read_joint_state(self) -> tuple[Sequence[float], Sequence[float], Sequence[float]]:
        pass

    def step_position_targets(self, positions_in_abi_order: Sequence[float]) -> None:
        pass

    def reset_without_physics_step(self) -> None:
        pass


@dataclass(frozen=True)
class FrameResult:
    frame_kind: FrameKind
    generation: int
    sequence: int
    physics_tick: int
    commanded_positions: tuple[float, ...]


class ArenaLockstepManager:
    """Drive one B=1 environment from generation-tagged barrier frames."""

    def __init__(
        self,
        *,
        client: ArenaBarrierClient,
        simulation: SimulationAdapter,
        joint_mapping: JointOrderMapping,
        controller_specs: Sequence[ControllerTimingSpec],
        physics_dt_ns: int = 5_000_000,
        command_timeout_s: float = 2.0,
    ):
        if physics_dt_ns != 5_000_000:
            raise ValueError("arena_droid_b1 v0.1 requires a 200 Hz (5 ms) physics base")
        if tuple(simulation.joint_names) != joint_mapping.simulation_joint_names:
            raise ValueError("simulation joint roster does not match the explicit mapping")
        self._client = client
        self._simulation = simulation
        self._joint_mapping = joint_mapping
        self._controller_specs = tuple(controller_specs)
        self._physics_dt_ns = physics_dt_ns
        self._command_timeout_s = command_timeout_s

    @property
    def generation(self) -> int | None:
        return self._client.attached_generation

    @property
    def sequence(self) -> int:
        return self._client.expected_sequence

    @property
    def physics_tick(self) -> int:
        return self._client.expected_physics_tick

    @property
    def barrier_status(self) -> BarrierStatus:
        """Return owner generation/serviceability observed between frame exchanges."""
        return self._client.status

    def attach_initial_generation(self, *, timeout_s: float = 10.0) -> int:
        generation = self._client.wait_for_generation(timeout_s=timeout_s)
        self._simulation.reset_without_physics_step()
        self._client.attach_generation(generation, timeout_s=timeout_s)
        return generation

    def attach_next_generation(self, *, timeout_s: float = 10.0) -> int:
        current = self.generation
        if current is None:
            raise RuntimeError("cannot wait for a next generation before initial attachment")
        generation = self._client.wait_for_generation(after=current, timeout_s=timeout_s)
        if generation != current + 1:
            raise RuntimeError(f"generation skipped: expected {current + 1}, got {generation}")
        self._simulation.reset_without_physics_step()
        self._client.attach_generation(generation, timeout_s=timeout_s)
        return generation

    def fence(self) -> FrameResult:
        """Tick controllers once while physics remains frozen and discard the command."""
        return self._run_frame(FrameKind.FENCE)

    def physics_step(self) -> FrameResult:
        """Exchange one PHYSICS frame, apply its target, and advance Kit exactly once."""
        return self._run_frame(FrameKind.PHYSICS)

    def _run_frame(self, frame_kind: FrameKind) -> FrameResult:
        generation = self.generation
        if generation is None:
            raise RuntimeError("lockstep manager has no attached generation")
        position, velocity, effort = self._simulation.read_joint_state()
        positions = self._joint_mapping.to_abi_order(position)
        velocities = self._joint_mapping.to_abi_order(velocity)
        efforts = self._joint_mapping.to_abi_order(effort)
        joints = tuple(
            JointState(joint_position, joint_velocity, joint_effort)
            for joint_position, joint_velocity, joint_effort in zip(positions, velocities, efforts, strict=True)
        )
        frame = make_state_frame(
            generation=generation,
            sequence=self.sequence,
            physics_tick=self.physics_tick,
            physics_dt_ns=self._physics_dt_ns,
            frame_kind=frame_kind,
            controller_specs=self._controller_specs,
            joints=joints,
        )
        command = self._client.begin_exchange(frame, timeout_s=self._command_timeout_s)
        targets = tuple(command.joints[index].position for index in range(command.header.joint_count))
        try:
            if frame_kind == FrameKind.PHYSICS:
                self._simulation.step_position_targets(targets)
            # FENCE commands are structurally discarded and never reach the simulation.
        except Exception:
            self._client.fail_pending_exchange(FaultCode.BARRIER_STATE_VIOLATION)
            raise
        self._client.complete_exchange()
        return FrameResult(
            frame_kind=frame_kind,
            generation=generation,
            sequence=frame.header.sequence,
            physics_tick=frame.header.physics_tick,
            commanded_positions=targets,
        )

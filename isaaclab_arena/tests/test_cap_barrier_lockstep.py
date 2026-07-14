# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence

import pytest

from isaaclab_arena.integrations.cap_barrier.joint_mapping import (
    FR3_ARM_JOINTS,
    PANDA_ARM_JOINTS,
    JointOrderMapping,
    make_franka_joint_mapping,
)
from isaaclab_arena.integrations.cap_barrier.lockstep_manager import ArenaLockstepManager
from isaaclab_arena.integrations.cap_barrier.protocol import (
    ControllerTimingSpec,
    FrameKind,
    make_command_frame,
    validate_state_frame,
)


class _FakeSimulation:
    def __init__(self):
        self._joint_names = ("panda_finger_joint1", *reversed(PANDA_ARM_JOINTS), "panda_finger_joint2")
        self.positions = [float(index) for index in range(9)]
        self.velocities = [0.1 * index for index in range(9)]
        self.efforts = [0.2 * index for index in range(9)]
        self.step_count = 0
        self.reset_count = 0
        self.targets: list[tuple[float, ...]] = []

    @property
    def joint_names(self) -> Sequence[str]:
        return self._joint_names

    def read_joint_state(self):
        return self.positions, self.velocities, self.efforts

    def step_position_targets(self, positions_in_abi_order):
        self.targets.append(tuple(positions_in_abi_order))
        self.step_count += 1

    def reset_without_physics_step(self):
        self.reset_count += 1
        self.positions = [float(index) for index in range(9)]


class _FakeBarrierClient:
    def __init__(self, simulation: _FakeSimulation):
        self.simulation = simulation
        self.generations = [1, 2]
        self.attached_generation = None
        self.expected_sequence = 0
        self.expected_physics_tick = 0
        self._pending_kind = None
        self._step_count_at_begin = 0
        self.completed: list[FrameKind] = []

    def wait_for_generation(self, *, after=None, timeout_s=10.0):
        del timeout_s
        return self.generations[0] if after is None else self.generations[1]

    def attach_generation(self, generation, *, timeout_s=10.0):
        del timeout_s
        self.attached_generation = generation
        self.expected_sequence = 0
        self.expected_physics_tick = 0

    def begin_exchange(self, state, *, timeout_s=1.0):
        del timeout_s
        validate_state_frame(state)
        assert state.header.generation == self.attached_generation
        assert state.header.sequence == self.expected_sequence
        assert state.header.physics_tick == self.expected_physics_tick
        self._pending_kind = FrameKind(state.header.frame_kind)
        self._step_count_at_begin = self.simulation.step_count
        due_mask = 0 if self._pending_kind == FrameKind.FENCE else 1
        command = make_command_frame(state, due_mask)
        for index in range(command.header.joint_count):
            command.joints[index].position = 0.25 * (index + 1)
        return command

    def complete_exchange(self):
        expected_steps = self._step_count_at_begin + (self._pending_kind == FrameKind.PHYSICS)
        assert self.simulation.step_count == expected_steps, "barrier released before the Kit physics step completed"
        self.completed.append(self._pending_kind)
        self.expected_sequence += 1
        if self._pending_kind == FrameKind.PHYSICS:
            self.expected_physics_tick += 1
        self._pending_kind = None

    def fail_pending_exchange(self, code):
        raise AssertionError(f"unexpected simulated exchange failure: {code}")


def _manager() -> tuple[ArenaLockstepManager, _FakeSimulation, _FakeBarrierClient]:
    simulation = _FakeSimulation()
    client = _FakeBarrierClient(simulation)
    mapping = make_franka_joint_mapping(simulation.joint_names)
    manager = ArenaLockstepManager(
        client=client,
        simulation=simulation,
        joint_mapping=mapping,
        controller_specs=[ControllerTimingSpec("hold_controller")],
    )
    return manager, simulation, client


def test_franka_mapping_is_explicit_and_order_preserving() -> None:
    simulation_names = ("panda_finger_joint1", *reversed(PANDA_ARM_JOINTS), "panda_finger_joint2")
    mapping = make_franka_joint_mapping(simulation_names)
    assert mapping.abi_joint_names == FR3_ARM_JOINTS
    assert mapping.mapped_simulation_joint_names == PANDA_ARM_JOINTS
    assert mapping.to_abi_order(tuple(range(9))) == (7, 6, 5, 4, 3, 2, 1)
    mapping.assert_action_order(PANDA_ARM_JOINTS)
    with pytest.raises(RuntimeError, match="action joint order"):
        mapping.assert_action_order(tuple(reversed(PANDA_ARM_JOINTS)))


def test_mapping_rejects_missing_or_duplicate_joints() -> None:
    with pytest.raises(ValueError, match="missing"):
        make_franka_joint_mapping(PANDA_ARM_JOINTS[:-1])
    with pytest.raises(ValueError, match="duplicates"):
        JointOrderMapping.from_names(("a", "a"), {"a": "x"}, ("x",))


def test_fence_discards_command_and_physics_steps_exactly_once() -> None:
    manager, simulation, client = _manager()
    assert manager.attach_initial_generation() == 1
    assert simulation.reset_count == 1
    before = tuple(simulation.positions)

    fence = manager.fence()
    assert fence.frame_kind == FrameKind.FENCE
    assert simulation.step_count == 0
    assert tuple(simulation.positions) == before
    assert manager.sequence == 1
    assert manager.physics_tick == 0

    physics = manager.physics_step()
    assert physics.frame_kind == FrameKind.PHYSICS
    assert physics.sequence == 1
    assert physics.physics_tick == 0
    assert simulation.step_count == 1
    assert simulation.targets == [(0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75)]
    assert manager.sequence == 2
    assert manager.physics_tick == 1
    assert client.completed == [FrameKind.FENCE, FrameKind.PHYSICS]


def test_next_generation_resets_sim_without_stepping_and_restarts_counters() -> None:
    manager, simulation, _ = _manager()
    manager.attach_initial_generation()
    manager.fence()
    manager.physics_step()
    old_steps = simulation.step_count

    assert manager.attach_next_generation() == 2
    assert simulation.reset_count == 2
    assert simulation.step_count == old_steps
    assert manager.sequence == 0
    assert manager.physics_tick == 0
    first = manager.physics_step()
    assert first.sequence == 0
    assert first.physics_tick == 0

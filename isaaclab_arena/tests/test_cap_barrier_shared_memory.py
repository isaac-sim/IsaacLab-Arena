# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import mmap
import os
import threading
import time
import uuid
from collections.abc import Sequence

import pytest

from isaaclab_arena.integrations.cap_barrier.joint_mapping import PANDA_ARM_JOINTS, make_franka_joint_mapping
from isaaclab_arena.integrations.cap_barrier.lockstep_manager import ArenaLockstepManager
from isaaclab_arena.integrations.cap_barrier.protocol import (
    PROTOCOL_MAJOR,
    PROTOCOL_MINOR,
    SHARED_MEMORY_MAGIC,
    BarrierPhase,
    ControllerTimingSpec,
    FaultCode,
    FrameKind,
    JointState,
    ProtocolError,
    SharedMemoryLayout,
    clone_struct,
    make_command_frame,
    make_state_frame,
    validate_state_frame,
)
from isaaclab_arena.integrations.cap_barrier.shared_memory import (
    _MEMORY_ORDER_ACQ_REL,
    _MEMORY_ORDER_ACQUIRE,
    _MEMORY_ORDER_RELEASE,
    _SYNC,
    ArenaBarrierClient,
)


class _OwnedTestBarrier:
    """Small C++-layout peer used only to exercise the Python transport."""

    def __init__(self):
        self.name = f"/cap_pytest_{uuid.uuid4().hex}"
        flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
        self.descriptor = _SYNC.libc.shm_open(self.name.encode(), flags, 0o600)
        if self.descriptor < 0:
            raise OSError(ctypes.get_errno(), "shm_open failed")
        os.ftruncate(self.descriptor, ctypes.sizeof(SharedMemoryLayout))
        self.mapping = mmap.mmap(
            self.descriptor,
            ctypes.sizeof(SharedMemoryLayout),
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
        self.layout = SharedMemoryLayout.from_buffer(self.mapping)
        self.base = ctypes.addressof(self.layout)
        self._configure_libc()
        self.layout.magic = SHARED_MEMORY_MAGIC
        self.layout.layout_size = ctypes.sizeof(SharedMemoryLayout)
        self.layout.protocol_major = PROTOCOL_MAJOR
        self.layout.protocol_minor = PROTOCOL_MINOR
        assert _SYNC.libc.sem_init(self.address("state_ready"), 1, 0) == 0
        assert _SYNC.libc.sem_init(self.address("command_ready"), 1, 0) == 0
        self.store4("initialized", 1)
        self.begin_generation(1)
        self.thread_errors: list[BaseException] = []

    @staticmethod
    def _configure_libc() -> None:
        _SYNC.libc.sem_init.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint]
        _SYNC.libc.sem_init.restype = ctypes.c_int
        _SYNC.libc.sem_wait.argtypes = [ctypes.c_void_p]
        _SYNC.libc.sem_wait.restype = ctypes.c_int
        _SYNC.libc.sem_destroy.argtypes = [ctypes.c_void_p]
        _SYNC.libc.sem_destroy.restype = ctypes.c_int
        _SYNC.libc.shm_unlink.argtypes = [ctypes.c_char_p]
        _SYNC.libc.shm_unlink.restype = ctypes.c_int

    def address(self, field_name: str) -> int:
        return self.base + getattr(SharedMemoryLayout, field_name).offset

    def load4(self, field_name: str) -> int:
        return int(_SYNC.load4(self.address(field_name), _MEMORY_ORDER_ACQUIRE))

    def store4(self, field_name: str, value: int) -> None:
        _SYNC.store4(self.address(field_name), value, _MEMORY_ORDER_RELEASE)

    def store8(self, field_name: str, value: int) -> None:
        _SYNC.store8(self.address(field_name), value, _MEMORY_ORDER_RELEASE)

    def compare_phase(self, expected: BarrierPhase, desired: BarrierPhase) -> bool:
        expected_value = ctypes.c_uint32(expected)
        return bool(
            _SYNC.compare_exchange4(
                self.address("phase"),
                ctypes.byref(expected_value),
                desired,
                False,
                _MEMORY_ORDER_ACQ_REL,
                _MEMORY_ORDER_ACQUIRE,
            )
        )

    def begin_generation(self, generation: int) -> None:
        assert self.load4("phase") in (BarrierPhase.UNINITIALIZED, BarrierPhase.AWAIT_STATE)
        ctypes.memset(self.address("state"), 0, ctypes.sizeof(self.layout.state))
        ctypes.memset(self.address("command"), 0, ctypes.sizeof(self.layout.command))
        self.store4("fault_code", FaultCode.NONE)
        self.store8("fault_generation", 0)
        self.store8("fault_sequence", 0)
        self.store8("active_generation", generation)
        self.store4("generation_initialized", 1)
        self.store4("phase", BarrierPhase.AWAIT_STATE)

    def serve(self, count: int, *, corrupt_controller_name: bool = False) -> threading.Thread:
        def run() -> None:
            try:
                for _ in range(count):
                    assert _SYNC.libc.sem_wait(self.address("state_ready")) == 0
                    assert self.load4("phase") == BarrierPhase.STATE_READY
                    state = clone_struct(self.layout.state)
                    validate_state_frame(state)
                    due_mask = 0 if state.header.frame_kind == FrameKind.FENCE else 1
                    command = make_command_frame(state, due_mask)
                    if corrupt_controller_name:
                        command.controller_timing[0].name = b"\xffcorrupt"
                    for index in range(command.header.joint_count):
                        command.joints[index].position = 0.1 * (index + 1)
                    ctypes.memmove(
                        self.address("command"),
                        ctypes.addressof(command),
                        ctypes.sizeof(command),
                    )
                    assert self.compare_phase(BarrierPhase.STATE_READY, BarrierPhase.COMMAND_READY)
                    assert _SYNC.libc.sem_post(self.address("command_ready")) == 0
            except BaseException as error:
                self.thread_errors.append(error)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread

    def wait_for_phase(self, phase: BarrierPhase, timeout_s: float = 1.0) -> None:
        deadline = time.monotonic() + timeout_s
        while self.load4("phase") != phase:
            if time.monotonic() >= deadline:
                raise AssertionError(f"timed out waiting for phase {phase}")
            time.sleep(0.001)

    def close(self) -> None:
        assert _SYNC.libc.sem_destroy(self.address("state_ready")) == 0
        assert _SYNC.libc.sem_destroy(self.address("command_ready")) == 0
        del self.layout
        self.mapping.close()
        os.close(self.descriptor)
        assert _SYNC.libc.shm_unlink(self.name.encode()) == 0


class _QuiescenceCheckingSimulation:
    def __init__(self, barrier: _OwnedTestBarrier):
        self.barrier = barrier
        self._joint_names = (*PANDA_ARM_JOINTS, "panda_finger_joint1", "panda_finger_joint2")
        self.positions = [0.0] * 9
        self.steps = 0
        self.resets = 0

    @property
    def joint_names(self) -> Sequence[str]:
        return self._joint_names

    def read_joint_state(self):
        return self.positions, [0.0] * 9, [0.0] * 9

    def step_position_targets(self, positions_in_abi_order):
        assert self.barrier.load4("phase") == BarrierPhase.COMMAND_READY
        self.positions[:7] = positions_in_abi_order
        self.steps += 1

    def reset_without_physics_step(self):
        self.resets += 1
        self.positions = [0.0] * 9


def test_real_shm_semaphores_and_atomics_hold_quiescence_until_sim_step() -> None:
    barrier = _OwnedTestBarrier()
    client = ArenaBarrierClient(barrier.name)
    try:
        simulation = _QuiescenceCheckingSimulation(barrier)
        manager = ArenaLockstepManager(
            client=client,
            simulation=simulation,
            joint_mapping=make_franka_joint_mapping(simulation.joint_names),
            controller_specs=[ControllerTimingSpec("hold_controller")],
        )
        thread = barrier.serve(2)
        manager.attach_initial_generation()
        manager.fence()
        assert simulation.steps == 0
        manager.physics_step()
        assert simulation.steps == 1
        barrier.wait_for_phase(BarrierPhase.AWAIT_STATE)
        thread.join(timeout=1.0)
        assert not thread.is_alive()
        assert barrier.thread_errors == []
    finally:
        client.close()
        barrier.close()


@pytest.mark.parametrize("frame_kind", [FrameKind.PHYSICS, FrameKind.FENCE])
def test_stale_generation_fails_closed_for_both_frame_kinds(frame_kind: FrameKind) -> None:
    barrier = _OwnedTestBarrier()
    client = ArenaBarrierClient(barrier.name)
    try:
        client.attach_generation(1)
        barrier.begin_generation(2)
        state = make_state_frame(
            generation=1,
            sequence=0,
            physics_tick=0,
            physics_dt_ns=5_000_000,
            frame_kind=frame_kind,
            controller_specs=[ControllerTimingSpec("hold_controller")],
            joints=[JointState(0.0, 0.0, 0.0)],
        )
        with pytest.raises(ProtocolError) as error:
            client.begin_exchange(state)
        assert getattr(error.value, "code", None) == FaultCode.STALE_GENERATION
        assert barrier.load4("fault_code") == FaultCode.STALE_GENERATION
        assert barrier.load4("phase") == BarrierPhase.FAULT
    finally:
        client.close()
        barrier.close()


def test_generation_attach_waits_for_consumer_serviceability_latch() -> None:
    barrier = _OwnedTestBarrier()
    client = ArenaBarrierClient(barrier.name)
    try:
        barrier.store4("wait_interrupted", 1)
        barrier.begin_generation(2)
        with pytest.raises(ProtocolError) as error:
            client.attach_generation(2, timeout_s=0.01)
        assert error.value.code == FaultCode.RESET_VIOLATION

        barrier.store4("wait_interrupted", 0)
        client.attach_generation(2)
        assert client.attached_generation == 2
    finally:
        client.close()
        barrier.close()


def test_corrupt_command_name_bytes_latch_fault_instead_of_wedging() -> None:
    barrier = _OwnedTestBarrier()
    client = ArenaBarrierClient(barrier.name)
    try:
        client.attach_generation(1)
        thread = barrier.serve(1, corrupt_controller_name=True)
        state = make_state_frame(
            generation=1,
            sequence=0,
            physics_tick=0,
            physics_dt_ns=5_000_000,
            frame_kind=FrameKind.PHYSICS,
            controller_specs=[ControllerTimingSpec("hold_controller")],
            joints=[JointState(0.0, 0.0, 0.0)],
        )
        with pytest.raises(ProtocolError) as error:
            client.begin_exchange(state)
        assert error.value.code == FaultCode.TIMING_MISMATCH
        assert barrier.load4("fault_code") == FaultCode.TIMING_MISMATCH
        assert barrier.load4("phase") == BarrierPhase.FAULT
        thread.join(timeout=1.0)
        assert barrier.thread_errors == []
    finally:
        client.close()
        barrier.close()


@pytest.mark.parametrize("failure_point", ["post", "wait"])
def test_native_sync_errors_latch_barrier_fault(monkeypatch, failure_point: str) -> None:
    barrier = _OwnedTestBarrier()
    client = ArenaBarrierClient(barrier.name)
    try:
        client.attach_generation(1)

        def fail(*_args, **_kwargs):
            raise OSError(22, f"injected {failure_point} failure")

        monkeypatch.setattr(client, "_sem_post" if failure_point == "post" else "_sem_timedwait", fail)
        state = make_state_frame(
            generation=1,
            sequence=0,
            physics_tick=0,
            physics_dt_ns=5_000_000,
            frame_kind=FrameKind.PHYSICS,
            controller_specs=[ControllerTimingSpec("hold_controller")],
            joints=[JointState(0.0, 0.0, 0.0)],
        )
        with pytest.raises(ProtocolError) as error:
            client.begin_exchange(state)
        assert error.value.code == FaultCode.BARRIER_STATE_VIOLATION
        assert barrier.load4("fault_code") == FaultCode.BARRIER_STATE_VIOLATION
        assert barrier.load4("phase") == BarrierPhase.FAULT
    finally:
        client.close()
        barrier.close()

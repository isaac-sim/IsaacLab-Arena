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

import isaaclab_arena.integrations.cap_barrier.shared_memory as shared_memory_module
from isaaclab_arena.integrations.cap_barrier.joint_mapping import (
    DROID_FINGER_JOINT,
    DROID_GRIPPER_CLOSED_POSITION_RAD,
    PANDA_ARM_JOINTS,
    make_droid_joint_mapping,
)
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
    ServiceabilityState,
    SharedMemoryLayout,
    StateFrame,
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
    BarrierInterrupted,
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

    def compare_serviceability(self, expected: ServiceabilityState, desired: ServiceabilityState) -> bool:
        expected_value = ctypes.c_uint32(expected)
        return bool(
            _SYNC.compare_exchange4(
                self.address("wait_interrupted"),
                ctypes.byref(expected_value),
                desired,
                False,
                _MEMORY_ORDER_ACQ_REL,
                _MEMORY_ORDER_ACQUIRE,
            )
        )

    def request_interrupt(self) -> None:
        while True:
            state = ServiceabilityState(self.load4("wait_interrupted"))
            if state == ServiceabilityState.SERVICEABLE:
                if self.compare_serviceability(state, ServiceabilityState.INTERRUPTED):
                    return
            elif state == ServiceabilityState.PRODUCER_RESERVED:
                if self.compare_serviceability(state, ServiceabilityState.PRODUCER_RESERVED_INTERRUPT_PENDING):
                    return
            elif state in (
                ServiceabilityState.INTERRUPTED,
                ServiceabilityState.PRODUCER_RESERVED_INTERRUPT_PENDING,
            ):
                return

    def clear_interrupt(self) -> None:
        assert self.compare_serviceability(
            ServiceabilityState.INTERRUPTED,
            ServiceabilityState.SERVICEABLE,
        )

    def begin_generation(self, generation: int) -> None:
        if self.load4("generation_initialized"):
            assert self.load4("wait_interrupted") == ServiceabilityState.INTERRUPTED
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
                    if command.header.joint_count == 8:
                        command.joints[7].position = DROID_GRIPPER_CLOSED_POSITION_RAD
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
        self._joint_names = (*PANDA_ARM_JOINTS, DROID_FINGER_JOINT)
        self.positions = [0.0] * 8
        self.steps = 0
        self.resets = 0

    @property
    def joint_names(self) -> Sequence[str]:
        return self._joint_names

    def read_joint_state(self):
        return self.positions, [0.0] * 8, [0.0] * 8

    def step_position_targets(self, positions_in_abi_order):
        assert self.barrier.load4("phase") == BarrierPhase.COMMAND_READY
        self.positions[:] = positions_in_abi_order
        self.steps += 1

    def reset_without_physics_step(self):
        self.resets += 1
        self.positions = [0.0] * 8


def _state_frame(*, sequence: int = 0, physics_tick: int = 0) -> StateFrame:
    return make_state_frame(
        generation=1,
        sequence=sequence,
        physics_tick=physics_tick,
        physics_dt_ns=5_000_000,
        frame_kind=FrameKind.PHYSICS,
        controller_specs=[ControllerTimingSpec("hold_controller")],
        joints=[JointState(0.0, 0.0, 0.0)],
    )


def test_real_shm_semaphores_and_atomics_hold_quiescence_until_sim_step() -> None:
    barrier = _OwnedTestBarrier()
    client = ArenaBarrierClient(barrier.name)
    try:
        simulation = _QuiescenceCheckingSimulation(barrier)
        manager = ArenaLockstepManager(
            client=client,
            simulation=simulation,
            joint_mapping=make_droid_joint_mapping(simulation.joint_names),
            controller_specs=[ControllerTimingSpec("hold_controller")],
        )
        thread = barrier.serve(2)
        manager.attach_initial_generation()
        manager.fence()
        assert simulation.steps == 0
        manager.physics_step()
        assert simulation.steps == 1
        assert barrier.layout.state.header.joint_count == 8
        assert simulation.positions[7] == pytest.approx(DROID_GRIPPER_CLOSED_POSITION_RAD)
        barrier.wait_for_phase(BarrierPhase.AWAIT_STATE)
        thread.join(timeout=1.0)
        assert not thread.is_alive()
        assert barrier.thread_errors == []
    finally:
        client.close()
        barrier.close()


def test_client_open_and_initialization_share_one_absolute_startup_deadline(monkeypatch) -> None:
    barrier = _OwnedTestBarrier()
    observed_deadlines = []
    original_open = ArenaBarrierClient._open_mapping
    original_wait = ArenaBarrierClient._wait_until_initialized

    def record_open(self, deadline_monotonic_s):
        observed_deadlines.append(deadline_monotonic_s)
        return original_open(self, deadline_monotonic_s)

    def record_wait(self, deadline_monotonic_s):
        observed_deadlines.append(deadline_monotonic_s)
        return original_wait(self, deadline_monotonic_s)

    monkeypatch.setattr(ArenaBarrierClient, "_open_mapping", record_open)
    monkeypatch.setattr(ArenaBarrierClient, "_wait_until_initialized", record_wait)
    deadline = time.monotonic() + 300.0
    client = ArenaBarrierClient(barrier.name, startup_deadline_monotonic_s=deadline)
    try:
        assert observed_deadlines == [deadline, deadline]
    finally:
        client.close()
        barrier.close()


def test_open_mapping_retries_object_visible_before_ftruncate() -> None:
    _OwnedTestBarrier._configure_libc()
    name = f"/cap_pytest_short_{uuid.uuid4().hex}"
    descriptor = _SYNC.libc.shm_open(name.encode(), os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    if descriptor < 0:
        raise OSError(ctypes.get_errno(), "shm_open failed")
    os.ftruncate(descriptor, 0)
    resized = threading.Event()

    def finish_sizing() -> None:
        time.sleep(0.02)
        os.ftruncate(descriptor, ctypes.sizeof(SharedMemoryLayout))
        resized.set()

    thread = threading.Thread(target=finish_sizing, daemon=True)
    thread.start()
    client = ArenaBarrierClient.__new__(ArenaBarrierClient)
    client.name = name
    mapping = None
    try:
        mapping = client._open_mapping(time.monotonic() + 1.0)
        assert resized.wait(timeout=1.0)
        assert len(mapping) == ctypes.sizeof(SharedMemoryLayout)
    finally:
        if mapping is not None:
            mapping.close()
        thread.join(timeout=1.0)
        os.close(descriptor)
        assert _SYNC.libc.shm_unlink(name.encode()) == 0


@pytest.mark.parametrize("frame_kind", [FrameKind.PHYSICS, FrameKind.FENCE])
def test_stale_generation_fails_closed_for_both_frame_kinds(frame_kind: FrameKind) -> None:
    barrier = _OwnedTestBarrier()
    client = ArenaBarrierClient(barrier.name)
    try:
        client.attach_generation(1)
        barrier.request_interrupt()
        barrier.begin_generation(2)
        barrier.clear_interrupt()
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
        barrier.request_interrupt()
        barrier.begin_generation(2)
        interrupted_status = client.status
        assert interrupted_status.active_generation == 2
        assert interrupted_status.phase == BarrierPhase.AWAIT_STATE
        assert not interrupted_status.consumer_serviceable
        assert not interrupted_status.permits_publish(2)
        with pytest.raises(ProtocolError) as error:
            client.attach_generation(2, timeout_s=0.01)
        assert error.value.code == FaultCode.RESET_VIOLATION

        barrier.clear_interrupt()
        client.attach_generation(2)
        assert client.attached_generation == 2
        assert client.status.permits_publish(2)
    finally:
        client.close()
        barrier.close()


def test_startup_attach_retries_interrupted_generation_until_sidecar_is_serviceable() -> None:
    barrier = _OwnedTestBarrier()
    client = ArenaBarrierClient(barrier.name)
    barrier.request_interrupt()

    def activate_sidecar() -> None:
        time.sleep(0.02)
        barrier.clear_interrupt()

    thread = threading.Thread(target=activate_sidecar, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 1.0
        generation = client.wait_for_generation(
            deadline_monotonic_s=deadline,
            startup_rendezvous=True,
        )
        client.attach_generation(
            generation,
            deadline_monotonic_s=deadline,
            startup_rendezvous=True,
        )
        assert client.attached_generation == 1
        assert client.status.consumer_serviceable
    finally:
        thread.join(timeout=1.0)
        client.close()
        barrier.close()


def test_startup_attach_expiry_names_sidecar_activation_not_operational_interrupt() -> None:
    barrier = _OwnedTestBarrier()
    client = ArenaBarrierClient(barrier.name)
    barrier.request_interrupt()
    try:
        with pytest.raises(ProtocolError, match="sidecar never reached active state") as error:
            client.attach_generation(1, timeout_s=0.01, startup_rendezvous=True)
        assert error.value.code == FaultCode.TIMEOUT
        assert "sidecar serviceability" in str(error.value)
        assert client.attached_generation is None
    finally:
        client.close()
        barrier.close()


def test_owner_reset_intent_wins_before_state_copy_without_latching_fault() -> None:
    barrier = _OwnedTestBarrier()
    client = ArenaBarrierClient(barrier.name)
    try:
        client.attach_generation(1)
        ctypes.memset(barrier.address("state"), 0xA5, ctypes.sizeof(StateFrame))
        state_before = ctypes.string_at(barrier.address("state"), ctypes.sizeof(StateFrame))
        barrier.request_interrupt()

        with pytest.raises(BarrierInterrupted):
            client.begin_exchange(_state_frame())

        assert barrier.load4("wait_interrupted") == ServiceabilityState.INTERRUPTED
        assert barrier.load4("phase") == BarrierPhase.AWAIT_STATE
        assert barrier.load4("fault_code") == FaultCode.NONE
        assert ctypes.string_at(barrier.address("state"), ctypes.sizeof(StateFrame)) == state_before
    finally:
        client.close()
        barrier.close()


def test_inflight_exchange_completes_and_releases_pending_interrupt() -> None:
    barrier = _OwnedTestBarrier()
    client = ArenaBarrierClient(barrier.name)
    try:
        client.attach_generation(1)
        thread = barrier.serve(1)
        client.begin_exchange(_state_frame())
        assert barrier.load4("wait_interrupted") == ServiceabilityState.PRODUCER_RESERVED

        barrier.request_interrupt()
        assert barrier.load4("wait_interrupted") == ServiceabilityState.PRODUCER_RESERVED_INTERRUPT_PENDING
        client.complete_exchange()

        assert barrier.load4("wait_interrupted") == ServiceabilityState.INTERRUPTED
        assert barrier.load4("phase") == BarrierPhase.AWAIT_STATE
        assert barrier.load4("fault_code") == FaultCode.NONE
        thread.join(timeout=1.0)
        assert not thread.is_alive()
        assert barrier.thread_errors == []
    finally:
        client.close()
        barrier.close()


def test_pending_interrupt_prevents_next_old_generation_frame_admission() -> None:
    barrier = _OwnedTestBarrier()
    client = ArenaBarrierClient(barrier.name)
    try:
        client.attach_generation(1)
        thread = barrier.serve(1)
        client.begin_exchange(_state_frame())
        barrier.request_interrupt()
        client.complete_exchange()
        shared_state = ctypes.string_at(barrier.address("state"), ctypes.sizeof(StateFrame))

        with pytest.raises(BarrierInterrupted):
            client.begin_exchange(_state_frame(sequence=1, physics_tick=1))

        assert barrier.load4("wait_interrupted") == ServiceabilityState.INTERRUPTED
        assert barrier.load4("fault_code") == FaultCode.NONE
        assert ctypes.string_at(barrier.address("state"), ctypes.sizeof(StateFrame)) == shared_state
        thread.join(timeout=1.0)
        assert not thread.is_alive()
        assert barrier.thread_errors == []
    finally:
        client.close()
        barrier.close()


def test_generation_reset_cannot_race_state_copy_while_producer_reserved(monkeypatch) -> None:
    barrier = _OwnedTestBarrier()
    client = ArenaBarrierClient(barrier.name)
    allow_state_copy = threading.Event()
    state_copy_started = threading.Event()
    producer_errors: list[BaseException] = []
    try:
        client.attach_generation(1)
        server_thread = barrier.serve(1)
        native_memmove = ctypes.memmove
        client_state_address = client._base_address + SharedMemoryLayout.state.offset

        def block_state_copy(destination, source, size):
            if destination == client_state_address:
                assert barrier.load4("wait_interrupted") == ServiceabilityState.PRODUCER_RESERVED
                state_copy_started.set()
                assert allow_state_copy.wait(timeout=1.0)
            return native_memmove(destination, source, size)

        monkeypatch.setattr(shared_memory_module.ctypes, "memmove", block_state_copy)

        def exchange() -> None:
            try:
                client.begin_exchange(_state_frame())
                client.complete_exchange()
            except BaseException as error:
                producer_errors.append(error)

        producer_thread = threading.Thread(target=exchange, daemon=True)
        producer_thread.start()
        assert state_copy_started.wait(timeout=1.0), producer_errors
        barrier.request_interrupt()
        shared_state = ctypes.string_at(barrier.address("state"), ctypes.sizeof(StateFrame))

        with pytest.raises(AssertionError):
            barrier.begin_generation(2)
        assert barrier.load4("active_generation") == 1
        assert ctypes.string_at(barrier.address("state"), ctypes.sizeof(StateFrame)) == shared_state

        allow_state_copy.set()
        producer_thread.join(timeout=1.0)
        assert not producer_thread.is_alive()
        assert producer_errors == []
        assert barrier.load4("wait_interrupted") == ServiceabilityState.INTERRUPTED
        barrier.begin_generation(2)
        assert barrier.load4("active_generation") == 2
        server_thread.join(timeout=1.0)
        assert not server_thread.is_alive()
        assert barrier.thread_errors == []
    finally:
        allow_state_copy.set()
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
        assert barrier.load4("wait_interrupted") == ServiceabilityState.SERVICEABLE
        thread.join(timeout=1.0)
        assert barrier.thread_errors == []
    finally:
        client.close()
        barrier.close()


def test_fault_releases_pending_interrupt_to_interrupted(monkeypatch) -> None:
    barrier = _OwnedTestBarrier()
    client = ArenaBarrierClient(barrier.name)
    try:
        client.attach_generation(1)
        thread = barrier.serve(1, corrupt_controller_name=True)
        native_wait = client._sem_timedwait

        def interrupt_then_wait(field_name: str, timeout_s: float) -> bool:
            barrier.request_interrupt()
            return native_wait(field_name, timeout_s)

        monkeypatch.setattr(client, "_sem_timedwait", interrupt_then_wait)
        with pytest.raises(ProtocolError) as error:
            client.begin_exchange(_state_frame())

        assert error.value.code == FaultCode.TIMING_MISMATCH
        assert barrier.load4("wait_interrupted") == ServiceabilityState.INTERRUPTED
        assert barrier.load4("fault_code") == FaultCode.TIMING_MISMATCH
        thread.join(timeout=1.0)
        assert not thread.is_alive()
        assert barrier.thread_errors == []
    finally:
        client.close()
        barrier.close()


def test_base_exception_during_reserved_exchange_faults_before_release(monkeypatch) -> None:
    class InjectedAbort(BaseException):
        pass

    barrier = _OwnedTestBarrier()
    client = ArenaBarrierClient(barrier.name)
    try:
        client.attach_generation(1)

        def abort_wait(_field_name: str, _timeout_s: float) -> bool:
            raise InjectedAbort

        monkeypatch.setattr(client, "_sem_timedwait", abort_wait)
        with pytest.raises(InjectedAbort):
            client.begin_exchange(_state_frame())

        assert barrier.load4("fault_code") == FaultCode.BARRIER_STATE_VIOLATION
        assert barrier.load4("phase") == BarrierPhase.FAULT
        assert barrier.load4("wait_interrupted") == ServiceabilityState.SERVICEABLE
    finally:
        client.close()
        barrier.close()


@pytest.mark.parametrize(
    ("interrupt_pending", "expected_serviceability"),
    [
        (False, ServiceabilityState.SERVICEABLE),
        (True, ServiceabilityState.INTERRUPTED),
    ],
)
def test_base_exception_after_reservation_cas_recovers_and_releases(
    monkeypatch,
    interrupt_pending: bool,
    expected_serviceability: ServiceabilityState,
) -> None:
    class InjectedAbort(BaseException):
        pass

    barrier = _OwnedTestBarrier()
    client = ArenaBarrierClient(barrier.name)
    try:
        client.attach_generation(1)
        compare_exchange = client._compare_exchange4

        def reserve_then_abort(field_name: str, expected: int, desired: int) -> tuple[bool, int]:
            result = compare_exchange(field_name, expected, desired)
            if field_name == "wait_interrupted" and desired == ServiceabilityState.PRODUCER_RESERVED and result[0]:
                if interrupt_pending:
                    barrier.request_interrupt()
                raise InjectedAbort
            return result

        monkeypatch.setattr(client, "_compare_exchange4", reserve_then_abort)
        with pytest.raises(InjectedAbort):
            client.begin_exchange(_state_frame())

        assert barrier.load4("fault_code") == FaultCode.BARRIER_STATE_VIOLATION
        assert barrier.load4("phase") == BarrierPhase.FAULT
        assert barrier.load4("wait_interrupted") == expected_serviceability
        assert not client._has_producer_reservation
        client.close()
        assert barrier.load4("wait_interrupted") == expected_serviceability
    finally:
        client.close()
        barrier.close()


def test_close_with_pending_command_faults_before_release() -> None:
    barrier = _OwnedTestBarrier()
    client = ArenaBarrierClient(barrier.name)
    try:
        client.attach_generation(1)
        thread = barrier.serve(1)
        client.begin_exchange(_state_frame())
        assert barrier.load4("phase") == BarrierPhase.COMMAND_READY
        assert barrier.load4("wait_interrupted") == ServiceabilityState.PRODUCER_RESERVED

        client.close()

        assert barrier.load4("fault_code") == FaultCode.BARRIER_STATE_VIOLATION
        assert barrier.load4("phase") == BarrierPhase.FAULT
        assert barrier.load4("wait_interrupted") == ServiceabilityState.SERVICEABLE
        thread.join(timeout=1.0)
        assert not thread.is_alive()
        assert barrier.thread_errors == []
    finally:
        client.close()
        barrier.close()


def test_invalid_producer_reservation_release_fails_closed() -> None:
    barrier = _OwnedTestBarrier()
    client = ArenaBarrierClient(barrier.name)
    try:
        client.attach_generation(1)
        thread = barrier.serve(1)
        client.begin_exchange(_state_frame())
        barrier.store4("wait_interrupted", ServiceabilityState.SERVICEABLE)

        with pytest.raises(ProtocolError) as error:
            client.complete_exchange()

        assert error.value.code == FaultCode.BARRIER_STATE_VIOLATION
        assert barrier.load4("fault_code") == FaultCode.BARRIER_STATE_VIOLATION
        assert barrier.load4("phase") == BarrierPhase.FAULT
        thread.join(timeout=1.0)
        assert not thread.is_alive()
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
        assert barrier.load4("wait_interrupted") == ServiceabilityState.SERVICEABLE
    finally:
        client.close()
        barrier.close()

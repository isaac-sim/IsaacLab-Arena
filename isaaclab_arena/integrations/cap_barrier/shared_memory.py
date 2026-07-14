# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Arena-owned endpoint for the CAP POSIX shared-memory barrier."""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import mmap
import os
import time
from dataclasses import dataclass
from types import TracebackType

from .protocol import (
    PROTOCOL_MAJOR,
    PROTOCOL_MINOR,
    SHARED_MEMORY_MAGIC,
    BarrierPhase,
    CommandFrame,
    FaultCode,
    ProtocolError,
    ServiceabilityState,
    SharedMemoryLayout,
    StateFrame,
    clone_struct,
    validate_matching_command,
    validate_state_frame,
)

_MEMORY_ORDER_RELAXED = 0
_MEMORY_ORDER_ACQUIRE = 2
_MEMORY_ORDER_RELEASE = 3
_MEMORY_ORDER_ACQ_REL = 4


class _Timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]


class _NativeSync:
    def __init__(self) -> None:
        atomic_path = ctypes.util.find_library("atomic")
        if atomic_path is None:
            raise RuntimeError("libatomic is required for the CAP shared-memory barrier")
        self.atomic = ctypes.CDLL(atomic_path, use_errno=True)
        self.libc = ctypes.CDLL(None, use_errno=True)

        self.load4 = getattr(self.atomic, "__atomic_load_4")
        self.load4.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.load4.restype = ctypes.c_uint32
        self.load8 = getattr(self.atomic, "__atomic_load_8")
        self.load8.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.load8.restype = ctypes.c_uint64
        self.store4 = getattr(self.atomic, "__atomic_store_4")
        self.store4.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int]
        self.store4.restype = None
        self.store8 = getattr(self.atomic, "__atomic_store_8")
        self.store8.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int]
        self.store8.restype = None
        self.compare_exchange4 = getattr(self.atomic, "__atomic_compare_exchange_4")
        self.compare_exchange4.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32,
            ctypes.c_bool,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.compare_exchange4.restype = ctypes.c_bool

        self.libc.shm_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint]
        self.libc.shm_open.restype = ctypes.c_int
        self.libc.sem_post.argtypes = [ctypes.c_void_p]
        self.libc.sem_post.restype = ctypes.c_int
        self.libc.sem_timedwait.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Timespec)]
        self.libc.sem_timedwait.restype = ctypes.c_int
        self.libc.clock_gettime.argtypes = [ctypes.c_int, ctypes.POINTER(_Timespec)]
        self.libc.clock_gettime.restype = ctypes.c_int


_SYNC = _NativeSync()


@dataclass(frozen=True)
class BarrierStatus:
    """Owner state observed by the Kit producer at a quiescent frame boundary."""

    active_generation: int | None
    phase: BarrierPhase
    serviceability: ServiceabilityState

    @property
    def consumer_serviceable(self) -> bool:
        """Return whether a producer may attempt to reserve the barrier."""
        return self.serviceability == ServiceabilityState.SERVICEABLE

    def permits_publish(self, generation: int) -> bool:
        """Return whether the attached generation may publish its next frame."""
        return (
            self.active_generation == generation
            and self.phase == BarrierPhase.AWAIT_STATE
            and self.serviceability == ServiceabilityState.SERVICEABLE
        )


class BarrierInterrupted(RuntimeError):
    """Reset intent won before the producer reserved its next frame."""


class ArenaBarrierClient:
    """Non-owning Kit-side endpoint; the ROS sidecar creates and unlinks the region."""

    def __init__(self, name: str, *, open_timeout_s: float = 10.0):
        self.name = self._normalize_name(name)
        self._mapping = self._open_mapping(open_timeout_s)
        try:
            self._layout = SharedMemoryLayout.from_buffer(self._mapping)
            self._base_address = ctypes.addressof(self._layout)
            if self._base_address % 64 != 0:
                raise ProtocolError(FaultCode.ABI_MISMATCH, "shared-memory mapping is not 64-byte aligned")
            self._wait_until_initialized(open_timeout_s)
            self._validate_layout()
        except Exception:
            self.close()
            raise
        self._generation: int | None = None
        self._expected_sequence = 0
        self._expected_tick = 0
        self._has_pending_state = False
        self._pending_state: StateFrame | None = None
        self._has_producer_reservation = False

    @staticmethod
    def _normalize_name(name: str) -> str:
        if not name:
            raise ValueError("shared-memory name must not be empty")
        normalized = name if name.startswith("/") else f"/{name}"
        if "/" in normalized[1:]:
            raise ValueError("POSIX shared-memory name must not contain nested slashes")
        return normalized

    def _open_mapping(self, timeout_s: float) -> mmap.mmap:
        deadline = time.monotonic() + timeout_s
        while True:
            descriptor = _SYNC.libc.shm_open(self.name.encode(), os.O_RDWR, 0o600)
            if descriptor >= 0:
                try:
                    status = os.fstat(descriptor)
                    if status.st_size < ctypes.sizeof(SharedMemoryLayout):
                        raise ProtocolError(FaultCode.ABI_MISMATCH, "shared-memory object has the wrong size")
                    return mmap.mmap(
                        descriptor,
                        ctypes.sizeof(SharedMemoryLayout),
                        flags=mmap.MAP_SHARED,
                        prot=mmap.PROT_READ | mmap.PROT_WRITE,
                    )
                finally:
                    os.close(descriptor)
            error = ctypes.get_errno()
            if error != errno.ENOENT or time.monotonic() >= deadline:
                raise OSError(error, os.strerror(error), self.name)
            time.sleep(0.01)

    def _field_address(self, field_name: str) -> int:
        return self._base_address + getattr(SharedMemoryLayout, field_name).offset

    def _load4(self, field_name: str, order: int = _MEMORY_ORDER_ACQUIRE) -> int:
        return int(_SYNC.load4(self._field_address(field_name), order))

    def _load8(self, field_name: str, order: int = _MEMORY_ORDER_ACQUIRE) -> int:
        return int(_SYNC.load8(self._field_address(field_name), order))

    def _store4(self, field_name: str, value: int, order: int = _MEMORY_ORDER_RELEASE) -> None:
        _SYNC.store4(self._field_address(field_name), value, order)

    def _store8(self, field_name: str, value: int, order: int = _MEMORY_ORDER_RELEASE) -> None:
        _SYNC.store8(self._field_address(field_name), value, order)

    def _compare_exchange4(self, field_name: str, expected: int, desired: int) -> tuple[bool, int]:
        expected_value = ctypes.c_uint32(expected)
        exchanged = bool(
            _SYNC.compare_exchange4(
                self._field_address(field_name),
                ctypes.byref(expected_value),
                desired,
                False,
                _MEMORY_ORDER_ACQ_REL,
                _MEMORY_ORDER_ACQUIRE,
            )
        )
        return exchanged, int(expected_value.value)

    def _compare_exchange_phase(self, expected: BarrierPhase, desired: BarrierPhase) -> bool:
        exchanged, _ = self._compare_exchange4("phase", expected, desired)
        return exchanged

    def _wait_until_initialized(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while self._load4("initialized") != 1:
            if time.monotonic() >= deadline:
                raise ProtocolError(FaultCode.TIMEOUT, "timed out waiting for barrier initialization")
            time.sleep(0.001)

    def _validate_layout(self) -> None:
        if (
            self._layout.magic != SHARED_MEMORY_MAGIC
            or self._layout.layout_size != ctypes.sizeof(SharedMemoryLayout)
            or self._layout.protocol_major != PROTOCOL_MAJOR
            or self._layout.protocol_minor > PROTOCOL_MINOR
        ):
            raise ProtocolError(FaultCode.ABI_MISMATCH, "shared-memory barrier ABI mismatch")

    @property
    def active_generation(self) -> int | None:
        if self._load4("generation_initialized") == 0:
            return None
        return self._load8("active_generation")

    @property
    def status(self) -> BarrierStatus:
        """Return a coherent best-effort snapshot of generation serviceability.

        The generation and serviceability word are independent ABI atomics. Reading each
        twice prevents returning a snapshot assembled across an observed transition. The
        owner may still request a transition after this method returns; ``begin_exchange``
        atomically arbitrates that race through the serviceability reservation.
        """
        self._throw_if_faulted()
        while True:
            initialized_before = self._load4("generation_initialized")
            generation_before = self._load8("active_generation") if initialized_before else None
            serviceability_before = self._load4("wait_interrupted")
            phase_value = self._load4("phase")
            serviceability_after = self._load4("wait_interrupted")
            initialized_after = self._load4("generation_initialized")
            generation_after = self._load8("active_generation") if initialized_after else None
            if (
                initialized_before == initialized_after
                and generation_before == generation_after
                and serviceability_before == serviceability_after
            ):
                break
        try:
            phase = BarrierPhase(phase_value)
            serviceability = ServiceabilityState(serviceability_after)
        except ValueError as error:
            raise ProtocolError(
                FaultCode.BARRIER_STATE_VIOLATION,
                "barrier has an invalid phase or serviceability state",
            ) from error
        return BarrierStatus(
            active_generation=generation_after,
            phase=phase,
            serviceability=serviceability,
        )

    @property
    def attached_generation(self) -> int | None:
        return self._generation

    @property
    def expected_sequence(self) -> int:
        return self._expected_sequence

    @property
    def expected_physics_tick(self) -> int:
        return self._expected_tick

    @property
    def fault_code(self) -> FaultCode:
        return FaultCode(self._load4("fault_code"))

    def _throw_if_faulted(self) -> None:
        code = self.fault_code
        if code != FaultCode.NONE:
            generation = self._load8("fault_generation", _MEMORY_ORDER_RELAXED)
            sequence = self._load8("fault_sequence", _MEMORY_ORDER_RELAXED)
            raise ProtocolError(
                code, f"shared-memory barrier fault latched: generation={generation} sequence={sequence}"
            )

    def wait_for_generation(self, *, after: int | None = None, timeout_s: float = 10.0) -> int:
        deadline = time.monotonic() + timeout_s
        while True:
            self._throw_if_faulted()
            generation = self.active_generation
            if generation is not None and (after is None or generation > after):
                return generation
            if time.monotonic() >= deadline:
                raise ProtocolError(FaultCode.TIMEOUT, "timed out waiting for active generation")
            time.sleep(0.001)

    def attach_generation(self, generation: int, *, timeout_s: float = 10.0) -> None:
        deadline = time.monotonic() + timeout_s
        while True:
            self._throw_if_faulted()
            if (
                self.active_generation == generation
                and self._load4("phase") == BarrierPhase.AWAIT_STATE
                and self._load4("wait_interrupted") == ServiceabilityState.SERVICEABLE
            ):
                break
            if time.monotonic() >= deadline:
                raise ProtocolError(FaultCode.RESET_VIOLATION, "cannot attach Arena before generation is serviceable")
            time.sleep(0.001)
        self._generation = generation
        self._expected_sequence = 0
        self._expected_tick = 0
        self._has_pending_state = False
        self._pending_state = None
        self._has_producer_reservation = False

    def _reserve_producer(self) -> None:
        if self._has_producer_reservation:
            raise ProtocolError(FaultCode.BARRIER_STATE_VIOLATION, "producer reservation is already held")
        try:
            exchanged, observed = self._compare_exchange4(
                "wait_interrupted",
                ServiceabilityState.SERVICEABLE,
                ServiceabilityState.PRODUCER_RESERVED,
            )
            if exchanged:
                self._has_producer_reservation = True
                return
        except BaseException:
            # Python may deliver an asynchronous exception after the native CAS succeeds but
            # before local ownership is recorded. The profile has one producer, so an otherwise
            # untracked 2/3 reservation belongs to this endpoint and must be retired normally.
            serviceability = self._load4("wait_interrupted")
            if serviceability in (
                ServiceabilityState.PRODUCER_RESERVED,
                ServiceabilityState.PRODUCER_RESERVED_INTERRUPT_PENDING,
            ):
                self._has_producer_reservation = True
                self._fault_and_release_abandoned_producer()
            raise
        try:
            state = ServiceabilityState(observed)
        except ValueError as error:
            raise ProtocolError(
                FaultCode.BARRIER_STATE_VIOLATION,
                f"invalid serviceability state {observed}",
            ) from error
        if state in (
            ServiceabilityState.INTERRUPTED,
            ServiceabilityState.PRODUCER_RESERVED_INTERRUPT_PENDING,
        ):
            raise BarrierInterrupted("barrier consumer is interrupted for reset")
        raise ProtocolError(
            FaultCode.BARRIER_STATE_VIOLATION,
            f"producer reservation unavailable in state {state.name}",
        )

    def _release_producer(self) -> None:
        if not getattr(self, "_has_producer_reservation", False):
            return
        exchanged, observed = self._compare_exchange4(
            "wait_interrupted",
            ServiceabilityState.PRODUCER_RESERVED,
            ServiceabilityState.SERVICEABLE,
        )
        if not exchanged and observed == ServiceabilityState.PRODUCER_RESERVED_INTERRUPT_PENDING:
            exchanged, observed = self._compare_exchange4(
                "wait_interrupted",
                ServiceabilityState.PRODUCER_RESERVED_INTERRUPT_PENDING,
                ServiceabilityState.INTERRUPTED,
            )
        self._has_producer_reservation = False
        if not exchanged:
            raise ProtocolError(
                FaultCode.BARRIER_STATE_VIOLATION,
                f"producer reservation release observed invalid state {observed}",
            )

    def _fault_and_release_abandoned_producer(self) -> None:
        if not getattr(self, "_has_producer_reservation", False):
            return
        state = getattr(self, "_pending_state", None)
        generation = state.header.generation if state is not None else (self._generation or 0)
        sequence = state.header.sequence if state is not None else self._expected_sequence
        self._latch_fault(FaultCode.BARRIER_STATE_VIOLATION, generation, sequence)
        self._has_pending_state = False
        self._pending_state = None
        self._release_producer()

    def _latch_fault(self, code: FaultCode, generation: int, sequence: int) -> None:
        if code == FaultCode.NONE:
            return
        expected = ctypes.c_uint32(FaultCode.NONE)
        if _SYNC.compare_exchange4(
            self._field_address("fault_code"),
            ctypes.byref(expected),
            code,
            False,
            _MEMORY_ORDER_ACQ_REL,
            _MEMORY_ORDER_ACQUIRE,
        ):
            self._store8("fault_generation", generation, _MEMORY_ORDER_RELAXED)
            self._store8("fault_sequence", sequence, _MEMORY_ORDER_RELAXED)
            self._store4("phase", BarrierPhase.FAULT)
            # Fault latching is best-effort/noexcept in the C++ owner too. A
            # broken semaphore must not replace the original protocol error.
            _SYNC.libc.sem_post(self._sem_address("state_ready"))
            _SYNC.libc.sem_post(self._sem_address("command_ready"))

    def _sem_address(self, field_name: str) -> int:
        return self._field_address(field_name)

    def _sem_post(self, field_name: str) -> None:
        if _SYNC.libc.sem_post(self._sem_address(field_name)) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), field_name)

    def _sem_timedwait(self, field_name: str, timeout_s: float) -> bool:
        deadline = _Timespec()
        if _SYNC.libc.clock_gettime(time.CLOCK_REALTIME, ctypes.byref(deadline)) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), "clock_gettime")
        timeout_ns = int(timeout_s * 1_000_000_000)
        deadline.tv_sec += timeout_ns // 1_000_000_000
        deadline.tv_nsec += timeout_ns % 1_000_000_000
        if deadline.tv_nsec >= 1_000_000_000:
            deadline.tv_sec += 1
            deadline.tv_nsec -= 1_000_000_000
        while _SYNC.libc.sem_timedwait(self._sem_address(field_name), ctypes.byref(deadline)) != 0:
            error = ctypes.get_errno()
            if error == errno.EINTR:
                continue
            if error == errno.ETIMEDOUT:
                return False
            raise OSError(error, os.strerror(error), field_name)
        return True

    def begin_exchange(self, state: StateFrame, *, timeout_s: float = 1.0) -> CommandFrame:
        """Publish state and receive its command without releasing the barrier.

        The caller must apply the command and perform the one PHYSICS step (or
        discard a FENCE command) before calling :meth:`complete_exchange`.
        Keeping ``COMMAND_READY`` asserted until then makes ``AWAIT_STATE`` a
        true simulator-quiescence point for CR-20 generation resets.
        """
        if self._generation is None or self._has_pending_state or self._has_producer_reservation:
            generation = self._generation or state.header.generation
            self._latch_fault(FaultCode.BARRIER_STATE_VIOLATION, generation, self._expected_sequence)
            raise ProtocolError(FaultCode.BARRIER_STATE_VIOLATION, "Arena is not ready to publish state")
        self._throw_if_faulted()
        try:
            validate_state_frame(state)
        except ProtocolError as error:
            self._latch_fault(error.code, state.header.generation, state.header.sequence)
            raise
        if state.header.generation != self._generation:
            self._latch_fault(FaultCode.STALE_GENERATION, state.header.generation, state.header.sequence)
            raise ProtocolError(FaultCode.STALE_GENERATION, "Arena published stale generation")
        if state.header.sequence != self._expected_sequence:
            self._latch_fault(FaultCode.SEQUENCE_MISMATCH, state.header.generation, state.header.sequence)
            raise ProtocolError(FaultCode.SEQUENCE_MISMATCH, "Arena state sequence is not exact-next")
        if state.header.physics_tick != self._expected_tick:
            self._latch_fault(FaultCode.TICK_MISMATCH, state.header.generation, state.header.sequence)
            raise ProtocolError(FaultCode.TICK_MISMATCH, "Arena physics tick is not exact-next")
        try:
            self._reserve_producer()
        except BarrierInterrupted:
            raise
        except ProtocolError as error:
            self._latch_fault(error.code, state.header.generation, state.header.sequence)
            raise
        try:
            self._throw_if_faulted()
            if self.active_generation != self._generation:
                raise ProtocolError(FaultCode.STALE_GENERATION, "barrier generation changed before state publish")

            ctypes.memmove(
                self._base_address + SharedMemoryLayout.state.offset,
                ctypes.addressof(state),
                ctypes.sizeof(StateFrame),
            )
            if not self._compare_exchange_phase(BarrierPhase.AWAIT_STATE, BarrierPhase.STATE_READY):
                raise ProtocolError(FaultCode.BARRIER_STATE_VIOLATION, "cannot publish state in current phase")
            self._sem_post("state_ready")
            self._has_pending_state = True

            if not self._sem_timedwait("command_ready", timeout_s):
                raise ProtocolError(FaultCode.TIMEOUT, "timed out waiting for command frame")
            self._throw_if_faulted()
            if self._load4("phase") != BarrierPhase.COMMAND_READY:
                raise ProtocolError(FaultCode.BARRIER_STATE_VIOLATION, "command semaphore/phase mismatch")
            command = clone_struct(self._layout.command)
            validate_matching_command(state, command)
            self._pending_state = clone_struct(state)
            return command
        except ProtocolError as error:
            self._latch_fault(error.code, state.header.generation, state.header.sequence)
            self._has_pending_state = False
            self._pending_state = None
            self._release_producer()
            raise
        except OSError as error:
            code = FaultCode.BARRIER_STATE_VIOLATION
            self._latch_fault(code, state.header.generation, state.header.sequence)
            self._has_pending_state = False
            self._pending_state = None
            self._release_producer()
            raise ProtocolError(code, f"native barrier synchronization failed: {error}") from error
        except BaseException:
            self._fault_and_release_abandoned_producer()
            raise

    def complete_exchange(self) -> None:
        """Release a received command after Kit has applied or discarded it."""
        if not self._has_pending_state or self._pending_state is None:
            generation = self._generation or 0
            self._latch_fault(FaultCode.BARRIER_STATE_VIOLATION, generation, self._expected_sequence)
            raise ProtocolError(FaultCode.BARRIER_STATE_VIOLATION, "no pending command exchange")
        state = self._pending_state
        if not self._compare_exchange_phase(BarrierPhase.COMMAND_READY, BarrierPhase.AWAIT_STATE):
            self._latch_fault(FaultCode.BARRIER_STATE_VIOLATION, state.header.generation, state.header.sequence)
            self._has_pending_state = False
            self._pending_state = None
            self._release_producer()
            raise ProtocolError(FaultCode.BARRIER_STATE_VIOLATION, "cannot complete command exchange")
        self._expected_sequence += 1
        if state.header.frame_kind == 1:
            self._expected_tick += 1
        self._has_pending_state = False
        self._pending_state = None
        try:
            self._release_producer()
        except ProtocolError as error:
            self._latch_fault(error.code, state.header.generation, state.header.sequence)
            raise

    def fail_pending_exchange(self, code: FaultCode = FaultCode.BARRIER_STATE_VIOLATION) -> None:
        """Latch a protocol fault when Kit cannot finish a received command."""
        state = self._pending_state
        generation = state.header.generation if state is not None else (self._generation or 0)
        sequence = state.header.sequence if state is not None else self._expected_sequence
        self._latch_fault(code, generation, sequence)
        self._has_pending_state = False
        self._pending_state = None
        self._release_producer()

    def close(self) -> None:
        if getattr(self, "_mapping", None) is None:
            return
        try:
            self._fault_and_release_abandoned_producer()
        finally:
            if hasattr(self, "_layout"):
                del self._layout
            self._mapping.close()
            self._mapping = None

    def __enter__(self) -> ArenaBarrierClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

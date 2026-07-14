# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Native-layout mirror of ``cap_backend_arena/protocol.hpp``.

The C++ header in ``isaac_ros_cap`` is the normative ABI. This module is deliberately
limited to the x86-64 Linux/glibc profile used by the Arena and ROS development
environments. Import-time layout assertions make an unsupported ABI fail before a
shared-memory object is touched.
"""

from __future__ import annotations

import ctypes
import platform
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum

PROTOCOL_MAGIC = 0x43415041524E4131
SHARED_MEMORY_MAGIC = 0x43415053484D3031
PROTOCOL_MAJOR = 1
PROTOCOL_MINOR = 0
MAX_CONTROLLERS = 32
MAX_JOINTS = 64
CONTROLLER_NAME_SIZE = 64


class FrameDirection(IntEnum):
    STATE = 1
    COMMAND = 2


class FrameKind(IntEnum):
    PHYSICS = 1
    FENCE = 2


class FaultCode(IntEnum):
    NONE = 0
    ABI_MISMATCH = 1
    INVALID_FRAME = 2
    STALE_GENERATION = 3
    SEQUENCE_MISMATCH = 4
    TICK_MISMATCH = 5
    TIMING_MISMATCH = 6
    BARRIER_STATE_VIOLATION = 7
    TIMEOUT = 8
    CONTROLLER_ERROR = 9
    RESET_VIOLATION = 10


class BarrierPhase(IntEnum):
    UNINITIALIZED = 0
    AWAIT_STATE = 1
    STATE_READY = 2
    COMMAND_READY = 3
    FAULT = 4


class ServiceabilityState(IntEnum):
    """Atomic producer/reset reservation state stored in ``wait_interrupted``."""

    SERVICEABLE = 0
    INTERRUPTED = 1
    PRODUCER_RESERVED = 2
    PRODUCER_RESERVED_INTERRUPT_PENDING = 3


class ProtocolError(RuntimeError):
    """A fail-closed barrier protocol violation."""

    def __init__(self, code: FaultCode, message: str):
        super().__init__(message)
        self.code = code


class ControllerTiming(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * CONTROLLER_NAME_SIZE),
        ("nominal_dt_ns", ctypes.c_uint64),
        ("decimation", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class FrameHeader(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_uint64),
        ("frame_size", ctypes.c_uint32),
        ("protocol_major", ctypes.c_uint16),
        ("protocol_minor", ctypes.c_uint16),
        ("direction", ctypes.c_uint16),
        ("frame_kind", ctypes.c_uint16),
        ("controller_count", ctypes.c_uint32),
        ("joint_count", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("generation", ctypes.c_uint64),
        ("sequence", ctypes.c_uint64),
        ("physics_tick", ctypes.c_uint64),
        ("physics_dt_ns", ctypes.c_uint64),
        ("nominal_controller_due_mask", ctypes.c_uint64),
    ]


class JointStateSample(ctypes.Structure):
    _fields_ = [
        ("position", ctypes.c_double),
        ("velocity", ctypes.c_double),
        ("effort", ctypes.c_double),
    ]


class JointCommandSample(ctypes.Structure):
    _fields_ = [
        ("position", ctypes.c_double),
        ("velocity", ctypes.c_double),
        ("effort", ctypes.c_double),
        ("kp", ctypes.c_double),
        ("kd", ctypes.c_double),
    ]


class StateFrame(ctypes.Structure):
    _fields_ = [
        ("header", FrameHeader),
        ("controller_timing", ControllerTiming * MAX_CONTROLLERS),
        ("joints", JointStateSample * MAX_JOINTS),
    ]


class CommandFrame(ctypes.Structure):
    _fields_ = [
        ("header", FrameHeader),
        ("controller_timing", ControllerTiming * MAX_CONTROLLERS),
        ("joints", JointCommandSample * MAX_JOINTS),
    ]


class PosixSemaphore(ctypes.Structure):
    """glibc x86-64 ``sem_t`` storage; only libc may operate on it."""

    _fields_ = [("storage", ctypes.c_ubyte * 32)]


class SharedMemoryLayout(ctypes.Structure):
    """Mirror of the 64-byte-aligned C++ layout, including its tail padding."""

    _fields_ = [
        ("magic", ctypes.c_uint64),
        ("layout_size", ctypes.c_uint32),
        ("protocol_major", ctypes.c_uint16),
        ("protocol_minor", ctypes.c_uint16),
        ("initialized", ctypes.c_uint32),
        ("phase", ctypes.c_uint32),
        ("fault_code", ctypes.c_uint32),
        ("wait_interrupted", ctypes.c_uint32),
        ("generation_initialized", ctypes.c_uint32),
        ("active_generation", ctypes.c_uint64),
        ("fault_generation", ctypes.c_uint64),
        ("fault_sequence", ctypes.c_uint64),
        ("state_ready", PosixSemaphore),
        ("command_ready", PosixSemaphore),
        ("state", StateFrame),
        ("command", CommandFrame),
        ("_alignas_64_tail_padding", ctypes.c_ubyte * 48),
    ]


@dataclass(frozen=True)
class ControllerTimingSpec:
    name: str
    decimation: int = 1


@dataclass(frozen=True)
class JointState:
    position: float
    velocity: float
    effort: float


EXPECTED_LAYOUT = {
    "ControllerTiming": (80, {"name": 0, "nominal_dt_ns": 64, "decimation": 72, "reserved": 76}),
    "FrameHeader": (
        72,
        {
            "magic": 0,
            "frame_size": 8,
            "protocol_major": 12,
            "protocol_minor": 14,
            "direction": 16,
            "frame_kind": 18,
            "controller_count": 20,
            "joint_count": 24,
            "reserved": 28,
            "generation": 32,
            "sequence": 40,
            "physics_tick": 48,
            "physics_dt_ns": 56,
            "nominal_controller_due_mask": 64,
        },
    ),
    "JointStateSample": (24, {"position": 0, "velocity": 8, "effort": 16}),
    "JointCommandSample": (40, {"position": 0, "velocity": 8, "effort": 16, "kp": 24, "kd": 32}),
    "StateFrame": (4168, {"header": 0, "controller_timing": 72, "joints": 2632}),
    "CommandFrame": (5192, {"header": 0, "controller_timing": 72, "joints": 2632}),
    "SharedMemoryLayout": (
        9536,
        {
            "magic": 0,
            "layout_size": 8,
            "protocol_major": 12,
            "protocol_minor": 14,
            "initialized": 16,
            "phase": 20,
            "fault_code": 24,
            "wait_interrupted": 28,
            "generation_initialized": 32,
            "active_generation": 40,
            "fault_generation": 48,
            "fault_sequence": 56,
            "state_ready": 64,
            "command_ready": 96,
            "state": 128,
            "command": 4296,
        },
    ),
}

EXPECTED_CXX_ALIGNMENTS = {
    "ControllerTiming": 8,
    "FrameHeader": 8,
    "JointStateSample": 8,
    "JointCommandSample": 8,
    "StateFrame": 8,
    "CommandFrame": 8,
    "SharedMemoryLayout": 64,
}


def assert_native_layout() -> None:
    libc_name, _ = platform.libc_ver()
    if (
        sys.byteorder != "little"
        or platform.system() != "Linux"
        or platform.machine() not in ("x86_64", "AMD64")
        or libc_name != "glibc"
        or ctypes.sizeof(ctypes.c_void_p) != 8
        or ctypes.sizeof(ctypes.c_long) != 8
    ):
        raise RuntimeError("CAP Arena barrier supports only little-endian x86-64 Linux/glibc")
    types = {
        "ControllerTiming": ControllerTiming,
        "FrameHeader": FrameHeader,
        "JointStateSample": JointStateSample,
        "JointCommandSample": JointCommandSample,
        "StateFrame": StateFrame,
        "CommandFrame": CommandFrame,
        "SharedMemoryLayout": SharedMemoryLayout,
    }
    for type_name, (expected_size, expected_offsets) in EXPECTED_LAYOUT.items():
        struct_type = types[type_name]
        actual_size = ctypes.sizeof(struct_type)
        if actual_size != expected_size:
            raise RuntimeError(f"{type_name} ABI size mismatch: expected {expected_size}, got {actual_size}")
        for field_name, expected_offset in expected_offsets.items():
            actual_offset = getattr(struct_type, field_name).offset
            if actual_offset != expected_offset:
                raise RuntimeError(
                    f"{type_name}.{field_name} ABI offset mismatch: expected {expected_offset}, got {actual_offset}"
                )
        # ctypes cannot express C++ alignas(64) for externally mapped storage.
        # ArenaBarrierClient checks the mmap base address instead.
        if type_name != "SharedMemoryLayout" and ctypes.alignment(struct_type) != EXPECTED_CXX_ALIGNMENTS[type_name]:
            raise RuntimeError(
                f"{type_name} ABI alignment mismatch: expected {EXPECTED_CXX_ALIGNMENTS[type_name]}, "
                f"got {ctypes.alignment(struct_type)}"
            )


def struct_bytes(value: ctypes.Structure) -> bytes:
    return ctypes.string_at(ctypes.addressof(value), ctypes.sizeof(value))


def clone_struct(value):
    return type(value).from_buffer_copy(struct_bytes(value))


def controller_name(timing: ControllerTiming) -> bytes:
    """Return the bounded opaque name bytes, matching C++ ``string_view`` semantics."""
    raw = bytes(timing.name)
    return raw.split(b"\0", 1)[0]


def make_controller_timing(spec: ControllerTimingSpec, physics_dt_ns: int) -> ControllerTiming:
    encoded = spec.name.encode("utf-8")
    if not encoded or len(encoded) >= CONTROLLER_NAME_SIZE or b"\0" in encoded:
        raise ProtocolError(FaultCode.TIMING_MISMATCH, "controller name is empty, too long, or contains NUL")
    if spec.decimation <= 0 or physics_dt_ns <= 0:
        raise ProtocolError(FaultCode.TIMING_MISMATCH, "controller timing must be positive")
    nominal_dt_ns = physics_dt_ns * spec.decimation
    if nominal_dt_ns > 0xFFFFFFFFFFFFFFFF:
        raise ProtocolError(FaultCode.TIMING_MISMATCH, "controller nominal dt overflows uint64")
    timing = ControllerTiming()
    timing.name = encoded
    timing.nominal_dt_ns = nominal_dt_ns
    timing.decimation = spec.decimation
    return timing


def make_state_frame(
    *,
    generation: int,
    sequence: int,
    physics_tick: int,
    physics_dt_ns: int,
    frame_kind: FrameKind,
    controller_specs: Sequence[ControllerTimingSpec],
    joints: Sequence[JointState],
) -> StateFrame:
    if len(controller_specs) > MAX_CONTROLLERS or len(joints) > MAX_JOINTS:
        raise ProtocolError(FaultCode.INVALID_FRAME, "frame exceeds fixed ABI capacity")
    frame = StateFrame()
    frame.header.magic = PROTOCOL_MAGIC
    frame.header.frame_size = ctypes.sizeof(StateFrame)
    frame.header.protocol_major = PROTOCOL_MAJOR
    frame.header.protocol_minor = PROTOCOL_MINOR
    frame.header.direction = FrameDirection.STATE
    frame.header.frame_kind = frame_kind
    frame.header.controller_count = len(controller_specs)
    frame.header.joint_count = len(joints)
    frame.header.generation = generation
    frame.header.sequence = sequence
    frame.header.physics_tick = physics_tick
    frame.header.physics_dt_ns = physics_dt_ns
    for index, spec in enumerate(controller_specs):
        frame.controller_timing[index] = make_controller_timing(spec, physics_dt_ns)
    for index, sample in enumerate(joints):
        frame.joints[index] = JointStateSample(sample.position, sample.velocity, sample.effort)
    validate_state_frame(frame)
    return frame


def make_command_frame(state: StateFrame, nominal_controller_due_mask: int) -> CommandFrame:
    """Mirror the C++ command-frame constructor for fixtures and fake peers."""
    command = CommandFrame()
    command.header.magic = PROTOCOL_MAGIC
    command.header.frame_size = ctypes.sizeof(CommandFrame)
    command.header.protocol_major = PROTOCOL_MAJOR
    command.header.protocol_minor = PROTOCOL_MINOR
    command.header.direction = FrameDirection.COMMAND
    command.header.frame_kind = state.header.frame_kind
    command.header.controller_count = state.header.controller_count
    command.header.joint_count = state.header.joint_count
    command.header.generation = state.header.generation
    command.header.sequence = state.header.sequence
    command.header.physics_tick = state.header.physics_tick
    command.header.physics_dt_ns = state.header.physics_dt_ns
    command.header.nominal_controller_due_mask = nominal_controller_due_mask
    timing_size = state.header.controller_count * ctypes.sizeof(ControllerTiming)
    ctypes.memmove(
        ctypes.addressof(command.controller_timing),
        ctypes.addressof(state.controller_timing),
        timing_size,
    )
    return command


def _validate_header(frame, expected_direction: FrameDirection) -> None:
    header = frame.header
    if (
        header.magic != PROTOCOL_MAGIC
        or header.protocol_major != PROTOCOL_MAJOR
        or header.protocol_minor > PROTOCOL_MINOR
        or header.frame_size != ctypes.sizeof(frame)
        or header.direction != expected_direction
    ):
        raise ProtocolError(FaultCode.ABI_MISMATCH, "barrier frame ABI mismatch")
    if header.physics_dt_ns == 0:
        raise ProtocolError(FaultCode.INVALID_FRAME, "zero physics_dt in barrier frame")
    if header.frame_kind not in (FrameKind.PHYSICS, FrameKind.FENCE):
        raise ProtocolError(FaultCode.INVALID_FRAME, "unknown barrier frame kind")
    if header.controller_count > MAX_CONTROLLERS or header.joint_count > MAX_JOINTS:
        raise ProtocolError(FaultCode.INVALID_FRAME, "barrier frame exceeds fixed ABI capacity")
    names: set[bytes] = set()
    for index in range(header.controller_count):
        timing = frame.controller_timing[index]
        raw_name = bytes(timing.name)
        name = controller_name(timing)
        if not name or len(raw_name) == CONTROLLER_NAME_SIZE:
            raise ProtocolError(FaultCode.TIMING_MISMATCH, "controller name is empty or unterminated")
        if name in names:
            raise ProtocolError(FaultCode.TIMING_MISMATCH, "duplicate controller timing entry")
        names.add(name)
        if timing.decimation == 0 or timing.nominal_dt_ns != header.physics_dt_ns * timing.decimation:
            raise ProtocolError(FaultCode.TIMING_MISMATCH, "controller timing contract mismatch")


def validate_state_frame(frame: StateFrame) -> None:
    _validate_header(frame, FrameDirection.STATE)
    if frame.header.nominal_controller_due_mask != 0:
        raise ProtocolError(FaultCode.INVALID_FRAME, "state frame has a controller update mask")


def validate_command_frame(frame: CommandFrame) -> None:
    _validate_header(frame, FrameDirection.COMMAND)
    count = frame.header.controller_count
    valid_mask = (1 << count) - 1
    if frame.header.nominal_controller_due_mask & ~valid_mask:
        raise ProtocolError(FaultCode.INVALID_FRAME, "command update mask exceeds controller count")
    if frame.header.frame_kind == FrameKind.FENCE and frame.header.nominal_controller_due_mask != 0:
        raise ProtocolError(FaultCode.INVALID_FRAME, "FENCE command has a controller due mask")


def validate_matching_command(state: StateFrame, command: CommandFrame) -> None:
    validate_state_frame(state)
    validate_command_frame(command)
    state_header = state.header
    command_header = command.header
    matching_fields = (
        "generation",
        "sequence",
        "physics_tick",
        "frame_kind",
        "physics_dt_ns",
        "controller_count",
        "joint_count",
    )
    if any(getattr(state_header, field) != getattr(command_header, field) for field in matching_fields):
        raise ProtocolError(FaultCode.SEQUENCE_MISMATCH, "command does not match pending state frame")
    timing_size = state_header.controller_count * ctypes.sizeof(ControllerTiming)
    if ctypes.string_at(ctypes.addressof(state.controller_timing), timing_size) != ctypes.string_at(
        ctypes.addressof(command.controller_timing), timing_size
    ):
        raise ProtocolError(FaultCode.TIMING_MISMATCH, "command changed controller timing contract")


assert_native_layout()

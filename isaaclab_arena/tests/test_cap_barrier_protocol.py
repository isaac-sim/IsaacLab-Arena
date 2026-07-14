# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import json
from pathlib import Path

import pytest

from isaaclab_arena.integrations.cap_barrier.protocol import (
    CONTROLLER_NAME_SIZE,
    EXPECTED_CXX_ALIGNMENTS,
    EXPECTED_LAYOUT,
    MAX_CONTROLLERS,
    MAX_JOINTS,
    PROTOCOL_MAGIC,
    SHARED_MEMORY_MAGIC,
    BarrierPhase,
    CommandFrame,
    ControllerTimingSpec,
    FaultCode,
    FrameDirection,
    FrameKind,
    JointState,
    ProtocolError,
    ServiceabilityState,
    StateFrame,
    make_command_frame,
    make_state_frame,
    struct_bytes,
    validate_command_frame,
    validate_matching_command,
)

_FIXTURE_DIR = Path(__file__).parent / "test_data" / "cap_barrier"


def _golden_frames() -> tuple[StateFrame, CommandFrame]:
    state = make_state_frame(
        generation=0x0102030405060708,
        sequence=0x1112131415161718,
        physics_tick=0x2122232425262728,
        physics_dt_ns=5_000_000,
        frame_kind=FrameKind.PHYSICS,
        controller_specs=[ControllerTimingSpec(f"controller_{index:02d}", index + 1) for index in range(32)],
        joints=[JointState(index + 0.125, -(index + 0.25), 2.0 * index + 0.5) for index in range(64)],
    )
    command = make_command_frame(state, 0xA5A55A5A)
    for index in range(MAX_JOINTS):
        command.joints[index].position = 100.0 + index + 0.125
        command.joints[index].velocity = -(100.0 + index + 0.25)
        command.joints[index].effort = 200.0 + index + 0.5
        command.joints[index].kp = 300.0 + index + 0.75
        command.joints[index].kd = 400.0 + index + 0.875
    validate_matching_command(state, command)
    return state, command


def test_python_structs_match_cpp_layout_oracle() -> None:
    oracle = json.loads((_FIXTURE_DIR / "abi_layout.json").read_text())
    assert oracle["endianness"] == "little"
    assert int(oracle["protocol"]["magic_hex"], 16) == PROTOCOL_MAGIC
    assert int(oracle["protocol"]["shared_memory_magic_hex"], 16) == SHARED_MEMORY_MAGIC
    assert oracle["protocol"]["max_controllers"] == MAX_CONTROLLERS
    assert oracle["protocol"]["max_joints"] == MAX_JOINTS
    assert oracle["protocol"]["controller_name_size"] == CONTROLLER_NAME_SIZE
    for type_name, (size, offsets) in EXPECTED_LAYOUT.items():
        assert oracle["structs"][type_name]["size"] == size
        assert oracle["structs"][type_name]["offsets"] == offsets
        assert oracle["structs"][type_name]["alignment"] == EXPECTED_CXX_ALIGNMENTS[type_name]
    assert oracle["enum_values"]["frame_direction"] == {"state": 1, "command": 2}
    assert oracle["enum_values"]["frame_kind"] == {"physics": 1, "fence": 2}
    assert oracle["enum_values"]["barrier_phase"] == {
        "uninitialized": BarrierPhase.UNINITIALIZED,
        "await_state": BarrierPhase.AWAIT_STATE,
        "state_ready": BarrierPhase.STATE_READY,
        "command_ready": BarrierPhase.COMMAND_READY,
        "fault": BarrierPhase.FAULT,
    }


def test_python_frames_match_cpp_golden_bytes() -> None:
    state, command = _golden_frames()
    generated = struct_bytes(state) + struct_bytes(command)
    expected = (_FIXTURE_DIR / "abi_golden.bin").read_bytes()
    assert len(generated) == ctypes.sizeof(StateFrame) + ctypes.sizeof(CommandFrame) == 9360
    assert generated == expected


def test_fence_command_must_have_zero_due_mask() -> None:
    state = make_state_frame(
        generation=1,
        sequence=0,
        physics_tick=0,
        physics_dt_ns=5_000_000,
        frame_kind=FrameKind.FENCE,
        controller_specs=[ControllerTimingSpec("hold_controller")],
        joints=[JointState(0.0, 0.0, 0.0)],
    )
    command = make_command_frame(state, 1)
    with pytest.raises(ProtocolError, match="FENCE command") as error:
        validate_command_frame(command)
    assert error.value.code == FaultCode.INVALID_FRAME


def test_matching_command_rejects_timing_mutation() -> None:
    state = make_state_frame(
        generation=7,
        sequence=3,
        physics_tick=2,
        physics_dt_ns=5_000_000,
        frame_kind=FrameKind.PHYSICS,
        controller_specs=[ControllerTimingSpec("joint_streaming_controller")],
        joints=[JointState(0.0, 0.0, 0.0)],
    )
    command = make_command_frame(state, 1)
    command.controller_timing[0].nominal_dt_ns += 1
    with pytest.raises(ProtocolError) as error:
        validate_matching_command(state, command)
    assert error.value.code == FaultCode.TIMING_MISMATCH


def test_protocol_enum_widths_and_values_are_fixed() -> None:
    assert FrameDirection.STATE == 1
    assert FrameDirection.COMMAND == 2
    assert FrameKind.PHYSICS == 1
    assert FrameKind.FENCE == 2
    assert list(ServiceabilityState) == [ServiceabilityState(value) for value in range(4)]
    assert list(FaultCode) == [FaultCode(value) for value in range(11)]

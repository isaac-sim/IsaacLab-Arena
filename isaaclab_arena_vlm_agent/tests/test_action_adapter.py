# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check absolute command rejection without constraining generic adapters."""

import numpy as np
from types import SimpleNamespace

import pytest

from isaaclab_arena_vlm_agent.absolute_action_adapter import AbsoluteChunkActionAdapter
from isaaclab_arena_vlm_agent.command_indices import AbsoluteCommandIndices, GripperCommandIndex, PoseCommandIndices
from isaaclab_arena_vlm_agent.commands import Command
from isaaclab_arena_vlm_agent.response_errors import InvalidAgentResponseError


@pytest.fixture
def pose_indices(monkeypatch):
    indices = AbsoluteCommandIndices(
        8, (PoseCommandIndices(slice(0, 7), "robot", "tool"),), (), (GripperCommandIndex(7, "gripper", 0.0, 1.0),)
    )
    monkeypatch.setattr(
        "isaaclab_arena_vlm_agent.absolute_action_adapter.read_absolute_command_indices", lambda env: indices
    )
    return indices


@pytest.mark.parametrize(
    "values,message",
    [
        ([0] * 7, "Expected 8 values"),
        ([0, 0, 0, 0, 0, 0.71, 0.71, 0], "unit XYZW quaternion"),
        ([0, 0, 0, 0, 0, 0, 1, 0.5], "Value 7 must be"),
        ([float("nan"), 0, 0, 0, 0, 0, 1, 0], "must be finite"),
    ],
)
def test_decode_reports_the_invalid_constraint(pose_indices, values, message):
    adapter = AbsoluteChunkActionAdapter(1)
    assert adapter.is_robot_action_space_supported(None)
    with pytest.raises(InvalidAgentResponseError, match=message):
        adapter.decode_command(adapter.response_model(commands=[Command(values=values)]), None, 0)


@pytest.mark.parametrize("dimension", [7, 29])
def test_joint_chunks_use_active_controller_dimension(monkeypatch, dimension):
    indices = AbsoluteCommandIndices(dimension, (), (slice(0, dimension),), ())
    monkeypatch.setattr(
        "isaaclab_arena_vlm_agent.absolute_action_adapter.read_absolute_command_indices", lambda env: indices
    )
    adapter = AbsoluteChunkActionAdapter(chunk_size=2)
    assert adapter.is_robot_action_space_supported(None)
    response = adapter.response_model(commands=[Command(values=[0.0] * dimension)] * 2)
    command = adapter.decode_command(response, None, 0)
    assert adapter.is_command_valid(command)
    assert command.shape == (2, dimension)
    assert not adapter.is_command_valid(np.full((2, dimension), np.nan))


@pytest.mark.parametrize("binary_values", [(0.0, 1.0), (1.0, -1.0)])
def test_bimanual_pose_and_binary_indices(monkeypatch, binary_values):
    indices = AbsoluteCommandIndices(
        16,
        (PoseCommandIndices(slice(0, 7), "left", "tool"), PoseCommandIndices(slice(8, 15), "right", "tool")),
        (),
        (
            GripperCommandIndex(7, "left_gripper", *binary_values),
            GripperCommandIndex(15, "right_gripper", *binary_values),
        ),
    )
    monkeypatch.setattr(
        "isaaclab_arena_vlm_agent.absolute_action_adapter.read_absolute_command_indices", lambda env: indices
    )
    adapter = AbsoluteChunkActionAdapter(2)
    assert adapter.is_robot_action_space_supported(None)
    row = [0, 0, 0, 0, 0, 0, 1, binary_values[0]] * 2
    response = adapter.response_model(commands=[Command(values=row)] * 2)
    assert adapter.is_command_valid(adapter.decode_command(response, None, 0))
    response.commands[1] = Command(values=row[:11] + [0, 0, 0, 0] + row[15:])
    with pytest.raises(InvalidAgentResponseError, match="Values 11 to 14"):
        adapter.decode_command(response, None, 0)


def test_unsupported_controller_clears_previous_indices(monkeypatch, pose_indices):
    adapter = AbsoluteChunkActionAdapter(1)
    assert adapter.is_robot_action_space_supported(None)
    monkeypatch.setattr(
        "isaaclab_arena_vlm_agent.absolute_action_adapter.read_absolute_command_indices", lambda env: None
    )
    assert not adapter.is_robot_action_space_supported(None)
    with pytest.raises(AssertionError, match="Check controller compatibility"):
        adapter.decode_command(adapter.response_model(commands=[Command(values=[0] * 8)]), None, 0)


def test_chunk_size_is_explicit_and_checked(pose_indices):
    with pytest.raises(TypeError):
        AbsoluteChunkActionAdapter()
    adapter = AbsoluteChunkActionAdapter(2)
    assert adapter.is_robot_action_space_supported(None)
    with pytest.raises(InvalidAgentResponseError, match="Expected 2 commands"):
        adapter.decode_command(SimpleNamespace(commands=[]), None, 0)

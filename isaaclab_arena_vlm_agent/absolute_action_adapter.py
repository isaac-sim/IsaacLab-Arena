# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Concrete adapters for controllers accepting absolute targets directly."""

import numpy as np

from isaaclab_arena_vlm_agent.action_adapter import ChunkActionAdapter, GoalActionAdapter
from isaaclab_arena_vlm_agent.command_indices import AbsoluteCommandIndices, read_absolute_command_indices
from isaaclab_arena_vlm_agent.commands import Command, CommandChunk
from isaaclab_arena_vlm_agent.motion_generator import MotionGenerator
from isaaclab_arena_vlm_agent.response_errors import InvalidAgentResponseError


class AbsoluteChunkActionAdapter(ChunkActionAdapter):
    """Decode absolute command chunks; use chunk_size=1 for one prediction per step."""

    response_model = CommandChunk

    def __init__(self, chunk_size: int):
        super().__init__(chunk_size)
        self._command_indices = None

    def is_robot_action_space_supported(self, env) -> bool:
        """Resolve command indices from the active controller before policy inference."""
        self._command_indices = read_absolute_command_indices(env)
        return self._command_indices is not None

    def decode_command(self, response: CommandChunk, arena_world, env_id: int) -> np.ndarray:
        """Validate and stack absolute commands in execution order."""
        if len(response.commands) != self.chunk_size:
            raise InvalidAgentResponseError(f"Expected {self.chunk_size} commands, got {len(response.commands)}")
        commands = []
        for command in response.commands:
            commands.append(_decode_absolute_command(command, self._command_indices))
        return np.stack(commands)


class AbsoluteGoalActionAdapter(GoalActionAdapter):
    """Decode absolute targets for an explicitly supplied motion generator."""

    response_model = Command

    def __init__(self, motion_generator: MotionGenerator):
        super().__init__(motion_generator)
        self._command_indices = None

    def is_robot_action_space_supported(self, env) -> bool:
        """Resolve command indices from the active controller before policy inference."""
        self._command_indices = read_absolute_command_indices(env)
        return self._command_indices is not None

    def decode_command(self, response: Command, arena_world, env_id: int) -> np.ndarray:
        """Decode a validated response into an absolute target for the motion generator."""
        return _decode_absolute_command(response, self._command_indices)


def _decode_absolute_command(response: Command, indices: AbsoluteCommandIndices | None) -> np.ndarray:
    """Decode native targets, requesting corrections for invalid dimensions or values."""
    assert indices is not None, "Check controller compatibility before decoding absolute commands"
    command = np.asarray(response.values, dtype=np.float32)
    if command.shape != (indices.action_dim,):
        raise InvalidAgentResponseError(f"Expected {indices.action_dim} values, got {command.size}")
    if not np.isfinite(command).all():
        raise InvalidAgentResponseError("All command values must be finite")
    for pose in indices.poses:
        start, stop = pose.indices.start + 3, pose.indices.stop
        if not np.isclose(np.linalg.norm(command[start:stop]), 1.0, atol=1e-3):
            raise InvalidAgentResponseError(f"Values {start} to {stop - 1} must form a unit XYZW quaternion")
    for gripper in indices.grippers:
        if command[gripper.index] not in (gripper.open_value, gripper.close_value):
            raise InvalidAgentResponseError(
                f"Value {gripper.index} must be {gripper.open_value} (open) or {gripper.close_value} (closed)"
            )
    return command

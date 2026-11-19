# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Action contracts for VLM policies."""

from __future__ import annotations

import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import BaseModel

from isaaclab_arena_vlm_agent.motion_generator import MotionGenerator

if TYPE_CHECKING:
    from isaaclab_arena.environments.arena_world import ArenaWorld


class AgentActionAdapter(ABC):
    """Validate model responses and convert commands into robot actions."""

    response_model: type[BaseModel]
    """Response model used to generate the requested schema and validate model output."""

    @abstractmethod
    def is_robot_action_space_supported(self, env) -> bool:
        """Return whether the robot action space supports this adapter's commands.

        Args:
            env: Arena environment, optionally gym-wrapped.
        """

    @abstractmethod
    def decode_command(self, response: BaseModel, arena_world: ArenaWorld, env_id: int) -> np.ndarray:
        """Decode a validated response into commands with adapter-defined dimensions and units.

        Use InvalidAgentResponseError to request a correction for robot-specific constraints.
        Live state supports conversions such as absolute joint targets to joint deltas;
        goal tracking and motion generation belong to MotionGenerator.

        Args:
            response: Model output validated against response_model.
            arena_world: Live Arena scene state and cache derived geometry.
            env_id: Environment index within the batched world state.

        Returns:
            A nonempty, finite command vector, or a chunk as defined by the adapter.
        """

    def is_command_valid(self, command: np.ndarray) -> bool:
        """Return whether a decoded command is a nonempty, finite vector.

        Args:
            command: Decoded command to validate before execution.
        """
        return bool(command.ndim == 1 and command.size > 0 and np.isfinite(command).all())

    def command_to_action(self, env, command: np.ndarray) -> torch.Tensor:
        """Convert a command into one native action on the environment device.

        Args:
            env: Arena environment, optionally gym-wrapped.
            command: One validated action row for one environment.

        Returns:
            One-dimensional float32 action tensor; override for other action conventions.
        """
        return torch.as_tensor(command, dtype=torch.float32, device=env.unwrapped.device)


class ChunkActionAdapter(AgentActionAdapter):
    """Decode fixed-length command sequences for VLMChunkPolicy."""

    def __init__(self, chunk_size: int):
        assert type(chunk_size) is int and chunk_size > 0, "chunk_size must be a positive integer"
        self.chunk_size = chunk_size

    def is_command_valid(self, command: np.ndarray) -> bool:
        """Return whether a decoded chunk has the required shape and finite values.

        Args:
            command: Decoded array shaped (chunk_size, command_dim).
        """
        return bool(
            command.ndim == 2
            and command.shape[0] == self.chunk_size
            and command.shape[1] > 0
            and np.isfinite(command).all()
        )


class GoalActionAdapter(AgentActionAdapter):
    """Decode and validate goals for VLMGoalPolicy."""

    def __init__(self, motion_generator: MotionGenerator):
        assert isinstance(motion_generator, MotionGenerator)
        self.motion_generator = motion_generator
        """Generate commands for this adapter's decoded goals."""

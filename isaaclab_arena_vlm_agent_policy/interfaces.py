# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Contracts for robot adapters and low-level command executors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

from isaaclab_arena_vlm_agent_policy.commands import AgentCommand

if TYPE_CHECKING:
    import torch

CommandT = TypeVar("CommandT", bound=AgentCommand)
ReferenceT = TypeVar("ReferenceT")


@dataclass(frozen=True)
class CameraInput:
    """Pair an encoded image URL with calibration matching its size, crop, and capture time."""

    key: str
    image_url: str
    width: int
    height: int
    calibration: dict[str, Any] | None = None


@dataclass(frozen=True)
class DecisionInput:
    """Collect one environment's synchronized model-facing inputs."""

    robot_state: dict[str, Any]
    cameras: tuple[CameraInput, ...] = ()


@dataclass(frozen=True)
class ExecutionFeedback:
    """Describe measured execution status without assuming task success."""

    status: Literal["running", "succeeded", "timed_out", "tracking_failed", "cancelled"]
    elapsed_steps: int
    reason: str | None = None
    measurements: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionStep(Generic[ReferenceT]):
    """Pair feedback with a control reference, or None when execution has ended."""

    reference: ReferenceT | None
    feedback: ExecutionFeedback


class VLMObservationAdapter(ABC):
    """Extract model-facing inputs and lightweight state for one environment."""

    @abstractmethod
    def extract_decision_input(self, env: Any, observation: dict, env_id: int) -> DecisionInput:
        """Read images, robot state, and optional camera calibration."""

    def extract_tracking_state(self, env: Any, observation: dict, env_id: int) -> dict[str, Any]:
        """Default to decision state; override to avoid image processing between decisions."""
        return self.extract_decision_input(env, observation, env_id).robot_state


class AgentActionAdapter(ABC, Generic[CommandT, ReferenceT]):
    """Validate commands and encode references for a bound embodiment."""

    @abstractmethod
    def bind(self, env: Any) -> None:
        """Validate action layout and frames against env.unwrapped before first use."""

    @property
    @abstractmethod
    def response_schema(self) -> dict[str, Any]:
        """Declare the model's command schema."""

    @abstractmethod
    def decode_command(self, data: dict[str, Any], measured_state: dict[str, Any]) -> CommandT:
        """Validate schema, frames, workspace, and budgets before accepting a command."""

    @abstractmethod
    def command_to_action(self, env: Any, env_id: int, reference: ReferenceT) -> torch.Tensor:
        """Encode one action row; the policy stacks rows on the environment's device."""


class AgentCommandExecutor(ABC, Generic[CommandT, ReferenceT]):
    """Own per-environment references, tracking, and completion for accepted commands."""

    @abstractmethod
    def start(self, env_id: int, command: CommandT, measured_state: dict[str, Any]) -> None:
        """Initialize execution from measured state."""

    @abstractmethod
    def advance(self, env_id: int, measured_state: dict[str, Any], step_dt: float) -> ExecutionStep[ReferenceT]:
        """Check previous execution and produce the reference for the next env.step()."""

    @abstractmethod
    def reset(self, env_ids: list[int] | None = None) -> None:
        """Clear selected environments, or all environments when env_ids is None."""

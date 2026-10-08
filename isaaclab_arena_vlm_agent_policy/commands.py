# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Model-facing pose commands, independent of robot action layouts."""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AgentCommand(BaseModel):
    """Base for validated commands; adapters may define their own command subclasses."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


CommandScalar = Annotated[float, Field(strict=True)]
PositionVector = Annotated[tuple[CommandScalar, ...], Field(min_length=3, max_length=3)]
QuaternionXYZW = Annotated[tuple[CommandScalar, ...], Field(min_length=4, max_length=4)]


class PoseTarget(AgentCommand):
    """Describe T_A_B: map controlled frame B into reference frame A, using meters and XYZW."""

    reference_frame: str = Field(min_length=1)
    controlled_frame: str = Field(min_length=1)
    position_m: PositionVector
    quaternion_xyzw: QuaternionXYZW

    @field_validator("quaternion_xyzw")
    @classmethod
    def validate_quaternion(cls, value):
        """Require a unit quaternion rather than silently changing the model's command."""
        if not math.isclose(math.hypot(*value), 1.0, abs_tol=1e-3):
            raise ValueError("quaternion_xyzw must have unit length")
        return value


GripperTarget = Annotated[float, Field(strict=True, ge=0.0, le=1.0)]
"""Normalized opening: zero is closed and one is open; adapters map to controller units."""

StepBudget = Annotated[int, Field(strict=True, gt=0)]
"""Count calls to env.step(), not physics substeps."""


class MoveToCommand(AgentCommand):
    """Reach an absolute pose within a bounded number of policy steps."""

    kind: Literal["move_to"] = "move_to"
    target: PoseTarget
    gripper: GripperTarget | None = None
    """None preserves the active gripper target."""

    max_steps: StepBudget


class SetGripperCommand(AgentCommand):
    """Set the gripper opening while holding the arm reference."""

    kind: Literal["set_gripper"] = "set_gripper"
    gripper: GripperTarget
    max_steps: StepBudget


class WaitCommand(AgentCommand):
    """Maintain the current reference for a bounded duration."""

    kind: Literal["wait"] = "wait"
    steps: StepBudget


class PoseReference(AgentCommand):
    """Specify an absolute pose and optional gripper opening for chunk execution."""

    target: PoseTarget
    gripper: GripperTarget | None = None
    duration_steps: StepBudget = 1


class ActionChunkCommand(AgentCommand):
    """Execute ordered references; adapters enforce configured horizon and motion limits."""

    kind: Literal["action_chunk"] = "action_chunk"
    references: tuple[PoseReference, ...] = Field(min_length=1)


GoalCommand = Annotated[MoveToCommand | SetGripperCommand | WaitCommand, Field(discriminator="kind")]

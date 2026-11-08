# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Load and validate object-control recipes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, TypeAdapter, ValidationError, model_validator

NonEmptyString = Annotated[str, Field(min_length=1)]
NonNegativeFloat = Annotated[FiniteFloat, Field(ge=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
NormalizedFloat = Annotated[FiniteFloat, Field(ge=0, le=1)]
Vector3 = Annotated[list[FiniteFloat], Field(min_length=3, max_length=3)]
Quaternion = Annotated[list[FiniteFloat], Field(min_length=4, max_length=4)]


class Axis(str, Enum):
    """World-frame axis used by a reference-relative target."""

    X = "x"
    Y = "y"
    Z = "z"


class ReferencePoint(str, Enum):
    """Point on a referenced object used to construct a move target."""

    POSITION = "position"
    AABB_CENTER = "aabb_center"


class RotationMode(str, Enum):
    """Rotation-controller command representation."""

    ACCELERATION = "acceleration"
    TORQUE = "torque"


class MarkBoundary(str, Enum):
    """Recipe milestone boundary."""

    START = "start"
    END = "end"


AxisValue = Annotated[Axis, Field(strict=False)]
ReferencePointValue = Annotated[ReferencePoint, Field(strict=False)]
RotationModeValue = Annotated[RotationMode, Field(strict=False)]
MarkBoundaryValue = Annotated[MarkBoundary, Field(strict=False)]


class StrictRecipeModel(BaseModel):
    """Forbid unknown fields and implicit type coercion in recipe data."""

    model_config = ConfigDict(extra="forbid", strict=True)


class CommandBase(StrictRecipeModel):
    """Strict base model for one recipe command."""


class SettleHold(StrictRecipeModel):
    """Effort hold applied at a joint's reset position."""

    gain: NonNegativeFloat
    damping: NonNegativeFloat
    limit: NonNegativeFloat


class SetupJointCommand(CommandBase):
    """Configure one effort-controlled articulation joint."""

    action: Literal["setup_joint"] = "setup_joint"
    name: NonEmptyString
    joint_name: NonEmptyString
    effort_limit: NonNegativeFloat = 100.0
    replace_existing: bool = False
    settle_hold: SettleHold | None = None
    drive_stiffness: NonNegativeFloat = 0.0
    drive_damping: NonNegativeFloat = 0.0


class PoseControlCommand(CommandBase):
    """Shared bounded pose-controller settings."""

    seconds: NonNegativeFloat = 2.0
    linear_gain: NonNegativeFloat | None = None
    linear_damping: NonNegativeFloat | None = None
    critical_damping: bool | None = None
    minimum_stiffness: NonNegativeFloat | None = None
    max_force: NonNegativeFloat | None = None
    rotation_mode: RotationModeValue | None = None
    rotation_gain: NonNegativeFloat | None = None
    angular_damping: NonNegativeFloat | None = None
    max_torque: NonNegativeFloat | None = None


class MoveCommand(PoseControlCommand):
    """Drive an object's pose toward an absolute or reference-relative target."""

    action: Literal["move"] = "move"
    name: NonEmptyString
    goal: Vector3 | None = None
    reference: NonEmptyString | None = None
    reference_point: ReferencePointValue = ReferencePoint.POSITION
    reference_axes: list[AxisValue] | None = None
    offset: Vector3 = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    orientation: Quaternion | None = None

    @model_validator(mode="after")
    def validate_target(self) -> MoveCommand:
        """Validate cross-field target requirements."""
        if self.goal is None and self.reference is None:
            raise ValueError("move requires 'goal' or 'reference'")
        if self.reference is None and (
            self.offset != [0.0, 0.0, 0.0] or self.reference_point is not ReferencePoint.POSITION
        ):
            raise ValueError("'offset' and 'reference_point' require 'reference'")
        if self.goal is not None and self.reference is not None and self.reference_axes is None:
            raise ValueError("'goal' with 'reference' requires 'reference_axes'")
        if self.reference_axes is not None:
            if self.reference is None:
                raise ValueError("'reference_axes' requires 'reference'")
            if not self.reference_axes or len(self.reference_axes) != len(set(self.reference_axes)):
                raise ValueError("'reference_axes' must contain unique x, y, or z values")
        if self.orientation is not None and not any(self.orientation):
            raise ValueError("'orientation' must be nonzero")
        return self


class LiftCommand(PoseControlCommand):
    """Drive an object's world-frame height while preserving its x-y position."""

    action: Literal["lift"] = "lift"
    name: NonEmptyString
    height: FiniteFloat


class JointCommand(CommandBase):
    """Drive one articulation joint toward an absolute or normalized target."""

    action: Literal["joint"] = "joint"
    name: NonEmptyString
    joint_name: NonEmptyString
    goal: FiniteFloat | None = None
    normalized_goal: NormalizedFloat | None = None
    seconds: NonNegativeFloat = 2.0
    gain: NonNegativeFloat = 40.0
    damping: NonNegativeFloat = 6.0
    limit: NonNegativeFloat = 30.0

    @model_validator(mode="after")
    def validate_goal(self) -> JointCommand:
        """Require exactly one joint target representation."""
        if (self.goal is None) == (self.normalized_goal is None):
            raise ValueError("joint requires exactly one of 'goal' and 'normalized_goal'")
        return self


class ReleaseCommand(CommandBase):
    """Release persistent pose or joint control."""

    action: Literal["release"] = "release"
    name: NonEmptyString
    joint_name: NonEmptyString | None = None


class WaitCommand(CommandBase):
    """Advance a physics-step or simulation-time budget."""

    action: Literal["wait"] = "wait"
    steps: NonNegativeInt | None = None
    seconds: NonNegativeFloat | None = None

    @model_validator(mode="after")
    def validate_budget(self) -> WaitCommand:
        """Require exactly one wait budget representation."""
        if (self.steps is None) == (self.seconds is None):
            raise ValueError("wait requires exactly one of 'steps' and 'seconds'")
        return self


class MarkCommand(CommandBase):
    """Record a named subtask boundary."""

    action: Literal["mark"] = "mark"
    index: NonNegativeInt
    boundary: MarkBoundaryValue
    title: NonEmptyString


class CheckCommand(CommandBase):
    """Wait for one subtask completion condition."""

    action: Literal["check"] = "check"
    index: NonNegativeInt
    timeout_steps: NonNegativeInt = 600


class WaitFinalCommand(CommandBase):
    """Wait for the overall task success condition."""

    action: Literal["wait_final"] = "wait_final"
    timeout_steps: NonNegativeInt = 1200


class InspectCommand(CommandBase):
    """Print the current physical and task state."""

    action: Literal["inspect"] = "inspect"


class FinishCommand(CommandBase):
    """Stop consuming commands."""

    action: Literal["finish"] = "finish"


RuntimeCommand = Annotated[
    MoveCommand
    | LiftCommand
    | JointCommand
    | ReleaseCommand
    | WaitCommand
    | MarkCommand
    | CheckCommand
    | WaitFinalCommand
    | InspectCommand
    | FinishCommand,
    Field(discriminator="action"),
]
ObjectControlCommand = Annotated[SetupJointCommand | RuntimeCommand, Field(discriminator="action")]
_COMMAND_ADAPTER = TypeAdapter(ObjectControlCommand)


@dataclass(frozen=True)
class ObjectControlRecipe:
    """Validated setup and runtime commands loaded from JSONL."""

    joint_setups: tuple[SetupJointCommand, ...]
    commands: tuple[RuntimeCommand, ...]


def load_object_control_recipe(path: Path, require_runtime_commands: bool = True) -> ObjectControlRecipe:
    """Load and validate an object-control JSONL recipe."""
    joint_setups = []
    commands = []
    runtime_command_seen = False
    for line_number, line in enumerate(read_recipe_lines(path, complete_only=not require_runtime_commands), 1):
        if not line.strip():
            continue
        command = parse_object_control_line(line, line_number)
        if isinstance(command, SetupJointCommand):
            assert (
                not runtime_command_seen
            ), f"Recipe line {line_number}: setup_joint commands must precede runtime commands"
            joint_setups.append(command)
        else:
            runtime_command_seen = True
            commands.append(command)
    assert commands or not require_runtime_commands, f"Recipe '{path}' has no runtime commands"
    return ObjectControlRecipe(tuple(joint_setups), tuple(commands))


def read_recipe_lines(path: Path, complete_only: bool = False) -> list[str]:
    """Read JSONL records, withholding an unterminated append in follow mode."""
    content = path.read_bytes()
    if complete_only:
        content = content[: content.rfind(b"\n") + 1]
    return content.decode("utf-8").splitlines(keepends=True)


def parse_object_control_line(line: str, line_number: int) -> ObjectControlCommand:
    """Decode and validate one JSONL command."""
    try:
        command = json.loads(line)
    except json.JSONDecodeError as error:
        raise AssertionError(f"Recipe line {line_number}: invalid JSON: {error.msg}") from error
    return validate_object_control_command(command, line_number)


def validate_object_control_command(command: object, line_number: int) -> ObjectControlCommand:
    """Validate and return one typed JSONL command."""
    try:
        return _COMMAND_ADAPTER.validate_python(command)
    except ValidationError as error:
        action = command.get("action") if isinstance(command, dict) else None
        details = []
        for item in error.errors(include_url=False):
            location = ".".join(str(part) for part in item["loc"] if part != action)
            prefix = f"'{location}': " if location else ""
            details.append(prefix + item["msg"])
        raise AssertionError(f"Recipe line {line_number}: {'; '.join(details)}") from None

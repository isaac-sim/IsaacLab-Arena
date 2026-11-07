# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Load and validate physical force-control recipes."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_VECTOR_FIELDS = {"offset", "orientation"}
_NON_NEGATIVE_NUMBER_FIELDS = {
    "seconds",
    "linear_gain",
    "linear_damping",
    "minimum_stiffness",
    "max_force",
    "rotation_gain",
    "angular_damping",
    "max_torque",
    "gain",
    "damping",
    "limit",
    "effort_limit",
    "drive_stiffness",
    "drive_damping",
}
_COMMAND_FIELDS = {
    "setup_joint": (
        {"name", "joint_name"},
        {"effort_limit", "replace_existing", "settle_hold", "drive_stiffness", "drive_damping"},
    ),
    "move": (
        {"name"},
        {
            "goal",
            "reference",
            "reference_point",
            "reference_axes",
            "offset",
            "orientation",
            "seconds",
            "linear_gain",
            "linear_damping",
            "minimum_stiffness",
            "max_force",
            "rotation_mode",
            "rotation_gain",
            "angular_damping",
            "max_torque",
        },
    ),
    "lift": (
        {"name", "height"},
        {
            "seconds",
            "linear_gain",
            "linear_damping",
            "minimum_stiffness",
            "max_force",
            "rotation_mode",
            "rotation_gain",
            "angular_damping",
            "max_torque",
        },
    ),
    "joint": (
        {"name", "joint_name"},
        {"goal", "normalized_goal", "seconds", "gain", "damping", "limit"},
    ),
    "release": ({"name"}, set()),
    "wait": (set(), {"steps", "seconds"}),
    "mark": ({"index", "boundary", "title"}, set()),
    "check": ({"index"}, {"timeout_steps"}),
    "wait_final": (set(), {"timeout_steps"}),
    "inspect": (set(), set()),
    "finish": (set(), set()),
}


@dataclass(frozen=True)
class ForceControlRecipe:
    """Validated setup and runtime commands loaded from JSONL."""

    joint_setups: tuple[dict[str, Any], ...]
    commands: tuple[dict[str, Any], ...]


def load_force_control_recipe(path: Path, require_runtime_commands: bool = True) -> ForceControlRecipe:
    """Load and validate a force-control JSONL recipe."""
    joint_setups = []
    commands = []
    runtime_command_seen = False
    for line_number, line in enumerate(read_recipe_lines(path, complete_only=not require_runtime_commands), 1):
        if not line.strip():
            continue
        try:
            raw_command = json.loads(line)
        except json.JSONDecodeError as error:
            raise AssertionError(f"Recipe line {line_number}: invalid JSON: {error.msg}") from error
        command = validate_force_control_command(raw_command, line_number)
        if command["action"] == "setup_joint":
            assert (
                not runtime_command_seen
            ), f"Recipe line {line_number}: setup_joint commands must precede runtime commands"
            joint_setups.append(command)
        else:
            runtime_command_seen = True
            commands.append(command)
    assert commands or not require_runtime_commands, f"Recipe '{path}' has no runtime commands"
    return ForceControlRecipe(tuple(joint_setups), tuple(commands))


def read_recipe_lines(path: Path, complete_only: bool = False) -> list[str]:
    """Read JSONL records, withholding an unterminated append in follow mode."""
    content = path.read_bytes()
    if complete_only:
        content = content[: content.rfind(b"\n") + 1]
    return content.decode("utf-8").splitlines(keepends=True)


def validate_force_control_command(command: Any, line_number: int) -> dict[str, Any]:
    """Validate and return one JSONL command."""
    assert isinstance(command, dict), f"Recipe line {line_number}: each line must contain a JSON object"
    assert isinstance(command.get("action"), str), f"Recipe line {line_number}: missing string field 'action'"
    action = command["action"]
    assert action in _COMMAND_FIELDS, f"Recipe line {line_number}: unknown action '{action}'"
    required, optional = _COMMAND_FIELDS[action]
    fields = set(command) - {"action"}
    missing = required - fields
    unknown = fields - required - optional
    assert not missing, f"Recipe line {line_number}: action '{action}' is missing {sorted(missing)}"
    assert not unknown, f"Recipe line {line_number}: action '{action}' has unknown fields {sorted(unknown)}"

    for field in ("name", "joint_name", "reference", "reference_point", "rotation_mode", "title", "boundary"):
        if field in command:
            assert (
                isinstance(command[field], str) and command[field]
            ), f"Recipe line {line_number}: '{field}' must be a non-empty string"
    for field in (
        "height",
        "normalized_goal",
        "seconds",
        "linear_gain",
        "linear_damping",
        "minimum_stiffness",
        "max_force",
        "rotation_gain",
        "angular_damping",
        "max_torque",
        "gain",
        "damping",
        "limit",
        "effort_limit",
        "drive_stiffness",
        "drive_damping",
    ):
        if field in command:
            _assert_number(command[field], field, line_number)
            if field in _NON_NEGATIVE_NUMBER_FIELDS:
                assert command[field] >= 0, f"Recipe line {line_number}: '{field}' must be non-negative"
    for field in ("index", "steps", "timeout_steps"):
        if field in command:
            assert (
                isinstance(command[field], int) and command[field] >= 0
            ), f"Recipe line {line_number}: '{field}' must be a non-negative integer"
    for field in _VECTOR_FIELDS & set(command):
        expected_length = 4 if field == "orientation" else 3
        value = command[field]
        assert (
            isinstance(value, list) and len(value) == expected_length
        ), f"Recipe line {line_number}: '{field}' must contain {expected_length} numbers"
        for component in value:
            _assert_number(component, field, line_number)
        if field == "orientation":
            assert any(value), f"Recipe line {line_number}: 'orientation' must be nonzero"
    if "reference_axes" in command:
        axes = command["reference_axes"]
        assert isinstance(axes, list) and axes, f"Recipe line {line_number}: 'reference_axes' must be a non-empty list"
        assert len(axes) == len(set(axes)) and set(axes) <= {
            "x",
            "y",
            "z",
        }, f"Recipe line {line_number}: 'reference_axes' must contain unique x, y, or z values"
    if action == "move":
        assert (
            "goal" in command or "reference" in command
        ), f"Recipe line {line_number}: move requires 'goal' or 'reference'"
        if "goal" in command:
            goal = command["goal"]
            assert (
                isinstance(goal, list) and len(goal) == 3
            ), f"Recipe line {line_number}: move 'goal' must contain 3 numbers"
            for component in goal:
                _assert_number(component, "goal", line_number)
        assert (
            "reference_axes" not in command or "reference" in command
        ), f"Recipe line {line_number}: 'reference_axes' requires 'reference'"
        if "reference_point" in command:
            assert command["reference_point"] in {
                "position",
                "aabb_center",
            }, f"Recipe line {line_number}: 'reference_point' must be 'position' or 'aabb_center'"
    if action == "joint":
        assert ("goal" in command) != (
            "normalized_goal" in command
        ), f"Recipe line {line_number}: joint requires exactly one of 'goal' and 'normalized_goal'"
        if "goal" in command:
            _assert_number(command["goal"], "goal", line_number)
        if "normalized_goal" in command:
            assert (
                0.0 <= command["normalized_goal"] <= 1.0
            ), f"Recipe line {line_number}: 'normalized_goal' must be between 0 and 1"
    if action == "wait":
        assert ("steps" in command) != (
            "seconds" in command
        ), f"Recipe line {line_number}: wait requires exactly one of 'steps' and 'seconds'"
    if "rotation_mode" in command:
        assert command["rotation_mode"] in {
            "acceleration",
            "torque",
        }, f"Recipe line {line_number}: 'rotation_mode' must be 'acceleration' or 'torque'"
    if "replace_existing" in command:
        assert isinstance(
            command["replace_existing"], bool
        ), f"Recipe line {line_number}: 'replace_existing' must be a boolean"
    if "settle_hold" in command:
        hold = command["settle_hold"]
        assert isinstance(hold, dict) and set(hold) == {
            "gain",
            "damping",
            "limit",
        }, f"Recipe line {line_number}: 'settle_hold' requires exactly gain, damping, and limit"
        for field, value in hold.items():
            _assert_number(value, f"settle_hold.{field}", line_number)
            assert value >= 0, f"Recipe line {line_number}: 'settle_hold.{field}' must be non-negative"
    return command


def _assert_number(value: Any, field: str, line_number: int) -> None:
    assert (
        isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)
    ), f"Recipe line {line_number}: '{field}' must be a finite number"

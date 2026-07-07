# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Validation helpers for agent-generated environment graph specs."""

from __future__ import annotations

import inspect
from typing import Any

from pydantic import ValidationError

from isaaclab_arena.assets.registries import TaskRegistry
from isaaclab_arena.environments.arena_env_graph_spec import ArenaEnvGraphSpec


def required_task_init_param_names(task_cls: type) -> list[str]:
    """Return required ``__init__`` parameter names for a task class."""
    sig = inspect.signature(task_cls.__init__)
    required: list[str] = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return required


_ASSERTION_FAILED_PREFIX = "Assertion failed, "


def _clean_validation_msg(msg: str) -> str:
    """Strip pydantic's assertion wrapper from validator error text."""
    if msg.startswith(_ASSERTION_FAILED_PREFIX):
        return msg[len(_ASSERTION_FAILED_PREFIX) :]
    return msg


def format_validation_error(exc: ValidationError) -> list[str]:
    """Flatten a Pydantic ``ValidationError`` into human-readable trace lines."""
    lines: list[str] = []
    for err in exc.errors():
        msg = _clean_validation_msg(err["msg"])
        loc = ".".join(str(part) for part in err["loc"])
        lines.append(f"{loc}: {msg}" if loc else msg)
    return lines


def try_parse_env_graph_spec(data: dict[str, Any]) -> tuple[ArenaEnvGraphSpec | None, list[str]]:
    """Parse agent output into an ``ArenaEnvGraphSpec`` without raising.

    Args:
        data: Parsed JSON object from the model response.

    Returns:
        A ``(spec, validation_traces)`` tuple. ``spec`` is ``None`` when parsing fails.
    """
    try:
        return ArenaEnvGraphSpec.model_validate(data), []
    except ValidationError as exc:
        return None, format_validation_error(exc)


def collect_agent_ready_task_validation_traces(spec: ArenaEnvGraphSpec) -> list[str]:
    """Return agent-only task constraint violations not enforced by ``ArenaEnvGraphSpec``."""
    traces: list[str] = []
    task_registry = TaskRegistry()
    for task in spec.task.subtasks:
        task_cls = task_registry.get_task_by_name(task.kind)
        if not getattr(task_cls, "agent_ready", False):
            traces.append(f"Task {task.kind!r} is not agent-ready")
        for required_param in required_task_init_param_names(task_cls):
            if required_param not in task.params:
                traces.append(f"Task {task.kind!r} is missing required param {required_param!r}")
    return traces

# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Optional shared task state with explicit control-step and reset ownership."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class TaskRuntime(ABC):
    """Own mechanisms and measurements shared by a task's read-only predicates.

    Arena calls update before success evaluation, after physics. Reset hooks
    surround the existing reset events, including placement and variations.
    These hooks are evaluator infrastructure, not robot action interfaces.
    """

    def prepare_reset(self, env_ids) -> None:
        """Release episode-owned constraints before scene reset and variations."""

    @abstractmethod
    def reset(self, env_ids) -> None:
        """Initialize selected episodes after scene reset, placement, and variations."""

    @abstractmethod
    def update(self) -> None:
        """Measure the current control step before progress and observations read it."""


@dataclass
class TaskRuntimeCfg:
    """Construct one runtime using ``class_type(env, **params)`` per environment."""

    class_type: type[TaskRuntime]
    """Implementation that owns the task's shared state."""

    params: dict[str, Any] = field(default_factory=dict)
    """Constructor arguments, excluding the environment supplied by Arena."""

    def build(self, env) -> TaskRuntime:
        """Create the task runtime after scene handles are available."""
        assert issubclass(self.class_type, TaskRuntime), "Task runtimes must implement TaskRuntime."
        return self.class_type(env, **self.params)

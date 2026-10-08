# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bounded, isolated decision history for vectorized agent policies."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

from isaaclab_arena_vlm_agent_policy.commands import AgentCommand
from isaaclab_arena_vlm_agent_policy.interfaces import ExecutionFeedback


@dataclass(frozen=True)
class AgentContextCfg:
    """Limit retained events per environment; prompt token limits belong to the policy."""

    max_events: int = 64

    def __post_init__(self):
        assert self.max_events > 0, "max_events must be positive"


class AgentContext:
    """Store copied events and policy-owned notes independently for each environment."""

    def __init__(self, config: AgentContextCfg | None = None):
        self.config = config or AgentContextCfg()
        self._events: dict[int, deque] = {}
        self._notes: dict[int, dict[str, Any]] = {}

    def record_decision(self, env_id: int, decision_id: str, command: AgentCommand) -> None:
        """Record a newly accepted command before any execution result exists."""
        self._record(
            env_id, {"kind": "decision", "decision_id": decision_id, "command": command.model_dump(mode="json")}
        )

    def record_execution(self, env_id: int, decision_id: str, feedback: ExecutionFeedback) -> None:
        """Associate measured feedback with its decision, even after older events expire."""
        self._record(env_id, {"kind": "execution", "decision_id": decision_id, "feedback": asdict(feedback)})

    def record_task_progress(self, env_id: int, progress: dict[str, Any]) -> None:
        """Retain a progress snapshot only when the policy's feedback configuration permits it."""
        self._record(env_id, {"kind": "task_progress", "progress": progress})

    def update_notes(self, env_id: int, notes: dict[str, Any]) -> None:
        """Merge copied notes; policies bound their size before using them in prompts."""
        assert env_id >= 0, "env_id must be nonnegative"
        self._notes.setdefault(env_id, {}).update(deepcopy(notes))

    def query(self, env_id: int) -> dict[str, Any]:
        """Return an independent snapshot of retained events and notes."""
        assert env_id >= 0, "env_id must be nonnegative"
        return deepcopy({"events": list(self._events.get(env_id, ())), "notes": self._notes.get(env_id, {})})

    def reset(self, env_ids: list[int] | None = None) -> None:
        """Forget only selected environments, or all history and notes when omitted."""
        if env_ids is None:
            self._events.clear()
            self._notes.clear()
            return
        for env_id in env_ids:
            self._events.pop(env_id, None)
            self._notes.pop(env_id, None)

    def _record(self, env_id: int, event: dict[str, Any]) -> None:
        assert env_id >= 0, "env_id must be nonnegative"
        events = self._events.setdefault(env_id, deque(maxlen=self.config.max_events))
        events.append(deepcopy(event))

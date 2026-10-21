# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Reserve episode starts against quotas shared by fixed evaluation conditions."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

CONDITION_EPISODE_BUDGET_FILENAME = "condition_episode_budget.json"


@dataclass
class ObjectSetEpisodeBudgetCfg:
    """Allocate episodes to the named members of one ordered object set."""

    object_set: str
    """Name of the scene's single RigidObjectSet."""

    episodes_per_member: dict[str, int]
    """Positive episode counts keyed by unique member name."""

    max_steps: int | None = None
    """Optional control-step cap; reaching it before all quotas finish leaves the Run incomplete."""

    def __post_init__(self) -> None:
        assert isinstance(self.object_set, str) and self.object_set.strip(), "object_set must be a nonempty name"
        assert self.episodes_per_member, "episodes_per_member must not be empty"
        for name, count in self.episodes_per_member.items():
            assert isinstance(name, str) and name.strip(), "member names must be nonempty strings"
            assert type(count) is int and count > 0, "member episode budgets must be positive integers"
        assert self.max_steps is None or (
            type(self.max_steps) is int and self.max_steps > 0
        ), "max_steps must be a positive integer"


class ConditionEpisodeBudget:
    """Account for reserved and completed episodes without depending on completion speed."""

    def __init__(
        self, cfg: ObjectSetEpisodeBudgetCfg, members_by_env: list[str], member_definitions: dict | None = None
    ) -> None:
        cfg.__post_init__()
        assert members_by_env, "At least one environment must be assigned"
        assert set(members_by_env) == set(
            cfg.episodes_per_member
        ), "Every requested member must have an environment, and every environment must have a quota"
        self.object_set = cfg.object_set
        self.members_by_env = tuple(members_by_env)
        self.requested = dict(cfg.episodes_per_member)
        self.started = dict.fromkeys(self.requested, 0)
        self.completed = dict.fromkeys(self.requested, 0)
        self.active: dict[int, int] = {}
        self.member_definitions = deepcopy(member_definitions or {})
        json.dumps(self.member_definitions, allow_nan=False)
        self._output_path: Path | None = None

    @property
    def total(self) -> int:
        """Total number of episodes reserved by the configuration."""
        return sum(self.requested.values())

    @property
    def is_complete(self) -> bool:
        """Whether every requested episode finished."""
        return self.completed == self.requested and not self.active

    def assign(self, env_ids: list[int]) -> list[int]:
        """Reserve starts in environment-ID order and return the selected environments."""
        assert len(set(env_ids)) == len(env_ids), "Duplicate environment IDs"
        assert all(type(i) is int and 0 <= i < len(self.members_by_env) for i in env_ids), "Invalid environment ID"
        assert not self.active.keys() & set(env_ids), "Cannot replace an active episode"
        selected = []
        for env_id in sorted(env_ids):
            member = self.members_by_env[env_id]
            if self.started[member] < self.requested[member]:
                self.active[env_id] = self.started[member]
                self.started[member] += 1
                selected.append(env_id)
        return selected

    def complete(self, env_ids: list[int]) -> None:
        """Finalize assigned episodes regardless of success or timeout."""
        assert len(set(env_ids)) == len(env_ids), "Duplicate environment IDs"
        assert set(env_ids) <= self.active.keys(), "Cannot complete an unassigned episode"
        for env_id in env_ids:
            self.completed[self.members_by_env[env_id]] += 1
            del self.active[env_id]

    def episode_identity(self, env_id: int) -> dict[str, str | int]:
        """Return the condition and its reserved episode index before that episode is finalized."""
        return {
            "condition_object_set": self.object_set,
            "condition_member": self.members_by_env[env_id],
            "episode_in_condition": self.active[env_id],
        }

    def snapshot(self, status: str = "running", reason: str | None = None) -> dict:
        """Return allocation and outstanding assignments separately from task outcomes."""
        assert status in ("running", "completed", "incomplete")
        assert status != "completed" or self.is_complete, "Cannot mark unmet quotas completed"
        conditions = {}
        for member, requested in self.requested.items():
            conditions[member] = {
                "requested": requested,
                "started": self.started[member],
                "completed": self.completed[member],
                "in_flight": self.started[member] - self.completed[member],
                "unstarted": requested - self.started[member],
            }
        return {
            "object_set": self.object_set,
            "members_by_env": list(self.members_by_env),
            "member_definitions": deepcopy(self.member_definitions),
            "status": status,
            "reason": reason,
            "sampling_complete": self.is_complete,
            "conditions": conditions,
            "unfinished_assignments": [
                {"env_id": env_id, **self.episode_identity(env_id)} for env_id in sorted(self.active)
            ],
        }

    def set_output_path(self, path: Path) -> None:
        """Persist the requested allocation before the first episode is assigned."""
        assert not any(self.started.values()), "Set the budget output before the initial reset"
        self._output_path = path
        self.write()

    def write(self, status: str = "running", reason: str | None = None) -> dict:
        """Atomically replace the allocation record; an unfinished process leaves a running record."""
        snapshot = self.snapshot(status, reason)
        if self._output_path is not None:
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._output_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(snapshot, allow_nan=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(self._output_path)
        return snapshot

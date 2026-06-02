# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Per-env state machine that ticks fine-grained subtasks each step.

How it plugs in (handled automatically by the env builder when a task overrides
``TaskBase.get_fine_grained_subtasks``):

* A reset event is registered (``mode="reset"``) that lazily constructs a
  per-env ``FineGrainedStateMachine`` on first invocation and clears state
  for the supplied ``env_ids``.
* A termination term is registered whose function ticks the state machine
  every step, publishes ``env.extras["fine_grained_subtask"]``, and returns
  all-False (it does not influence termination — it only piggybacks on the
  termination manager's per-step dispatch).

The runtime surface looks like::

    env.extras["fine_grained_subtask"] = {
        "states": [
            {
                "subtasks": {
                    "<subtask_name>": {
                        "completed_groups": int,
                        "total_groups": int,
                        "score": float,                # 0..1, normalized within subtask
                        "is_complete": bool,
                        "active_predicates": {group: str | None},
                    },
                    ...
                },
                "overall_score": float,                # weighted by FineGrainedSubtask.score
                "all_complete": bool,
            },
            ...                                        # one entry per env
        ],
        "events": [
            [{"step": int, "subtask": str, "group": str,
              "predicate_index": int, "predicate_name": str,
              "score_delta": float}, ...],
            ...                                        # one list per env, episode-scoped
        ],
    }
"""

from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import Any

from isaaclab.managers import EventTermCfg, TerminationTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.tasks.fine_grained_subtask import FineGrainedSubtask

_STATE_MACHINE_ATTR = "_fine_grained_subtask_state_machine"


def _predicate_repr(pred) -> str:
    """Best-effort human-readable name for a (possibly functools.partial) predicate."""
    fn = getattr(pred, "func", pred)
    name = getattr(fn, "__name__", repr(fn))
    kwargs = getattr(pred, "keywords", None) or {}
    args = getattr(pred, "args", ()) or ()
    parts = [repr(a) for a in args]
    for key, value in kwargs.items():
        if isinstance(value, (str, int, float, bool)):
            parts.append(f"{key}={value!r}")
    return f"{name}({', '.join(parts)})" if parts else name


class FineGrainedSubtaskRunner:
    """Per-subtask state, vectorized across all envs."""

    def __init__(self, subtask: FineGrainedSubtask, num_envs: int, device):
        self.subtask = subtask
        self.num_envs = num_envs
        self.device = device

        self.current_index: dict[str, torch.Tensor] = {}
        self.group_score: dict[str, torch.Tensor] = {}
        self.group_complete: dict[str, torch.Tensor] = {}
        for group in subtask.group_names:
            self.current_index[group] = torch.zeros(num_envs, dtype=torch.long, device=device)
            self.group_score[group] = torch.zeros(num_envs, dtype=torch.float32, device=device)
            self.group_complete[group] = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def _compute_gating_mask(self, env) -> torch.Tensor:
        """Per-env mask of whether this recipe is *active* this step.

        - ``parent_subtask_idx is None`` → recipe always active (returns all True).
        - ``env._current_subtask_idx`` missing → parent is not sequential
          (e.g. unordered ``CompositeTaskBase``), no gating, all True.
        - Otherwise (sequential parent) → True only for envs whose current
          parent-subtask index matches this recipe's ``parent_subtask_idx``.
        """
        if self.subtask.parent_subtask_idx is None:
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        current_idx = getattr(env, "_current_subtask_idx", None)
        if current_idx is None:
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        if torch.is_tensor(current_idx):
            ci = current_idx.to(self.device)
        else:
            ci = torch.as_tensor(current_idx, device=self.device)
        return ci == int(self.subtask.parent_subtask_idx)

    def step(self, env, step_index: torch.Tensor | None) -> list[dict]:
        """Advance each group's pointer where its currently-targeted predicate fires.

        Returns a list of transition-event dicts. Each event carries the env id
        in the ``env_idx`` key so the orchestrator can route it.
        """
        events: list[dict] = []
        gating_mask = self._compute_gating_mask(env)
        if not bool(gating_mask.any().item()):
            return events
        for group, chain in self.subtask.canonical_predicate_groups.items():
            chain_length = len(chain)
            advanced = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

            for chain_idx, (predicate, score_weight) in enumerate(chain):
                at_position = (self.current_index[group] == chain_idx) & ~advanced & gating_mask
                if not bool(at_position.any().item()):
                    continue

                result = predicate(env)
                if not torch.is_tensor(result):
                    result = torch.as_tensor(result, dtype=torch.bool)
                if result.device != self.device:
                    result = result.to(self.device)
                result = result.bool().reshape(-1)
                if result.shape[0] != self.num_envs:
                    raise RuntimeError(
                        f"Predicate {_predicate_repr(predicate)} returned shape {tuple(result.shape)};"
                        f" expected ({self.num_envs},)"
                    )

                advance_mask = at_position & result
                if not bool(advance_mask.any().item()):
                    continue

                self.current_index[group] = torch.where(
                    advance_mask,
                    self.current_index[group] + 1,
                    self.current_index[group],
                )
                self.group_score[group] = self.group_score[group] + advance_mask.float() * float(score_weight)
                advanced = advanced | advance_mask

                pred_name = _predicate_repr(predicate)
                for eid in torch.nonzero(advance_mask, as_tuple=False).flatten().tolist():
                    events.append({
                        "env_idx": int(eid),
                        "step": int(step_index[eid].item()) if step_index is not None else -1,
                        "subtask": self.subtask.name,
                        "group": group,
                        "predicate_index": chain_idx,
                        "predicate_name": pred_name,
                        "score_delta": float(score_weight),
                    })

            self.group_complete[group] = self.current_index[group] >= chain_length

        return events

    def reset(self, env_ids) -> None:
        for group in self.subtask.group_names:
            for eid in env_ids:
                self.current_index[group][eid] = 0
                self.group_score[group][eid] = 0.0
                self.group_complete[group][eid] = False

    def is_complete(self) -> torch.Tensor:
        groups = self.subtask.group_names
        stacked = torch.stack([self.group_complete[g] for g in groups], dim=1)
        if self.subtask.logical == "all":
            return stacked.all(dim=1)
        if self.subtask.logical == "any":
            return stacked.any(dim=1)
        return stacked.sum(dim=1) >= int(self.subtask.K or 1)

    def overall_score_per_env(self) -> torch.Tensor:
        """Mean group score within this subtask, in [0, 1]."""
        groups = self.subtask.group_names
        stacked = torch.stack([self.group_score[g] for g in groups], dim=1)
        return stacked.mean(dim=1)


class FineGrainedStateMachine:
    """Owns runners for all subtasks of a single task across all envs.

    The state machine is constructed lazily on the first reset or step so that
    ``num_envs`` and ``device`` are known.
    """

    def __init__(self, subtasks: list[FineGrainedSubtask], num_envs: int, device):
        self.subtasks = subtasks
        self.num_envs = num_envs
        self.device = device
        self.runners = [FineGrainedSubtaskRunner(s, num_envs, device) for s in subtasks]
        self._events: list[list[dict]] = [[] for _ in range(num_envs)]

    def step(self, env, step_index: torch.Tensor | None) -> None:
        for runner in self.runners:
            for event in runner.step(env, step_index):
                eid = event.pop("env_idx")
                self._events[eid].append(event)

    def reset(self, env_ids) -> None:
        for runner in self.runners:
            runner.reset(env_ids)
        for eid in env_ids:
            self._events[eid] = []

    def get_state(self) -> list[dict]:
        out: list[dict] = []
        for env_idx in range(self.num_envs):
            subtask_states: dict[str, dict] = {}
            overall_score = 0.0
            all_complete = True
            for runner in self.runners:
                subtask = runner.subtask
                completed_groups = 0
                total_groups = len(subtask.group_names)
                active_predicates: dict[str, str | None] = {}
                for group in subtask.group_names:
                    cur = int(runner.current_index[group][env_idx].item())
                    chain = subtask.canonical_predicate_groups[group]
                    if cur >= len(chain):
                        active_predicates[group] = None
                        completed_groups += 1
                    else:
                        active_predicates[group] = _predicate_repr(chain[cur][0])
                subtask_score = float(runner.overall_score_per_env()[env_idx].item())
                is_complete = bool(runner.is_complete()[env_idx].item())
                subtask_states[subtask.name] = {
                    "completed_groups": completed_groups,
                    "total_groups": total_groups,
                    "score": subtask_score,
                    "is_complete": is_complete,
                    "active_predicates": active_predicates,
                }
                overall_score += subtask.score * subtask_score
                all_complete = all_complete and is_complete
            out.append({
                "subtasks": subtask_states,
                "overall_score": overall_score,
                "all_complete": all_complete,
            })
        return out

    def get_events(self) -> list[list[dict]]:
        return [list(e) for e in self._events]


def _ensure_state_machine(env, subtasks: list[FineGrainedSubtask]) -> FineGrainedStateMachine:
    sm: FineGrainedStateMachine | None = getattr(env, _STATE_MACHINE_ATTR, None)
    if sm is None:
        sm = FineGrainedStateMachine(subtasks=subtasks, num_envs=env.num_envs, device=env.device)
        setattr(env, _STATE_MACHINE_ATTR, sm)
    return sm


def fine_grained_subtask_step_func(env, subtasks: list[FineGrainedSubtask]) -> torch.Tensor:
    """Per-step termination-term entry point.

    Ticks the state machine, publishes ``env.extras["fine_grained_subtask"]``,
    and returns all-False so it does not contribute to termination.
    """
    sm = _ensure_state_machine(env, subtasks)
    step_index = getattr(env, "episode_length_buf", None)
    sm.step(env, step_index=step_index)
    env.extras["fine_grained_subtask"] = {
        "states": sm.get_state(),
        "events": sm.get_events(),
    }
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def fine_grained_subtask_reset_func(env, env_ids, subtasks: list[FineGrainedSubtask]) -> None:
    """Per-reset event entry point: zeroes the state machine for the supplied envs."""
    sm = _ensure_state_machine(env, subtasks)
    if env_ids is None:
        env_ids = list(range(env.num_envs))
    elif torch.is_tensor(env_ids):
        env_ids = env_ids.tolist()
    sm.reset(env_ids)


@configclass
class FineGrainedSubtaskEventsCfg:
    reset_fine_grained_subtasks: EventTermCfg = MISSING


@configclass
class FineGrainedSubtaskTerminationsCfg:
    fine_grained_subtask_step: TerminationTermCfg = MISSING


def make_fine_grained_subtask_events_cfg(subtasks: list[FineGrainedSubtask]) -> Any:
    return FineGrainedSubtaskEventsCfg(
        reset_fine_grained_subtasks=EventTermCfg(
            func=fine_grained_subtask_reset_func,
            mode="reset",
            params={"subtasks": subtasks},
        )
    )


def make_fine_grained_subtask_termination_cfg(subtasks: list[FineGrainedSubtask]) -> Any:
    return FineGrainedSubtaskTerminationsCfg(
        fine_grained_subtask_step=TerminationTermCfg(
            func=fine_grained_subtask_step_func,
            params={"subtasks": subtasks},
        )
    )

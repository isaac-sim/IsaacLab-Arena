# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import Any

from isaaclab.managers import EventTermCfg, TerminationTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.tasks.fine_grained_subtask import FineGrainedSubtask

_STATE_MACHINE_ATTR = "_fine_grained_subtask_state_machine"


def _predicate_repr(pred) -> str:
    """Generate human-readable string representation for a predicate."""

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
    """State machine runner for a single FineGrainedSubtask object.

    Each runner is responsible for tracking the progress of all predicate_groups
    within a FineGrainedSubtask object across all parallelenvironments.
    """

    def __init__(self, fine_grained_subtask: FineGrainedSubtask, num_envs: int, device):
        self.fine_grained_subtask = fine_grained_subtask
        self.num_envs = num_envs
        self.device = device

        # Initialize the state machine's internal state.
        self.current_index: dict[str, torch.Tensor] = {}
        self.group_score: dict[str, torch.Tensor] = {}
        self.group_complete: dict[str, torch.Tensor] = {}

        for group_name in fine_grained_subtask.group_names:
            self.current_index[group_name] = torch.zeros(num_envs, dtype=torch.long, device=device)
            self.group_score[group_name] = torch.zeros(num_envs, dtype=torch.float32, device=device)
            self.group_complete[group_name] = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def _compute_composite_task_gating_mask(self, env) -> torch.Tensor:
        """Per-env mask of whether the FineGrainedSubtask is active.

        The gating is used to determine when tracking of predicates should
        be active for composite tasks.
        """

        # If no parent_subtask_idx -> always active (returns all True).
        if self.fine_grained_subtask.parent_subtask_idx is None:
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        # If no env._current_subtask_idx -> composite task is not sequential (returns all True).
        current_idx = getattr(env, "_current_subtask_idx", None)
        if current_idx is None:
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        # Otherwise return True only for envs whose current
        # parent-subtask index matches this FineGrainedSubtask's parent_subtask_idx.
        if torch.is_tensor(current_idx):
            ci = current_idx.to(self.device)
        else:
            ci = torch.as_tensor(current_idx, device=self.device)
        return ci == int(self.fine_grained_subtask.parent_subtask_idx)

    def step(self, env, step_index: torch.Tensor | None) -> list[dict]:
        """Step the state machine runner for a single env.step.

        Check each group's current predicate, move the state machine to the next predicate if
        the current predicate is True. Emit an event for each env where a predicate was advanced.
        """

        # List of state transition events (events are emitted for an env when a predicate flips True)
        events: list[dict] = []

        # If the FineGrainedSubtask is not active for the composite task, return.
        composite_task_gating_mask = self._compute_composite_task_gating_mask(env)
        if not bool(composite_task_gating_mask.any().item()):
            return events

        # Step through each group of the FineGrainedSubtask.
        for group_name, predicate_chain in self.fine_grained_subtask.canonical_predicate_groups.items():
            chain_length = len(predicate_chain)
            # Mask for which envs have advanced this step.
            advanced = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

            for chain_idx, (predicate, score_weight) in enumerate(predicate_chain):
                # Compute mask for which envs that should evaluate the predicate.
                # Envs should only be evaluated if:
                #   1) They are at the current predicate position
                #   2) They have not yet advanced this step
                #   3) The FineGrainedSubtask is active for the composite task
                at_position = (self.current_index[group_name] == chain_idx) & ~advanced & composite_task_gating_mask
                if not bool(at_position.any().item()):
                    continue

                # Evaluate the predicate for all envs, reshaped to a flat (num_envs,) bool tensor.
                result = torch.as_tensor(predicate(env), dtype=torch.bool, device=self.device).reshape(-1)
                if result.shape[0] != self.num_envs:
                    raise RuntimeError(
                        f"Predicate {_predicate_repr(predicate)} returned shape {tuple(result.shape)};"
                        f" expected ({self.num_envs},)"
                    )

                # Compute mask for which envs need to be advanced to the next predicate.
                advance_mask = at_position & result
                if not bool(advance_mask.any().item()):
                    continue

                # Advance the state machine to the next predicates.
                self.current_index[group_name] = torch.where(
                    advance_mask,
                    self.current_index[group_name] + 1,
                    self.current_index[group_name],
                )
                # Update the group score for the envs that were advanced.
                self.group_score[group_name] = self.group_score[group_name] + advance_mask.float() * float(score_weight)
                # Update the advanced mask for the envs that were advanced.
                advanced = advanced | advance_mask

                # Emit an event for each env where a predicate was advanced.
                pred_name = _predicate_repr(predicate)
                for eid in torch.nonzero(advance_mask, as_tuple=False).flatten().tolist():
                    events.append({
                        "env_idx": int(eid),
                        "step": int(step_index[eid].item()) if step_index is not None else -1,
                        "fine_grained_subtask": self.fine_grained_subtask.name,
                        "group": group_name,
                        "predicate_index": chain_idx,
                        "predicate_name": pred_name,
                        "score_delta": float(score_weight),
                    })

            # Update the group complete mask for the envs that have completed the group.
            self.group_complete[group_name] = self.current_index[group_name] >= chain_length

        return events

    def reset(self, env_ids) -> None:
        """Reset the state machine runner for the provided envs."""

        for group_name in self.fine_grained_subtask.group_names:
            for eid in env_ids:
                self.current_index[group_name][eid] = 0
                self.group_score[group_name][eid] = 0.0
                self.group_complete[group_name][eid] = False

    def is_complete(self) -> torch.Tensor:
        """Check if the FineGrainedSubtask is complete for all envs."""

        groups = self.fine_grained_subtask.group_names
        stacked = torch.stack([self.group_complete[g] for g in groups], dim=1)
        if self.fine_grained_subtask.logical == "all":
            return stacked.all(dim=1)
        if self.fine_grained_subtask.logical == "any":
            return stacked.any(dim=1)
        return stacked.sum(dim=1) >= int(self.fine_grained_subtask.K or 1)

    def overall_score_per_env(self) -> torch.Tensor:
        """Compute mean group score within this FineGrainedSubtask (in [0, 1])."""

        groups = self.fine_grained_subtask.group_names
        stacked = torch.stack([self.group_score[g] for g in groups], dim=1)
        return stacked.mean(dim=1)


class FineGrainedSubtaskTrackingStateMachine:
    """State machine that manages runners for all FineGrainedSubtasks.

    Attributes:
        fine_grained_subtasks: List of FineGrainedSubtasks to manage.
        num_envs: Number of parallelenvironments.
        device: Device to manage the state machine on.
        runners: List of runners for each FineGrainedSubtask.
        _events: List of events for each environment.
    """

    def __init__(self, fine_grained_subtasks: list[FineGrainedSubtask], num_envs: int, device):
        self.fine_grained_subtasks = fine_grained_subtasks
        self.num_envs = num_envs
        self.device = device
        self.runners = [FineGrainedSubtaskRunner(s, num_envs, device) for s in fine_grained_subtasks]
        self._events: list[list[dict]] = [[] for _ in range(num_envs)]

    def step(self, env, step_index: torch.Tensor | None) -> None:
        """Step each runner for a single env.step."""

        for runner in self.runners:
            for event in runner.step(env, step_index):
                eid = event.pop("env_idx")
                self._events[eid].append(event)

    def reset(self, env_ids) -> None:
        """Reset the runners for the provided envs."""

        for runner in self.runners:
            runner.reset(env_ids)
        for eid in env_ids:
            self._events[eid] = []

    def get_state(self) -> list[dict]:
        """Get the state of each FineGrainedSubtask for all envs."""

        output: list[dict] = []
        for env_idx in range(self.num_envs):
            # Build a per-env dict from each runner's state.
            fine_grained_subtask_states: dict[str, dict] = {}
            overall_score = 0.0
            all_complete = True

            for runner in self.runners:
                fine_grained_subtask = runner.fine_grained_subtask
                completed_groups = 0
                total_groups = len(fine_grained_subtask.group_names)
                active_predicates: dict[str, str | None] = {}

                # Compute the active predicates and completed groups.
                for group_name in fine_grained_subtask.group_names:
                    cur_group_index = int(runner.current_index[group_name][env_idx].item())
                    predicate_chain = fine_grained_subtask.canonical_predicate_groups[group_name]
                    if cur_group_index >= len(predicate_chain):
                        active_predicates[group_name] = None
                        completed_groups += 1
                    else:
                        active_predicates[group_name] = _predicate_repr(predicate_chain[cur_group_index][0])

                # Compute the overall score and completeness.
                fine_grained_subtask_score = float(runner.overall_score_per_env()[env_idx].item())
                is_complete = bool(runner.is_complete()[env_idx].item())
                fine_grained_subtask_states[fine_grained_subtask.name] = {
                    "completed_groups": completed_groups,
                    "total_groups": total_groups,
                    "score": fine_grained_subtask_score,
                    "is_complete": is_complete,
                    "active_predicates": active_predicates,
                }
                overall_score += fine_grained_subtask.score * fine_grained_subtask_score
                all_complete = all_complete and is_complete

            # Add the per-env state dict to the output.
            output.append({
                "fine_grained_subtasks": fine_grained_subtask_states,
                "overall_score": overall_score,
                "all_complete": all_complete,
            })
        return output

    def get_events(self) -> list[list[dict]]:
        """Get all events for all envs."""

        return [list(e) for e in self._events]


def _ensure_state_machine(
    env, fine_grained_subtasks: list[FineGrainedSubtask]
) -> FineGrainedSubtaskTrackingStateMachine:
    """Return the env's FineGrainedSubtaskTrackingStateMachine, lazily creating and caching it on first call."""

    sm: FineGrainedSubtaskTrackingStateMachine | None = getattr(env, _STATE_MACHINE_ATTR, None)
    if sm is None:
        sm = FineGrainedSubtaskTrackingStateMachine(
            fine_grained_subtasks=fine_grained_subtasks, num_envs=env.num_envs, device=env.device
        )
        setattr(env, _STATE_MACHINE_ATTR, sm)
    return sm


def fine_grained_subtask_step_func(env, fine_grained_subtasks: list[FineGrainedSubtask]) -> torch.Tensor:
    """Termination-term entry point.

    Ticks the state machine, writes events and states to env.extras["fine_grained_subtask"],
    and returns all-False so it does not contribute to termination.
    """

    sm = _ensure_state_machine(env, fine_grained_subtasks)
    step_index = getattr(env, "episode_length_buf", None)
    sm.step(env, step_index=step_index)

    """
    User-facing event/state information format:

    env.extras["fine_grained_subtask"] = {
        "states": [
            {
                "fine_grained_subtasks": {
                    "<fine_grained_subtask_name>": {
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
            [{"step": int, "fine_grained_subtask": str, "group": str,
              "predicate_index": int, "predicate_name": str,
              "score_delta": float}, ...],
            ...                                        # one list per env
        ],
    }
    """

    env.extras["fine_grained_subtask"] = {
        "states": sm.get_state(),
        "events": sm.get_events(),
    }

    # Return all-False so it does not contribute to termination.
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def fine_grained_subtask_reset_func(env, env_ids, fine_grained_subtasks: list[FineGrainedSubtask]) -> None:
    """Reset-event entry point.

    Resets the state machine whenever the Lab env is reset.
    """

    sm = _ensure_state_machine(env, fine_grained_subtasks)
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


def make_fine_grained_subtask_events_cfg(fine_grained_subtasks: list[FineGrainedSubtask]) -> Any:
    return FineGrainedSubtaskEventsCfg(
        reset_fine_grained_subtasks=EventTermCfg(
            func=fine_grained_subtask_reset_func,
            mode="reset",
            params={"fine_grained_subtasks": fine_grained_subtasks},
        )
    )


def make_fine_grained_subtask_termination_cfg(fine_grained_subtasks: list[FineGrainedSubtask]) -> Any:
    return FineGrainedSubtaskTerminationsCfg(
        fine_grained_subtask_step=TerminationTermCfg(
            func=fine_grained_subtask_step_func,
            params={"fine_grained_subtasks": fine_grained_subtasks},
        )
    )

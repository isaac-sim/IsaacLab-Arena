# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import Any

from isaaclab.managers import EventTermCfg
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.progress_tracking.fine_grained_progress_objective import FineGrainedProgressObjective

_STATE_MACHINE_ATTR = "_fine_grained_progress_tracker"


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


class FineGrainedProgressObjectiveRunner:
    """State machine runner for a single FineGrainedProgressObjective object.

    Each runner is responsible for tracking the progress of all predicate_groups
    within a FineGrainedProgressObjective object across all parallel environments.
    """

    def __init__(self, fine_grained_progress_objective: FineGrainedProgressObjective, num_envs: int, device):
        self.fine_grained_progress_objective = fine_grained_progress_objective
        self.num_envs = num_envs
        self.device = device

        # Initialize the state machine's internal state.
        self.current_index: dict[str, torch.Tensor] = {}
        self.group_score: dict[str, torch.Tensor] = {}
        self.group_complete: dict[str, torch.Tensor] = {}

        for group_name in fine_grained_progress_objective.group_names:
            self.current_index[group_name] = torch.zeros(num_envs, dtype=torch.long, device=device)
            self.group_score[group_name] = torch.zeros(num_envs, dtype=torch.float32, device=device)
            self.group_complete[group_name] = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def _compute_composite_task_gating_mask(self, env) -> torch.Tensor:
        """Per-env mask of whether the FineGrainedProgressObjective is active.

        The gating is used to determine when tracking of predicates should
        be active for composite tasks.
        """

        # If no parent_subtask_idx -> always active (returns all True).
        if self.fine_grained_progress_objective.parent_subtask_idx is None:
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        # If no env._current_subtask_idx -> composite task is not sequential (returns all True).
        current_idx = getattr(env, "_current_subtask_idx", None)
        if current_idx is None:
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        # Otherwise return True only for envs whose current
        # parent-subtask index matches this FineGrainedProgressObjective's parent_subtask_idx.
        if torch.is_tensor(current_idx):
            ci = current_idx.to(self.device)
        else:
            ci = torch.as_tensor(current_idx, device=self.device)
        return ci == int(self.fine_grained_progress_objective.parent_subtask_idx)

    def step(self, env, step_index: torch.Tensor | None) -> list[dict]:
        """Step the state machine runner for a single env.step.

        Advance each group's predicate chain by at most one position per env and return a
        transition event for every env/group that advanced this step.
        """

        # If the FineGrainedProgressObjective is not active for the composite task, there is
        # nothing to advance for any env.
        gating_mask = self._compute_composite_task_gating_mask(env)
        if not bool(gating_mask.any().item()):
            return []

        events: list[dict] = []
        for group_name, predicate_chain in self.fine_grained_progress_objective.canonical_predicate_groups.items():
            events += self._step_group(env, group_name, predicate_chain, gating_mask, step_index)
        return events

    def _step_group(
        self,
        env,
        group_name: str,
        predicate_chain: list[tuple],
        gating_mask: torch.Tensor,
        step_index: torch.Tensor | None,
    ) -> list[dict]:
        """Advance a single group's predicate chain by at most one position per env.

        Evaluates the current predicate for the envs sitting at each chain position, advances
        those whose predicate is satisfied, updates the group's score and completion mask, and
        returns one transition event per env that advanced.
        """

        # List of state transition events (events are emitted for an env when a predicate flips True)
        events: list[dict] = []
        chain_length = len(predicate_chain)
        # Mask for which envs have advanced this step (at most one advance per env per group).
        advanced = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        for chain_idx, (predicate, score_weight) in enumerate(predicate_chain):
            # Compute mask for which envs that should evaluate the predicate.
            # Envs should only be evaluated if:
            #   1) They are at the current predicate position
            #   2) They have not yet advanced this step
            #   3) The FineGrainedProgressObjective is active for the composite task
            at_position = (self.current_index[group_name] == chain_idx) & ~advanced & gating_mask
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
                    "fine_grained_progress_objective": self.fine_grained_progress_objective.name,
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

        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        for group_name in self.fine_grained_progress_objective.group_names:
            self.current_index[group_name][env_ids] = 0
            self.group_score[group_name][env_ids] = 0.0
            self.group_complete[group_name][env_ids] = False

    def is_complete(self) -> torch.Tensor:
        """Check if the FineGrainedProgressObjective is complete for all envs."""

        groups = self.fine_grained_progress_objective.group_names
        stacked = torch.stack([self.group_complete[g] for g in groups], dim=1)
        if self.fine_grained_progress_objective.logical == "all":
            return stacked.all(dim=1)
        if self.fine_grained_progress_objective.logical == "any":
            return stacked.any(dim=1)
        return stacked.sum(dim=1) >= int(self.fine_grained_progress_objective.K or 1)

    def overall_score_per_env(self) -> torch.Tensor:
        """Compute mean group score within this FineGrainedProgressObjective (in [0, 1])."""

        groups = self.fine_grained_progress_objective.group_names
        stacked = torch.stack([self.group_score[g] for g in groups], dim=1)
        return stacked.mean(dim=1)

    def get_state_for_env(self, env_idx: int, is_complete, score) -> dict:
        """Per-env view of this objective's progress.

        is_complete and score are passed in (rather than recomputed here) so the full
        (num_envs,) tensor reductions run once per runner in
        FineGrainedProgressTracker, instead of once per env.
        """

        objective = self.fine_grained_progress_objective
        completed_groups = 0
        active_predicates: dict[str, str | None] = {}
        # The active predicate for a group is the one at its current chain position. Any group
        # whose pointer has run off the end of the chain is complete (no active predicate).
        for group_name in objective.group_names:
            predicate_chain = objective.canonical_predicate_groups[group_name]
            cur_group_index = int(self.current_index[group_name][env_idx].item())
            if cur_group_index >= len(predicate_chain):
                active_predicates[group_name] = None
                completed_groups += 1
            else:
                active_predicates[group_name] = _predicate_repr(predicate_chain[cur_group_index][0])

        return {
            "completed_groups": completed_groups,
            "total_groups": len(objective.group_names),
            "score": float(score),
            "is_complete": bool(is_complete),
            "active_predicates": active_predicates,
        }


class FineGrainedProgressTracker:
    """State machine that manages runners for all FineGrainedProgressObjectives.

    Attributes:
        fine_grained_progress_objectives: List of FineGrainedProgressObjectives to manage.
        num_envs: Number of parallel environments.
        device: Device to manage the state machine on.
        runners: List of runners for each FineGrainedProgressObjective.
        _events: List of events for each environment.
    """

    def __init__(self, fine_grained_progress_objectives: list[FineGrainedProgressObjective], num_envs: int, device):
        self.fine_grained_progress_objectives = fine_grained_progress_objectives
        self.num_envs = num_envs
        self.device = device
        self.runners = [
            FineGrainedProgressObjectiveRunner(s, num_envs, device) for s in fine_grained_progress_objectives
        ]
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
        """Get the state of each FineGrainedProgressObjective for all envs."""

        # Compute the per-runner (num_envs,) tensors once
        completeness = [runner.is_complete() for runner in self.runners]
        scores = [runner.overall_score_per_env() for runner in self.runners]

        output: list[dict] = []
        for env_idx in range(self.num_envs):
            # Build a per-env dict from each runner's state.
            progress_objective_states: dict[str, dict] = {}
            overall_score = 0.0
            all_complete = True
            for i, runner in enumerate(self.runners):
                objective = runner.fine_grained_progress_objective
                state = runner.get_state_for_env(env_idx, completeness[i][env_idx], scores[i][env_idx])
                progress_objective_states[objective.name] = state
                overall_score += objective.score * state["score"]
                all_complete = all_complete and state["is_complete"]

            # Add the per-env state dict to the output.
            output.append({
                "fine_grained_progress_objectives": progress_objective_states,
                "overall_score": overall_score,
                "all_complete": all_complete,
            })
        return output

    def get_events(self) -> list[list[dict]]:
        """Get all events for all envs."""

        return [list(e) for e in self._events]


def _ensure_progress_tracker(
    env, fine_grained_progress_objectives: list[FineGrainedProgressObjective]
) -> FineGrainedProgressTracker:
    """Return the env's FineGrainedProgressTracker, lazily creating and caching it on first call."""

    sm: FineGrainedProgressTracker | None = getattr(env, _STATE_MACHINE_ATTR, None)
    if sm is None:
        sm = FineGrainedProgressTracker(
            fine_grained_progress_objectives=fine_grained_progress_objectives, num_envs=env.num_envs, device=env.device
        )
        setattr(env, _STATE_MACHINE_ATTR, sm)
    return sm


class FineGrainedProgressRecorder(RecorderTerm):
    """Per-step hook that ticks the FineGrainedProgressTracker. Records nothing.

    Registered as a recorder term so it runs once per env.step via
    record_post_step. It advances the state machine and publishes the per-step state/events to
    env.extras["fine_grained_progress"], then returns
    (None, None) so nothing is written to the recorded episode data.

    env.extras["fine_grained_progress"] format:

        {
            "states": [                                    # one entry per env
                {
                    "fine_grained_progress_objectives": {
                        "<name>": {
                            "completed_groups": int,
                            "total_groups": int,
                            "score": float,                # 0..1, normalized within objective
                            "is_complete": bool,
                            "active_predicates": {group: str | None},
                        },
                        ...
                    },
                    "overall_score": float,                # weighted by FineGrainedProgressObjective.score
                    "all_complete": bool,
                },
                ...
            ],
            "events": [                                    # one list per env
                [{"step": int, "fine_grained_progress_objective": str, "group": str,
                  "predicate_index": int, "predicate_name": str,
                  "score_delta": float}, ...],
                ...
            ],
        }
    """

    def __init__(self, cfg: FineGrainedProgressObjectiveRecorderCfg, env):
        super().__init__(cfg, env)
        self._fine_grained_progress_objectives = cfg.fine_grained_progress_objectives

    def record_post_step(self):
        """Ticks the state machine, writes events and states to env.extras["fine_grained_progress"]"""

        sm = _ensure_progress_tracker(self._env, self._fine_grained_progress_objectives)
        step_index = getattr(self._env, "episode_length_buf", None)
        sm.step(self._env, step_index=step_index)
        self._env.extras["fine_grained_progress"] = {
            "states": sm.get_state(),
            "events": sm.get_events(),
        }
        # This term is a per-step hook only — record nothing.
        return None, None


def fine_grained_progress_reset_func(
    env, env_ids, fine_grained_progress_objectives: list[FineGrainedProgressObjective]
) -> None:
    """Reset-event entry point.

    Resets the state machine whenever the Lab env is reset.
    """

    sm = _ensure_progress_tracker(env, fine_grained_progress_objectives)
    if env_ids is None:
        env_ids = list(range(env.num_envs))
    elif torch.is_tensor(env_ids):
        env_ids = env_ids.tolist()
    sm.reset(env_ids)


@configclass
class FineGrainedProgressObjectiveEventsCfg:
    reset_fine_grained_progress_objectives: EventTermCfg = MISSING


@configclass
class FineGrainedProgressObjectiveRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = FineGrainedProgressRecorder
    fine_grained_progress_objectives: list[FineGrainedProgressObjective] = MISSING


@configclass
class FineGrainedProgressObjectiveRecorderManagerCfg(RecorderManagerBaseCfg):
    fine_grained_progress: FineGrainedProgressObjectiveRecorderCfg = MISSING


def make_fine_grained_progress_objective_events_cfg(
    fine_grained_progress_objectives: list[FineGrainedProgressObjective],
) -> Any:
    return FineGrainedProgressObjectiveEventsCfg(
        reset_fine_grained_progress_objectives=EventTermCfg(
            func=fine_grained_progress_reset_func,
            mode="reset",
            params={"fine_grained_progress_objectives": fine_grained_progress_objectives},
        )
    )


def make_fine_grained_progress_objective_recorder_cfg(
    fine_grained_progress_objectives: list[FineGrainedProgressObjective],
) -> Any:
    return FineGrainedProgressObjectiveRecorderManagerCfg(
        fine_grained_progress=FineGrainedProgressObjectiveRecorderCfg(
            fine_grained_progress_objectives=fine_grained_progress_objectives,
        )
    )

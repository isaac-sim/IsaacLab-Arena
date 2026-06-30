# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from dataclasses import MISSING, dataclass
from typing import Any

from isaaclab.managers import EventTermCfg
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective, ProgressObjectiveCompletionMode
from isaaclab_arena.progress_tracking.progress_tracking_utils import _predicate_repr

_PROGRESS_TRACKER_ATTR = "_progress_tracker"


@dataclass
class PredicateEvent:
    """A single predicate transition event emitted by the progress tracker."""

    env_idx: int
    """Index of the environment that advanced."""

    step: int
    """Episode step at which the advance happened (-1 if no step index was available)."""

    progress_objective: str
    """Name of the ProgressObjective whose group advanced."""

    group: str
    """Name of the group whose predicate chain advanced."""

    predicate_index: int
    """Index within the group's chain of the predicate that was satisfied."""

    predicate_name: str
    """Human-readable string of that predicate."""

    score_delta: float
    """Normalized score this advance added to the group."""


@dataclass
class ProgressObjectiveState:
    """Per-env snapshot of a single ProgressObjective's progress."""

    completed_groups: int
    """Number of the objective's groups that are complete for this env."""

    total_groups: int
    """Total number of groups in the objective."""

    score: float
    """Progress score in [0, 1], normalized within the objective."""

    is_complete: bool
    """Whether the objective is complete for this env."""

    active_predicates: dict[str, str | None]
    """The human-readable string of the predicate currently being evaluated. None if the group is complete."""


@dataclass
class ProgressState:
    """Per-env snapshot of progress across all ProgressObjectives."""

    progress_objectives: dict[str, ProgressObjectiveState]
    """Per-objective state, keyed by ProgressObjective name."""

    overall_score: float
    """Sum of each objective's score weighted by ProgressObjective.score."""

    all_complete: bool
    """Whether every objective is complete for this env."""


class ProgressObjectiveRunner:
    """ProgressTracker runner for a single ProgressObjective object.

    Each runner is responsible for tracking the progress of all predicate_groups
    within a ProgressObjective object across all parallel environments.
    """

    def __init__(self, progress_objective: ProgressObjective, num_envs: int, device):
        self.progress_objective = progress_objective
        self.num_envs = num_envs
        self.device = device

        #   current_predicate_index: How far each env has advanced through the group's predicate chain.
        #   group_score: Each env's accumulated score for the group, normalized to [0, 1].
        #   group_complete: Whether each env has finished the group's entire predicate chain.
        self.current_predicate_index: dict[str, torch.Tensor] = {}
        self.group_score: dict[str, torch.Tensor] = {}
        self.group_complete: dict[str, torch.Tensor] = {}

        for group_name in progress_objective.group_names:
            self.current_predicate_index[group_name] = torch.zeros(num_envs, dtype=torch.long, device=device)
            self.group_score[group_name] = torch.zeros(num_envs, dtype=torch.float32, device=device)
            self.group_complete[group_name] = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def _compute_composite_task_gating_mask(self, env) -> torch.Tensor:
        """Per-env mask of whether the ProgressObjective is active.

        The gating is used to determine when tracking of predicates should
        be active for composite tasks.
        """

        # If no parent_subtask_idx -> always active (returns all True).
        if self.progress_objective.parent_subtask_idx is None:
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        # If no env._current_subtask_idx -> composite task is not sequential (returns all True).
        current_idx = getattr(env, "_current_subtask_idx", None)
        if current_idx is None:
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        # Otherwise return True only for envs whose current
        # parent-subtask index matches this ProgressObjective's parent_subtask_idx.
        if torch.is_tensor(current_idx):
            current_idx_tensor = current_idx.to(self.device)
        else:
            current_idx_tensor = torch.as_tensor(current_idx, device=self.device)
        return current_idx_tensor == int(self.progress_objective.parent_subtask_idx)

    def step(self, env, step_index: torch.Tensor | None) -> list[PredicateEvent]:
        """Step the runner for a single env.step.

        Advance each group's predicate chain by at most one position per env and return one
        PredicateEvent for every env/group that advanced this step.
        """

        # If the ProgressObjective is not active for the composite task, there is
        # nothing to advance for any env.
        gating_mask = self._compute_composite_task_gating_mask(env)
        if not bool(gating_mask.any().item()):
            return []

        events: list[PredicateEvent] = []
        for group_name, predicate_chain in self.progress_objective.canonical_predicate_groups.items():
            events += self._step_group(env, group_name, predicate_chain, gating_mask, step_index)
        return events

    def _step_group(
        self,
        env,
        group_name: str,
        predicate_chain: list[tuple],
        gating_mask: torch.Tensor,
        step_index: torch.Tensor | None,
    ) -> list[PredicateEvent]:
        """Advance a single group's predicate chain by at most one position per env.

        Evaluates the current predicate for the envs sitting at each chain position, advances
        those whose predicate is satisfied, updates the group's score and completion mask, and
        returns one transition event per env that advanced.
        """

        # List of state transition events (events are emitted for an env when a predicate flips True)
        events: list[PredicateEvent] = []
        chain_length = len(predicate_chain)
        # Mask for which envs have advanced this step (at most one advance per env per group).
        advanced = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        for chain_idx, (predicate, score_weight) in enumerate(predicate_chain):
            # Compute mask for which envs that should evaluate the predicate.
            # Envs should only be evaluated if:
            #   1) They are at the current predicate position
            #   2) They have not yet advanced this step
            #   3) The ProgressObjective is active for the composite task
            at_position = (self.current_predicate_index[group_name] == chain_idx) & ~advanced & gating_mask
            if not bool(at_position.any().item()):
                continue

            # Evaluate the predicate for all envs, reshaped to a flat (num_envs,) bool tensor.
            result = torch.as_tensor(predicate(env), dtype=torch.bool, device=self.device).reshape(-1)
            assert result.shape[0] == self.num_envs, (
                f"Predicate {_predicate_repr(predicate)} returned shape {tuple(result.shape)};"
                f" expected ({self.num_envs},)"
            )

            # Compute mask for which envs need to be advanced to the next predicate.
            advance_mask = at_position & result
            if not bool(advance_mask.any().item()):
                continue

            # Advance the runner to the next predicates.
            self.current_predicate_index[group_name] = torch.where(
                advance_mask,
                self.current_predicate_index[group_name] + 1,
                self.current_predicate_index[group_name],
            )
            # Update the group score for the envs that were advanced.
            self.group_score[group_name] = self.group_score[group_name] + advance_mask.float() * float(score_weight)
            # Update the advanced mask for the envs that were advanced.
            advanced = advanced | advance_mask

            # Emit an event for each env where a predicate was advanced.
            pred_name = _predicate_repr(predicate)
            for env_idx in torch.nonzero(advance_mask, as_tuple=False).flatten().tolist():
                events.append(
                    PredicateEvent(
                        env_idx=int(env_idx),
                        step=int(step_index[env_idx].item()) if step_index is not None else -1,
                        progress_objective=self.progress_objective.name,
                        group=group_name,
                        predicate_index=chain_idx,
                        predicate_name=pred_name,
                        score_delta=float(score_weight),
                    )
                )

        # Update the group complete mask for the envs that have completed the group.
        self.group_complete[group_name] = self.current_predicate_index[group_name] >= chain_length
        return events

    def reset(self, env_ids) -> None:
        """Reset the runner for the provided envs."""

        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        for group_name in self.progress_objective.group_names:
            self.current_predicate_index[group_name][env_ids] = 0
            self.group_score[group_name][env_ids] = 0.0
            self.group_complete[group_name][env_ids] = False

    def _num_required_groups(self) -> int:
        """Number of groups that must complete for the objective to be complete."""

        objective = self.progress_objective
        if objective.logical == ProgressObjectiveCompletionMode.ALL:
            return len(objective.group_names)
        if objective.logical == ProgressObjectiveCompletionMode.ANY:
            return 1
        assert objective.K is not None, "K is required (and validated) when logical='choose'"
        return int(objective.K)

    def is_complete(self) -> torch.Tensor:
        """Per-env mask: True once at least the required number of groups are complete."""

        groups = self.progress_objective.group_names
        stacked = torch.stack([self.group_complete[g] for g in groups], dim=1)
        return stacked.sum(dim=1) >= self._num_required_groups()

    def overall_score_per_env(self) -> torch.Tensor:
        """Per-env score in [0, 1] that reaches 1.0 exactly when the objective completes."""

        groups = self.progress_objective.group_names
        stacked = torch.stack([self.group_score[g] for g in groups], dim=1)
        return torch.topk(stacked, self._num_required_groups(), dim=1).values.mean(dim=1)

    def get_state_for_env(self, env_idx: int, is_complete, score) -> ProgressObjectiveState:
        """Per-env view of this objective's progress.

        is_complete and score are passed in (rather than recomputed here) so the full
        (num_envs,) tensor reductions run once per runner in
        ProgressTracker, instead of once per env.
        """

        objective = self.progress_objective
        completed_groups = 0
        active_predicates: dict[str, str | None] = {}
        # The active predicate for a group is the one at its current chain position. Any group
        # whose pointer has run off the end of the chain is complete (no active predicate).
        for group_name in objective.group_names:
            predicate_chain = objective.canonical_predicate_groups[group_name]
            cur_predicate_index = int(self.current_predicate_index[group_name][env_idx].item())
            if cur_predicate_index >= len(predicate_chain):
                active_predicates[group_name] = None
                completed_groups += 1
            else:
                active_predicates[group_name] = _predicate_repr(predicate_chain[cur_predicate_index][0])

        return ProgressObjectiveState(
            completed_groups=completed_groups,
            total_groups=len(objective.group_names),
            score=float(score),
            is_complete=bool(is_complete),
            active_predicates=active_predicates,
        )


class ProgressTracker:
    """The tracker object that manages runners for all ProgressObjectives.

    Attributes:
        progress_objectives: List of ProgressObjectives to manage.
        num_envs: Number of parallel environments.
        device: Device to manage the progress tracker on.
        runners: List of runners for each ProgressObjective.
        _events: List of events for each environment.
    """

    def __init__(self, progress_objectives: list[ProgressObjective], num_envs: int, device):
        self.progress_objectives = progress_objectives
        self.num_envs = num_envs
        self.device = device
        self.runners = [ProgressObjectiveRunner(s, num_envs, device) for s in progress_objectives]
        self._events: list[list[PredicateEvent]] = [[] for _ in range(num_envs)]

    def step(self, env, step_index: torch.Tensor | None) -> None:
        """Step each runner for a single env.step."""

        for runner in self.runners:
            for event in runner.step(env, step_index):
                self._events[event.env_idx].append(event)

    def reset(self, env_ids) -> None:
        """Reset the runners for the provided envs."""

        if torch.is_tensor(env_ids):
            env_ids = env_ids.tolist()
        for runner in self.runners:
            runner.reset(env_ids)
        for env_idx in env_ids:
            self._events[env_idx] = []

    def get_state(self) -> list[ProgressState]:
        """Get the progress state of all ProgressObjectives for each env."""

        # Compute the per-runner (num_envs,) tensors once
        completeness = [runner.is_complete() for runner in self.runners]
        scores = [runner.overall_score_per_env() for runner in self.runners]

        output: list[ProgressState] = []
        for env_idx in range(self.num_envs):
            # Build a per-env state from each runner's state.
            progress_objective_states: dict[str, ProgressObjectiveState] = {}
            overall_score = 0.0
            all_complete = True
            for i, runner in enumerate(self.runners):
                objective = runner.progress_objective
                state = runner.get_state_for_env(env_idx, completeness[i][env_idx], scores[i][env_idx])
                progress_objective_states[objective.name] = state
                overall_score += objective.score * state.score
                all_complete = all_complete and state.is_complete

            output.append(
                ProgressState(
                    progress_objectives=progress_objective_states,
                    overall_score=overall_score,
                    all_complete=all_complete,
                )
            )
        return output

    def get_events(self) -> list[list[PredicateEvent]]:
        """Get all events for all envs."""

        return [list(e) for e in self._events]


def _ensure_progress_tracker(env, progress_objectives: list[ProgressObjective]) -> ProgressTracker:
    """Return the env's ProgressTracker, lazily creating and caching it on first call."""

    progress_tracker: ProgressTracker | None = getattr(env, _PROGRESS_TRACKER_ATTR, None)
    if progress_tracker is None:
        progress_tracker = ProgressTracker(
            progress_objectives=progress_objectives, num_envs=env.num_envs, device=env.device
        )
        setattr(env, _PROGRESS_TRACKER_ATTR, progress_tracker)
    return progress_tracker


class ProgressTrackingRecorder(RecorderTerm):
    """Per-step hook that ticks the ProgressTracker. Records nothing.

    Registered as a recorder term so it runs once per env.step via
    record_post_step. It advances the progress tracker and publishes the per-step state/events to
    env.extras["progress_tracking"], then returns
    (None, None) so nothing is written to the recorded episode data.

    env.extras["progress_tracking"] format:

        {
            "states": [                                    # one ProgressState per env
                ProgressState(
                    progress_objectives={
                        "<name>": ProgressObjectiveState(
                            completed_groups, total_groups, score, is_complete, active_predicates
                        ),
                        ...
                    },
                    overall_score=float,                   # weighted by ProgressObjective.score
                    all_complete=bool,
                ),
                ...
            ],
            "events": [                                    # one list of PredicateEvent per env
                [PredicateEvent(env_idx, step, progress_objective, group,
                                predicate_index, predicate_name, score_delta), ...],
                ...
            ],
        }
    """

    def __init__(self, cfg: ProgressTrackingRecorderCfg, env):
        super().__init__(cfg, env)
        self._progress_objectives = cfg.progress_objectives

    def record_post_step(self):
        """Ticks the progress tracker, writes events and states to env.extras["progress_tracking"]"""

        progress_tracker = _ensure_progress_tracker(self._env, self._progress_objectives)
        step_index = getattr(self._env, "episode_length_buf", None)
        progress_tracker.step(self._env, step_index=step_index)
        self._env.extras["progress_tracking"] = {
            "states": progress_tracker.get_state(),
            "events": progress_tracker.get_events(),
        }
        # This term is a per-step hook only — record nothing.
        return None, None


def progress_tracking_reset_func(env, env_ids, progress_objectives: list[ProgressObjective]) -> None:
    """Reset-event entry point.

    Resets the progress tracker whenever the Lab env is reset.
    """

    progress_tracker = _ensure_progress_tracker(env, progress_objectives)
    if env_ids is None:
        env_ids = list(range(env.num_envs))
    elif torch.is_tensor(env_ids):
        env_ids = env_ids.tolist()
    progress_tracker.reset(env_ids)


@configclass
class ProgressTrackingEventsCfg:
    reset_progress_objectives: EventTermCfg = MISSING


@configclass
class ProgressTrackingRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = ProgressTrackingRecorder
    progress_objectives: list[ProgressObjective] = MISSING


@configclass
class ProgressTrackingRecorderManagerCfg(RecorderManagerBaseCfg):
    progress_tracking: ProgressTrackingRecorderCfg = MISSING


def make_progress_tracking_events_cfg(
    progress_objectives: list[ProgressObjective],
) -> Any:
    return ProgressTrackingEventsCfg(
        reset_progress_objectives=EventTermCfg(
            func=progress_tracking_reset_func,
            mode="reset",
            params={"progress_objectives": progress_objectives},
        )
    )


def make_progress_tracking_recorder_cfg(
    progress_objectives: list[ProgressObjective],
) -> Any:
    return ProgressTrackingRecorderManagerCfg(
        progress_tracking=ProgressTrackingRecorderCfg(
            progress_objectives=progress_objectives,
        )
    )

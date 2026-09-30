# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import functools
import torch
from collections.abc import Callable
from dataclasses import dataclass

from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria, CriteriaCompletionMode
from isaaclab_arena.progress_tracking.progress_tracking_utils import (
    DEFAULT_SEQUENCE_NAME,
    _is_predicate,
    _predicate_repr,
)
from isaaclab_arena.tasks.predicates.stateful_predicate import StatefulPredicate
from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg, _TrueForConsecutiveSteps


def _initialize_predicate_parameters(value, env) -> None:
    """Resolve scene references and construct nested Isaac Lab terms before their parents."""
    # TaskSuccessTerm constructs the tracker before env.termination_manager exists.
    # Isaac Lab therefore cannot initialize these nested configurations for us.
    if isinstance(value, TerminationTermCfg):
        assert not isinstance(
            value.func, StatefulPredicate
        ), "Configure a stateful predicate class, not a live instance."
        _initialize_predicate_parameters(value.params, env)
        if isinstance(value.func, type):
            value.func = value.func(value, env)
        assert callable(value.func) or isinstance(
            value.func, StatefulPredicate
        ), "Predicate configs must resolve to a predicate."
    elif isinstance(value, SceneEntityCfg):
        value.resolve(env.scene)
    elif isinstance(value, dict):
        for parameter in value.values():
            _initialize_predicate_parameters(parameter, env)
    elif isinstance(value, (list, tuple)):
        for parameter in value:
            _initialize_predicate_parameters(parameter, env)


def _prepare_predicate(predicate, num_envs: int, device, env=None):
    """Construct fresh stateful occurrences and preserve ordinary callable identity."""
    assert _is_predicate(predicate), "Expected a callable, TerminationTermCfg, or TrueForConsecutiveStepsCfg."
    if isinstance(predicate, TrueForConsecutiveStepsCfg):
        return _TrueForConsecutiveSteps(
            predicate=_prepare_predicate(predicate.predicate, num_envs, device, env),
            required_steps=predicate.required_steps,
            num_envs=num_envs,
            device=device,
        )
    if not isinstance(predicate, TerminationTermCfg):
        return predicate
    assert not isinstance(
        predicate.func, StatefulPredicate
    ), "Configure a stateful predicate class, not a live instance."
    assert env is not None, "An environment is required to initialize a configured progress predicate."
    predicate_cfg = copy.deepcopy(predicate)
    _initialize_predicate_parameters(predicate_cfg, env)
    if isinstance(predicate_cfg.func, StatefulPredicate):
        # Stateful classes bind their parameters during construction; keep their lifecycle visible.
        return predicate_cfg.func
    return functools.partial(predicate_cfg.func, **predicate_cfg.params)


class _PredicateEvaluation:
    """Cache predicate results for one control step, including nested predicate evaluations."""

    def __init__(self, env, num_envs: int, device):
        self.env = env
        self.num_envs = num_envs
        self.device = device
        self._results: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def evaluate(self, predicate: Callable | StatefulPredicate, active_envs: torch.Tensor) -> torch.Tensor:
        """Evaluate each stateful occurrence at most once per active environment.

        Ordinary callables evaluate the full batch once, keyed by callable identity.
        An empty active mask reads cached stateful results without activating the runtime.
        """
        predicate_key = id(predicate)
        if predicate_key not in self._results:
            self._results[predicate_key] = (
                torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            )
        cached_result, evaluated_envs = self._results[predicate_key]
        stateful = isinstance(predicate, StatefulPredicate)
        requested_envs = active_envs if stateful else torch.ones_like(active_envs)
        pending_envs = requested_envs & ~evaluated_envs
        if bool(pending_envs.any().item()):
            result = predicate.evaluate(self, pending_envs) if stateful else predicate(self.env)
            result = torch.as_tensor(result, dtype=torch.bool, device=self.device)
            assert result.shape == (self.num_envs,), (
                f"Predicate {_predicate_repr(predicate)} returned shape {tuple(result.shape)};"
                f" expected ({self.num_envs},)"
            )
            cached_result = torch.where(pending_envs, result, cached_result)
            self._results[predicate_key] = (cached_result, evaluated_envs | pending_envs)
        return cached_result


@dataclass
class PredicateEvent:
    """A single predicate transition event emitted by the progress tracker."""

    env_idx: int
    """Index of the environment that advanced."""

    step: int
    """Episode step at which the advance happened (-1 if no step index was available)."""

    criteria_name: str
    """Name of the CompletionCriteria whose sequence advanced."""

    sequence_name: str
    """Name of the sequence whose predicate chain advanced."""

    predicate_index: int
    """Index within the sequence's chain of the predicate that was satisfied."""

    predicate_name: str
    """Human-readable string of that predicate."""

    score_delta: float
    """Normalized score this advance added to the sequence."""


@dataclass
class CompletionCriteriaState:
    """Per-env snapshot of a single CompletionCriteria's progress."""

    completed_sequences: int
    """Number of the criteria's sequences that are complete for this env."""

    total_sequences: int
    """Total number of sequences in the criteria."""

    score: float
    """Progress score in [0, 1], normalized within the criteria."""

    is_complete: bool
    """Whether the completion criteria are met for this env."""

    active_predicates: dict[str, str | None]
    """Next predicate per sequence, or None while prerequisites are pending or the sequence is complete."""

    prerequisites_met: bool = True
    """Whether these criteria's prerequisites have been satisfied in this episode."""


@dataclass
class ProgressState:
    """Per-env snapshot of progress across all CompletionCriteria definitions."""

    criteria_by_name: dict[str, CompletionCriteriaState]
    """Per-criteria state, keyed by CompletionCriteria name."""

    overall_score: float
    """Weighted progress of the criteria sets, normalized to [0, 1]."""

    all_complete: bool
    """Whether the task's success requirements are met for this env."""


class CompletionCriteriaRunner:
    """Track a CompletionCriteria's predicate sequences across parallel environments."""

    def __init__(self, completion_criteria: CompletionCriteria, num_envs: int, device, env=None):
        self.completion_criteria = completion_criteria
        self.num_envs = num_envs
        self.device = device

        #   current_predicate_index: How far each env has advanced through the sequence's predicate chain.
        #   sequence_score: Each env's accumulated score for the sequence, normalized to [0, 1].
        #   sequence_complete: Whether each env has finished the sequence's entire predicate chain.
        self.current_predicate_index: dict[str, torch.Tensor] = {}
        self.sequence_score: dict[str, torch.Tensor] = {}
        self.sequence_complete: dict[str, torch.Tensor] = {}
        self.predicate_chains = {}
        self._stateful_predicates: list[StatefulPredicate] = []
        self.prerequisites = [
            self._prepare_owned_predicate(predicate, env) for predicate in completion_criteria.prerequisites
        ]
        self.prerequisites_met = torch.full((num_envs,), not self.prerequisites, dtype=torch.bool, device=device)
        for sequence_name, chain in completion_criteria.canonical_predicate_sequences.items():
            resolved_chain = []
            for predicate, score in chain:
                resolved_chain.append((self._prepare_owned_predicate(predicate, env), score))
            self.predicate_chains[sequence_name] = resolved_chain

        for sequence_name in completion_criteria.sequence_names:
            self.current_predicate_index[sequence_name] = torch.zeros(num_envs, dtype=torch.long, device=device)
            self.sequence_score[sequence_name] = torch.zeros(num_envs, dtype=torch.float32, device=device)
            self.sequence_complete[sequence_name] = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def _prepare_owned_predicate(self, predicate, env):
        """Register the runtime roots whose lifecycle belongs to this runner."""
        prepared_predicate = _prepare_predicate(predicate, self.num_envs, self.device, env)
        if isinstance(prepared_predicate, StatefulPredicate):
            self._stateful_predicates.append(prepared_predicate)
        return prepared_predicate

    @property
    def requires_step_index(self) -> bool:
        """Whether any owned predicate requires consecutive control-step indices."""
        return any(predicate.requires_step_index for predicate in self._stateful_predicates)

    def step(
        self,
        evaluation: _PredicateEvaluation,
        step_index: torch.Tensor | None,
        active_envs: torch.Tensor,
        check_final_conditions: bool = False,
    ) -> list[PredicateEvent]:
        """Step the runner for a single env.step.

        Advance each sequence's predicate chain by at most one position per env and return one
        PredicateEvent for every env/sequence that advanced this step.
        """

        criteria_complete = self.is_complete()
        final_condition_check_mask = (
            criteria_complete if check_final_conditions else torch.zeros_like(criteria_complete)
        )
        active_envs = active_envs & ~criteria_complete
        if not bool((active_envs | final_condition_check_mask).any().item()):
            return []

        waiting_envs = active_envs & ~self.prerequisites_met
        if bool(waiting_envs.any().item()):
            all_prerequisites_hold = waiting_envs.clone()
            for prerequisite in self.prerequisites:
                result = evaluation.evaluate(prerequisite, waiting_envs)
                all_prerequisites_hold &= result
            self.prerequisites_met |= all_prerequisites_hold
        active_envs = active_envs & self.prerequisites_met

        events: list[PredicateEvent] = []
        for sequence_name, predicate_chain in self.predicate_chains.items():
            sequence_final_condition_check_mask = final_condition_check_mask
            if check_final_conditions:
                sequence_final_condition_check_mask = (
                    sequence_final_condition_check_mask | self.sequence_complete[sequence_name]
                )
            events += self._step_sequence(
                evaluation,
                sequence_name,
                predicate_chain,
                active_envs,
                step_index,
                sequence_final_condition_check_mask,
            )
        return events

    def final_conditions_met(self, evaluation: _PredicateEvaluation) -> torch.Tensor:
        """Evaluate final predicates using the CompletionCriteria's ALL, ANY, or CHOOSE requirement."""
        completed_envs = self.is_complete()
        if not bool(completed_envs.any().item()):
            return completed_envs
        # Stateful final predicates were updated during step(); do not start newly reached predicates here.
        no_state_updates = torch.zeros_like(completed_envs)
        final_results = []
        for sequence_name, predicate_chain in self.predicate_chains.items():
            # A true final predicate cannot bypass earlier predicates in its sequence.
            reached_final_predicate = self.current_predicate_index[sequence_name] >= len(predicate_chain) - 1
            final_result = evaluation.evaluate(predicate_chain[-1][0], no_state_updates)
            final_results.append(reached_final_predicate & final_result)
        return torch.stack(final_results, dim=0).sum(dim=0) >= self._num_required_sequences()

    def _step_sequence(
        self,
        evaluation: _PredicateEvaluation,
        sequence_name: str,
        predicate_chain: list[tuple],
        active_envs: torch.Tensor,
        step_index: torch.Tensor | None,
        sequence_final_condition_check_mask: torch.Tensor,
    ) -> list[PredicateEvent]:
        """Advance a single sequence's predicate chain by at most one position per env.

        Evaluates the current predicate for the envs sitting at each chain position, advances
        those whose predicate is satisfied, updates the sequence's score and completion mask, and
        returns one transition event per env that advanced.
        """

        # List of state transition events (events are emitted for an env when a predicate flips True)
        events: list[PredicateEvent] = []
        chain_length = len(predicate_chain)
        # Mask for which envs have advanced this step (at most one advance per env per sequence).
        advanced = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        for chain_idx, (predicate, score_weight) in enumerate(predicate_chain):
            # Compute mask for which envs that should evaluate the predicate.
            # Envs should only be evaluated if:
            #   1) They are at the current predicate position
            #   2) They have not yet advanced this step
            #   3) This CompletionCriteria is active in that environment.
            at_position = (self.current_predicate_index[sequence_name] == chain_idx) & ~advanced & active_envs
            state_update_mask = at_position
            if chain_idx == chain_length - 1:
                # Include completed rows now so final checks reuse this evaluation and its diagnostics.
                state_update_mask = state_update_mask | (
                    sequence_final_condition_check_mask & (self.current_predicate_index[sequence_name] >= chain_idx)
                )
            if not bool(state_update_mask.any().item()):
                continue

            # Predicates return one boolean per environment; only active rows advance.
            result = evaluation.evaluate(predicate, state_update_mask)

            # Compute mask for which envs need to be advanced to the next predicate.
            advance_mask = at_position & result
            if not bool(advance_mask.any().item()):
                continue

            # Advance the runner to the next predicates.
            self.current_predicate_index[sequence_name] = torch.where(
                advance_mask,
                self.current_predicate_index[sequence_name] + 1,
                self.current_predicate_index[sequence_name],
            )
            # Update the sequence score for the envs that were advanced.
            self.sequence_score[sequence_name] = self.sequence_score[sequence_name] + advance_mask.float() * float(
                score_weight
            )
            # Update the advanced mask for the envs that were advanced.
            advanced = advanced | advance_mask

            # Emit an event for each env where a predicate was advanced.
            pred_name = _predicate_repr(predicate)
            for env_idx in torch.nonzero(advance_mask, as_tuple=False).flatten().tolist():
                events.append(
                    PredicateEvent(
                        env_idx=int(env_idx),
                        step=int(step_index[env_idx].item()) if step_index is not None else -1,
                        criteria_name=self.completion_criteria.name,
                        sequence_name=sequence_name,
                        predicate_index=chain_idx,
                        predicate_name=pred_name,
                        score_delta=float(score_weight),
                    )
                )

        # Update the sequence complete mask for the envs that have completed the sequence.
        self.sequence_complete[sequence_name] = self.current_predicate_index[sequence_name] >= chain_length
        return events

    def reset(self, env_ids) -> None:
        """Clear progress and owned predicate state for the selected environments."""

        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.prerequisites_met[env_ids] = not self.prerequisites
        for sequence_name in self.completion_criteria.sequence_names:
            self.current_predicate_index[sequence_name][env_ids] = 0
            self.sequence_score[sequence_name][env_ids] = 0.0
            self.sequence_complete[sequence_name][env_ids] = False

        for predicate in self._stateful_predicates:
            predicate.reset(env_ids)

    def _num_required_sequences(self) -> int:
        """Number of sequences that must complete for the criteria to be complete."""

        criteria = self.completion_criteria
        if criteria.logical == CriteriaCompletionMode.ALL:
            return len(criteria.sequence_names)
        if criteria.logical == CriteriaCompletionMode.ANY:
            return 1
        assert criteria.K is not None, "K is required (and validated) when logical='choose'"
        return int(criteria.K)

    def is_complete(self) -> torch.Tensor:
        """Return which environments have met the completion criteria."""

        sequence_names = self.completion_criteria.sequence_names
        stacked = torch.stack([self.sequence_complete[name] for name in sequence_names], dim=1)
        return stacked.sum(dim=1) >= self._num_required_sequences()

    def overall_score_per_env(self) -> torch.Tensor:
        """Return progress across the required number of predicate sequences."""
        sequence_names = self.completion_criteria.sequence_names
        stacked = torch.stack([self.sequence_score[name] for name in sequence_names], dim=1)
        return torch.topk(stacked, self._num_required_sequences(), dim=1).values.mean(dim=1)

    def get_state_for_env(self, env_idx: int, is_complete, score) -> CompletionCriteriaState:
        """Per-env view of progress toward the completion criteria.

        is_complete and score are passed in (rather than recomputed here) so the full
        (num_envs,) tensor reductions run once per runner in
        ProgressTracker, instead of once per env.
        """

        criteria = self.completion_criteria
        completed_sequences = 0
        active_predicates: dict[str, str | None] = {}
        # The active predicate for a sequence is the one at its current chain position. Any sequence
        # whose pointer has run off the end of the chain is complete (no active predicate).
        for sequence_name in criteria.sequence_names:
            predicate_chain = self.predicate_chains[sequence_name]
            cur_predicate_index = int(self.current_predicate_index[sequence_name][env_idx].item())
            if not bool(self.prerequisites_met[env_idx].item()):
                active_predicates[sequence_name] = None
            elif cur_predicate_index >= len(predicate_chain):
                active_predicates[sequence_name] = None
                completed_sequences += 1
            else:
                active_predicates[sequence_name] = _predicate_repr(predicate_chain[cur_predicate_index][0])

        return CompletionCriteriaState(
            completed_sequences=completed_sequences,
            total_sequences=len(criteria.sequence_names),
            score=float(score),
            is_complete=bool(is_complete),
            active_predicates=active_predicates,
            prerequisites_met=bool(self.prerequisites_met[env_idx].item()),
        )


class ProgressTracker:
    """Track predicate completion and coordinate a flat list of subtasks."""

    def __init__(
        self,
        completion_criteria: list[CompletionCriteria],
        num_envs: int,
        device,
        env=None,
        *,
        subtasks_are_sequential: bool = False,
        desired_subtask_success_state: list[bool | None] | None = None,
    ):
        assert completion_criteria, "Task success requires at least one set of completion criteria."
        criteria_names = [criteria.name for criteria in completion_criteria]
        assert len(set(criteria_names)) == len(criteria_names), "Completion criteria names must be unique."
        self.completion_criteria = completion_criteria
        self.num_envs = num_envs
        self.device = device
        self.runners = [
            CompletionCriteriaRunner(criteria, num_envs, device, env=env) for criteria in completion_criteria
        ]
        self._subtask_runners = self._group_runners_by_subtask(self.runners)
        assert not subtasks_are_sequential or self._subtask_runners, "Sequential tracking requires subtask indices."
        if desired_subtask_success_state is not None:
            assert self._subtask_runners, "Final subtask conditions require subtask indices."
            assert len(desired_subtask_success_state) == len(
                self._subtask_runners
            ), "Desired subtask states must have one entry per subtask."
            assert all(
                state is None or isinstance(state, bool) for state in desired_subtask_success_state
            ), "Desired subtask states must be True, False, or None."
            assert any(
                state is not None for state in desired_subtask_success_state
            ), "At least one subtask must participate in the success check."
        self.subtasks_are_sequential = subtasks_are_sequential
        self.desired_subtask_success_state = desired_subtask_success_state
        self._task_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._events: list[list[PredicateEvent]] = [[] for _ in range(num_envs)]
        self._last_processed_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self._requires_step_index = any(runner.requires_step_index for runner in self.runners)

    @staticmethod
    def _group_runners_by_subtask(runners: list[CompletionCriteriaRunner]) -> list[list[CompletionCriteriaRunner]]:
        """Group runners by the subtask indices assigned to their criteria sets by CompositeTaskBase."""
        subtask_indices = [runner.completion_criteria.parent_subtask_idx for runner in runners]
        if all(index is None for index in subtask_indices):
            return []
        assert all(index is not None for index in subtask_indices), "Every set of criteria must have a subtask index."
        num_subtasks = len(set(subtask_indices))
        assert set(subtask_indices) == set(range(num_subtasks)), "Subtask indices must be consecutive from zero."
        runners_by_subtask: list[list[CompletionCriteriaRunner]] = [[] for _ in range(num_subtasks)]
        for runner in runners:
            subtask_index = runner.completion_criteria.parent_subtask_idx
            runners_by_subtask[subtask_index].append(runner)
        return runners_by_subtask

    @staticmethod
    def _all_criteria_complete(runners: list[CompletionCriteriaRunner]) -> torch.Tensor:
        return torch.stack([runner.is_complete() for runner in runners], dim=1).all(dim=1)

    def step(self, env, step_index: torch.Tensor | None = None) -> None:
        """Advance predicate sequences and update task success for one control step.

        TaskSuccessTerm calls this once per control step. Other consumers read
        is_complete(), get_state(), or get_events() without advancing progress.
        Temporal requirements need a per-environment step_index. When supplied,
        indices must advance by exactly one between updates for each environment,
        except on its first update after construction or reset.
        """

        assert (
            step_index is not None or not self._requires_step_index
        ), "Stateful temporal predicates require a per-environment step_index."
        if step_index is not None:
            assert step_index.shape == (self.num_envs,), "step_index must contain one index per environment."
            assert step_index.dtype in (torch.int32, torch.int64), "step_index must contain integer indices."
            step_index = step_index.to(device=self.device)
            assert bool((step_index >= 0).all()), "step_index must be non-negative."
            first_update = self._last_processed_step < 0
            next_control_step = step_index == self._last_processed_step + 1
            assert bool(
                (first_update | next_control_step).all()
            ), "step_index must advance by exactly one per environment between resets."
        active_envs = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        # Progress advancement and final-condition checks share predicate results.
        # Evaluating a stateful predicate twice could advance its counter twice
        # without another simulation step.
        evaluation = _PredicateEvaluation(env, self.num_envs, self.device)
        for subtask_index, subtask_runners in enumerate(self._subtask_runners or [self.runners]):
            # Use completion before advancing so the next subtask starts on the following step.
            subtask_was_complete = self._all_criteria_complete(subtask_runners)
            check_final_conditions = (
                self.desired_subtask_success_state is not None
                and self.desired_subtask_success_state[subtask_index] is not None
            )
            for runner in subtask_runners:
                for event in runner.step(evaluation, step_index, active_envs, check_final_conditions):
                    self._events[event.env_idx].append(event)
            if self.subtasks_are_sequential:
                active_envs = active_envs & subtask_was_complete
        self._task_success = self._compute_task_success(evaluation)
        if step_index is not None:
            # The environment increments episode_length_buf in place; keep our own snapshot.
            self._last_processed_step.copy_(step_index)

    def _compute_task_success(self, evaluation: _PredicateEvaluation) -> torch.Tensor:
        """Combine recorded completion with any required current subtask conditions."""
        if self.desired_subtask_success_state is None:
            return self._all_criteria_complete(self.runners)

        # Preserve the existing 'don't care' behavior: None skips both history and final state.
        required_subtasks = [
            (runners, desired_state)
            for runners, desired_state in zip(self._subtask_runners, self.desired_subtask_success_state)
            if desired_state is not None
        ]
        success = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        for runners, _ in required_subtasks:
            success &= self._all_criteria_complete(runners)
        for runners, desired_state in required_subtasks:
            final_conditions_met = torch.stack(
                [runner.final_conditions_met(evaluation) for runner in runners], dim=1
            ).all(dim=1)
            success &= final_conditions_met == desired_state
        return success

    def is_complete(self) -> torch.Tensor:
        """Return task success from the latest step without evaluating predicates again."""
        return self._task_success.clone()

    def get_subtask_completion(self) -> torch.Tensor:
        """Return recorded completion for each environment and subtask, in subtask order."""
        assert self._subtask_runners, "Subtask completion requires criteria sets with subtask indices."
        return torch.stack([self._all_criteria_complete(runners) for runners in self._subtask_runners], dim=1)

    def get_predicate(
        self,
        criteria_name: str,
        sequence_name: str = DEFAULT_SEQUENCE_NAME,
        predicate_index: int = 0,
    ) -> Callable | StatefulPredicate:
        """Return the resolved predicate for reading diagnostics without evaluating it.

        Args:
            criteria_name: Name of the CompletionCriteria containing the predicate.
            sequence_name: Named sequence, or the default sequence for a list definition.
            predicate_index: Position of the predicate within that sequence.
        """
        for runner in self.runners:
            if runner.completion_criteria.name == criteria_name:
                predicate = runner.predicate_chains[sequence_name][predicate_index][0]
                while isinstance(predicate, _TrueForConsecutiveSteps):
                    predicate = predicate.predicate
                while isinstance(predicate, functools.partial):
                    predicate = predicate.func
                return predicate
        raise KeyError(f"Unknown completion criteria: {criteria_name!r}")

    def reset(self, env_ids: list[int] | torch.Tensor) -> None:
        """Clear progress and events for the specified environment IDs."""

        if torch.is_tensor(env_ids):
            env_ids = env_ids.tolist()
        self._task_success[env_ids] = False
        self._last_processed_step[env_ids] = -1
        for runner in self.runners:
            runner.reset(env_ids)
        for env_idx in env_ids:
            self._events[env_idx] = []

    def get_state(self) -> list[ProgressState]:
        """Get the progress state of all CompletionCriteria definitions for each env."""

        # Compute the per-runner (num_envs,) tensors once
        completeness = [runner.is_complete() for runner in self.runners]
        scores = [runner.overall_score_per_env() for runner in self.runners]
        task_complete = self.is_complete()

        # Total criteria weight for normalization.
        total_criteria_weight = sum(runner.completion_criteria.score for runner in self.runners)

        output: list[ProgressState] = []
        for env_idx in range(self.num_envs):
            # Build a per-env state from each runner's state.
            criteria_states: dict[str, CompletionCriteriaState] = {}
            for i, runner in enumerate(self.runners):
                criteria = runner.completion_criteria
                state = runner.get_state_for_env(env_idx, completeness[i][env_idx], scores[i][env_idx])
                criteria_states[criteria.name] = state
            weighted_score = sum(
                runner.completion_criteria.score * float(score[env_idx]) for runner, score in zip(self.runners, scores)
            )

            overall_score = (
                max(0.0, min(1.0, weighted_score / total_criteria_weight)) if total_criteria_weight > 0 else 0.0
            )
            output.append(
                ProgressState(
                    criteria_by_name=criteria_states,
                    overall_score=overall_score,
                    all_complete=bool(task_complete[env_idx]),
                )
            )
        return output

    def get_events(self) -> list[list[PredicateEvent]]:
        """Get all events for all envs."""

        return [list(e) for e in self._events]


class ProgressTrackingRecorder(RecorderTerm):
    """Publish the tracker state and events after termination computation. Records nothing.

    Registered as a recorder term so it runs once per env.step via
    record_post_step. It publishes the per-step state/events to
    env.extras["progress_tracking"], then returns
    (None, None) so nothing is written to the recorded episode data.

    env.extras["progress_tracking"] format:

        {
            "states": [                                    # one ProgressState per env
                ProgressState(
                    criteria_by_name={
                        "<name>": CompletionCriteriaState(
                            completed_sequences, total_sequences, score, is_complete, active_predicates
                        ),
                        ...
                    },
                    overall_score=float,                   # weighted mean of criteria scores, in [0, 1]
                    all_complete=bool,
                ),
                ...
            ],
            "events": [                                    # one list of PredicateEvent per env
                [PredicateEvent(env_idx, step, criteria_name, sequence_name,
                                predicate_index, predicate_name, score_delta), ...],
                ...
            ],
        }
    """

    def record_post_step(self):
        """Publish the current progress snapshot without advancing the tracker."""

        progress_tracker = self._env.progress_tracker
        assert progress_tracker is not None, "Task success must initialize the progress tracker before recording."
        self._env.extras["progress_tracking"] = {
            "states": progress_tracker.get_state(),
            "events": progress_tracker.get_events(),
        }
        # This term is a per-step hook only — record nothing.
        return None, None


@configclass
class ProgressTrackingRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = ProgressTrackingRecorder


@configclass
class ProgressTrackingRecorderManagerCfg(RecorderManagerBaseCfg):
    progress_tracking: ProgressTrackingRecorderCfg = ProgressTrackingRecorderCfg()

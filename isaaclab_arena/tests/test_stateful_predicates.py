# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Stateful predicate occurrences retain their own activation and episode state."""

from functools import partial
from types import SimpleNamespace

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _make_environment(heights=(0.5, 0.7)):
    import torch

    num_envs = len(heights)
    positions = torch.zeros(num_envs, 3)
    positions[:, 2] = torch.tensor(heights)
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        positions=positions,
        ready=torch.ones(num_envs, dtype=torch.bool),
        episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
        position_reads=0,
        predicate_calls=0,
    )

    def get_position_w(object_name):
        assert object_name == "object"
        env.position_reads += 1
        return env.positions

    env.arena_world = SimpleNamespace(get_position_w=get_position_w)
    return env


def _ready(env):
    return env.ready


def _step(tracker, env):
    env.episode_length_buf += 1
    tracker.step(env, step_index=env.episode_length_buf)


def _test_reused_configuration_has_independent_sequence_occurrences(_simulation_app):
    from isaaclab.managers import TerminationTermCfg

    from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.predicates.object_lifted import ObjectLifted
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    env = _make_environment()
    lifted = TerminationTermCfg(func=ObjectLifted, params={"object_name": "object"})
    held_lift = TrueForConsecutiveStepsCfg(lifted, required_steps=2)
    criteria = CompletionCriteria(name="lift_twice", predicate_sequence=[held_lift, held_lift])
    tracker = ProgressTracker([criteria], env.num_envs, env.device, env=env)

    _step(tracker, env)
    env.positions[:, 2] += 0.1
    _step(tracker, env)
    assert not tracker.is_complete().any()
    assert not any(tracker.get_events())
    _step(tracker, env)
    assert all(state.overall_score == 0.5 for state in tracker.get_state())

    # The second occurrence starts with a new reference at the already raised height.
    for _ in range(2):
        _step(tracker, env)
        assert all(state.overall_score == 0.5 for state in tracker.get_state())
        assert all(len(events) == 1 for events in tracker.get_events())

    env.positions[:, 2] += 0.1
    _step(tracker, env)
    assert not tracker.is_complete().any()
    _step(tracker, env)
    assert tracker.is_complete().all()
    for events in tracker.get_events():
        assert [event.step for event in events] == [3, 7]
    return True


def _test_reused_configuration_has_independent_tracker_state(_simulation_app):
    from isaaclab.managers import TerminationTermCfg

    from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.predicates.object_lifted import ObjectLifted
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    lifted = TerminationTermCfg(func=ObjectLifted, params={"object_name": "object"})
    held_lift = TrueForConsecutiveStepsCfg(lifted, required_steps=2)
    criteria = CompletionCriteria(name="held_lift", predicate_sequence=[held_lift])
    first_env = _make_environment((0.5,))
    second_env = _make_environment((1.0,))
    first_tracker = ProgressTracker([criteria], first_env.num_envs, first_env.device, env=first_env)
    second_tracker = ProgressTracker([criteria], second_env.num_envs, second_env.device, env=second_env)

    _step(first_tracker, first_env)
    first_env.positions[:, 2] += 0.1
    _step(first_tracker, first_env)
    assert not first_tracker.is_complete().item()

    _step(second_tracker, second_env)
    assert not second_tracker.is_complete().item()
    _step(first_tracker, first_env)
    assert first_tracker.is_complete().item()
    _step(second_tracker, second_env)
    assert not second_tracker.is_complete().item()

    second_env.positions[:, 2] += 0.1
    _step(second_tracker, second_env)
    assert not second_tracker.is_complete().item()
    _step(second_tracker, second_env)
    assert second_tracker.is_complete().item()
    assert first_tracker.get_events()[0][0].step == 3
    assert second_tracker.get_events()[0][0].step == 4
    return True


def _test_wrapped_lift_activates_and_resets_per_environment(_simulation_app):
    from isaaclab.managers import TerminationTermCfg

    from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.predicates.object_lifted import ObjectLifted
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    env = _make_environment()
    env.ready[1] = False
    lifted = TerminationTermCfg(func=ObjectLifted, params={"object_name": "object"})
    criteria = CompletionCriteria(
        name="held_lift",
        prerequisites=[_ready],
        predicate_sequence=[TrueForConsecutiveStepsCfg(lifted, required_steps=2)],
        parent_subtask_idx=0,
    )
    tracker = ProgressTracker([criteria], env.num_envs, env.device, env=env, desired_subtask_success_state=[True])

    _step(tracker, env)
    env.positions[:, 2] += 0.1
    _step(tracker, env)
    assert tracker.is_complete().tolist() == [False, False]
    env.ready[1] = True
    _step(tracker, env)
    assert tracker.is_complete().tolist() == [True, False]
    env.positions[1, 2] += 0.1
    _step(tracker, env)
    assert tracker.is_complete().tolist() == [True, False]
    _step(tracker, env)
    assert tracker.is_complete().tolist() == [True, True]
    second_env_events = tracker.get_events()[1]
    assert second_env_events[0].step == 5

    tracker.reset([0])
    env.episode_length_buf[0] = 0
    env.ready[0] = False
    env.positions[0, 2] += 1.0
    _step(tracker, env)
    assert tracker.is_complete().tolist() == [False, True]
    assert not tracker.get_state()[0].criteria_by_name["held_lift"].prerequisites_met

    env.ready[0] = True
    for _ in range(2):
        _step(tracker, env)
        assert tracker.is_complete().tolist() == [False, True]
    env.positions[0, 2] += 0.1
    _step(tracker, env)
    assert tracker.is_complete().tolist() == [False, True]
    _step(tracker, env)
    assert tracker.is_complete().tolist() == [True, True]
    assert tracker.get_events()[0][0].step == 5
    assert tracker.get_events()[1] == second_env_events
    return True


def _test_nested_temporal_lift_updates_once_during_final_condition_checks(_simulation_app):
    from isaaclab.managers import TerminationTermCfg

    from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.predicates.object_lifted import ObjectLifted
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    env = _make_environment((0.5,))
    lifted = TerminationTermCfg(func=ObjectLifted, params={"object_name": "object"})
    held_lift = TrueForConsecutiveStepsCfg(lifted, required_steps=2)
    criteria = CompletionCriteria(
        name="held_lift",
        predicate_sequence=[TrueForConsecutiveStepsCfg(held_lift, required_steps=2)],
        parent_subtask_idx=0,
    )
    tracker = ProgressTracker([criteria], env.num_envs, env.device, env=env, desired_subtask_success_state=[True])

    _step(tracker, env)
    env.positions[:, 2] += 0.1
    for expected_step in (2, 3):
        _step(tracker, env)
        for _ in range(3):
            assert not tracker.is_complete().item()
            assert not tracker.get_state()[0].all_complete
            assert not tracker.get_events()[0]
            tracker.get_predicate("held_lift")
        assert env.position_reads == expected_step
    _step(tracker, env)
    assert tracker.is_complete().item()
    assert env.position_reads == 4

    env.positions[:, 2] -= 0.1
    assert tracker.is_complete().item(), "Readers retain the result of the last control step."
    assert tracker.get_state()[0].all_complete
    assert env.position_reads == 4
    _step(tracker, env)
    assert not tracker.is_complete().item()
    assert tracker.get_subtask_completion().tolist() == [[True]]

    env.positions[:, 2] += 0.1
    for _ in range(2):
        _step(tracker, env)
        assert not tracker.is_complete().item()
    _step(tracker, env)
    assert tracker.is_complete().item()
    assert env.position_reads == 8
    assert [event.step for event in tracker.get_events()[0]] == [4]
    return True


def _test_nested_requirements_share_plain_callable_results_without_resetting_it(_simulation_app):
    from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    class _ReadyPredicate:
        def __call__(self, env):
            env.predicate_calls += 1
            return env.ready

        def reset(self, env_ids=None):
            raise AssertionError("Plain callable predicates do not participate in the stateful lifecycle.")

    env = _make_environment()
    ready = _ReadyPredicate()
    held_ready = TrueForConsecutiveStepsCfg(ready, required_steps=2)
    criteria = CompletionCriteria(
        name="ready",
        prerequisites=[ready],
        predicate_sequences={
            "instantaneous": [ready],
            "held": [TrueForConsecutiveStepsCfg(held_ready, required_steps=2)],
        },
    )
    tracker = ProgressTracker([criteria], env.num_envs, env.device, env=env)
    for expected_calls in range(1, 4):
        _step(tracker, env)
        assert env.predicate_calls == expected_calls
        assert tracker.is_complete().tolist() == [expected_calls == 3, expected_calls == 3]

    tracker.reset([0])
    env.episode_length_buf[0] = 0
    for expected_calls in range(4, 7):
        _step(tracker, env)
        assert env.predicate_calls == expected_calls
        assert tracker.is_complete().tolist() == [expected_calls == 6, True]
    return True


def _test_stateful_declarations_reject_runtime_instances_and_invalid_parameters(_simulation_app):
    import pytest
    from isaaclab.managers import TerminationTermCfg

    from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.predicates.object_lifted import ObjectLifted
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    class _CallablePredicate:
        def __call__(self, env):
            return env.ready

    env = _make_environment()
    lifted = TerminationTermCfg(func=ObjectLifted, params={"object_name": "object"})
    source_criteria = CompletionCriteria(name="source", predicate_sequence=[lifted])
    source_tracker = ProgressTracker([source_criteria], env.num_envs, env.device, env=env)
    runtime = source_tracker.get_predicate("source")
    configured_callable = partial(runtime, object_name="object")
    for live_predicate in (runtime, configured_callable):
        configured_runtime = TerminationTermCfg(func=live_predicate)
        invalid_criteria = CompletionCriteria(name="configured_runtime", predicate_sequence=[configured_runtime])
        with pytest.raises(AssertionError):
            ProgressTracker([invalid_criteria], env.num_envs, env.device, env=env)
    for invalid_predicate in (runtime, configured_callable, _CallablePredicate, ObjectLifted):
        with pytest.raises(AssertionError):
            TrueForConsecutiveStepsCfg(invalid_predicate, required_steps=2)
        with pytest.raises(AssertionError):
            CompletionCriteria(name="invalid", predicate_sequence=[invalid_predicate])
        with pytest.raises(AssertionError):
            CompletionCriteria(name="invalid", prerequisites=[invalid_predicate], predicate_sequence=[_ready])

    for invalid_name in ("", None, 7):
        invalid_lift = TerminationTermCfg(func=ObjectLifted, params={"object_name": invalid_name})
        invalid_criteria = CompletionCriteria(name="invalid_lift", predicate_sequence=[invalid_lift])
        with pytest.raises(AssertionError, match="object_name"):
            ProgressTracker([invalid_criteria], env.num_envs, env.device, env=env)
    for invalid_distance in (0.0, -0.01, float("nan"), float("inf"), float("-inf")):
        invalid_lift = TerminationTermCfg(
            func=ObjectLifted, params={"object_name": "object", "distance": invalid_distance}
        )
        invalid_criteria = CompletionCriteria(name="invalid_lift", predicate_sequence=[invalid_lift])
        with pytest.raises(AssertionError, match="distance"):
            ProgressTracker([invalid_criteria], env.num_envs, env.device, env=env)

    held_lift = TrueForConsecutiveStepsCfg(lifted, required_steps=2)
    nested_requirement = TrueForConsecutiveStepsCfg(held_lift, required_steps=2)
    criteria = CompletionCriteria(name="valid", predicate_sequence=[nested_requirement])
    tracker = ProgressTracker([criteria], env.num_envs, env.device, env=env)
    _step(tracker, env)
    assert not tracker.is_complete().any()
    return True


def _test_partial_predicates_keep_distinct_configured_arguments(_simulation_app):
    from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    def matches_readiness(env, expected_ready):
        env.predicate_calls += 1
        return env.ready == expected_ready

    env = _make_environment()
    env.ready[1] = False
    ready = partial(matches_readiness, expected_ready=True)
    waiting = partial(matches_readiness, expected_ready=False)
    criteria = CompletionCriteria(
        name="readiness_alternatives",
        predicate_sequences={
            "ready": [TrueForConsecutiveStepsCfg(ready, required_steps=2)],
            "waiting": [TrueForConsecutiveStepsCfg(waiting, required_steps=2)],
        },
        logical="any",
    )
    tracker = ProgressTracker([criteria], env.num_envs, env.device, env=env)
    _step(tracker, env)
    assert env.predicate_calls == 2
    assert not tracker.is_complete().any()
    _step(tracker, env)
    assert env.predicate_calls == 4
    assert tracker.is_complete().all()
    first_env_events, second_env_events = tracker.get_events()
    assert [event.sequence_name for event in first_env_events] == ["ready"]
    assert [event.sequence_name for event in second_env_events] == ["waiting"]
    return True


def _test_ordinary_manager_terms_keep_their_existing_call_and_reset_contract(_simulation_app):
    from isaaclab.managers import ManagerTermBase, TerminationTermCfg

    from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    class _ReadyTerm(ManagerTermBase):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self.calls = 0

        def __call__(self, env, expected_ready=True):
            self.calls += 1
            return env.ready == expected_ready

        def reset(self, env_ids=None):
            raise AssertionError("Existing manager terms retain their external reset lifecycle.")

    env = _make_environment()
    ready_cfg = TerminationTermCfg(func=_ReadyTerm, params={"expected_ready": True})
    criteria_sets = [
        CompletionCriteria(name="ready", predicate_sequence=[ready_cfg]),
        CompletionCriteria(
            name="held_ready", predicate_sequence=[TrueForConsecutiveStepsCfg(ready_cfg, required_steps=2)]
        ),
    ]
    tracker = ProgressTracker(criteria_sets, env.num_envs, env.device, env=env)
    ready_term = tracker.get_predicate("ready")
    held_ready_term = tracker.get_predicate("held_ready")
    assert isinstance(ready_term, _ReadyTerm)
    assert isinstance(held_ready_term, _ReadyTerm)
    assert ready_term is not held_ready_term
    assert ready_cfg.func is _ReadyTerm

    _step(tracker, env)
    assert not tracker.is_complete().any()
    assert ready_term.calls == 1
    assert held_ready_term.calls == 1
    _step(tracker, env)
    assert tracker.is_complete().all()
    assert ready_term.calls == 1
    assert held_ready_term.calls == 2

    tracker.reset([0])
    env.episode_length_buf[0] = 0
    _step(tracker, env)
    assert tracker.is_complete().tolist() == [False, True]
    _step(tracker, env)
    assert tracker.is_complete().all()
    assert ready_term.calls == 2
    assert held_ready_term.calls == 4
    return True


def test_reused_configuration_has_independent_sequence_occurrences():
    assert run_function_with_persistent_simulation_app(_test_reused_configuration_has_independent_sequence_occurrences)


def test_reused_configuration_has_independent_tracker_state():
    assert run_function_with_persistent_simulation_app(_test_reused_configuration_has_independent_tracker_state)


def test_wrapped_lift_activates_and_resets_per_environment():
    assert run_function_with_persistent_simulation_app(_test_wrapped_lift_activates_and_resets_per_environment)


def test_nested_temporal_lift_updates_once_during_final_condition_checks():
    assert run_function_with_persistent_simulation_app(
        _test_nested_temporal_lift_updates_once_during_final_condition_checks
    )


def test_nested_requirements_share_plain_callable_results_without_resetting_it():
    assert run_function_with_persistent_simulation_app(
        _test_nested_requirements_share_plain_callable_results_without_resetting_it
    )


def test_stateful_declarations_reject_runtime_instances_and_invalid_parameters():
    assert run_function_with_persistent_simulation_app(
        _test_stateful_declarations_reject_runtime_instances_and_invalid_parameters
    )


def test_partial_predicates_keep_distinct_configured_arguments():
    assert run_function_with_persistent_simulation_app(_test_partial_predicates_keep_distinct_configured_arguments)


def test_ordinary_manager_terms_keep_their_existing_call_and_reset_contract():
    assert run_function_with_persistent_simulation_app(
        _test_ordinary_manager_terms_keep_their_existing_call_and_reset_contract
    )

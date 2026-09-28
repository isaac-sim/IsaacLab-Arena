# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


class _ControlledPredicate:
    def __init__(self, values: list[bool], name: str = "condition"):
        self.values = values
        self.__name__ = name
        self.calls = 0

    def __call__(self, env):
        import torch

        self.calls += 1
        return torch.tensor(self.values, dtype=torch.bool)


def _step(tracker, env, step_indices: list[int]):
    import torch

    tracker.step(env, step_index=torch.tensor(step_indices, dtype=torch.long))


def _test_runtime_requirement_updates_only_active_environments(simulation_app):
    import torch

    from isaaclab_arena.tasks.predicates.temporal import _TrueForConsecutiveSteps

    predicate = _ControlledPredicate([True, True, True])
    requirement = _TrueForConsecutiveSteps(predicate=predicate, required_steps=2, num_envs=3, device="cpu")
    samples = [
        ([True, True, True], [True, False, True], [False, False, False]),
        # Inactive false results must not clear an existing count.
        ([False, True, True], [False, False, True], [False, False, True]),
        ([True, True, False], [True, False, False], [True, False, True]),
        # The second environment has been inactive and starts counting only now.
        ([True, True, True], [False, True, False], [True, False, True]),
        ([True, True, False], [False, True, True], [True, True, False]),
    ]
    for predicate_results, active_envs, expected_completion in samples:
        completion = requirement.update(torch.tensor(predicate_results), torch.tensor(active_envs))
        assert completion.tolist() == expected_completion
    assert predicate.calls == 0
    return True


def _test_interrupted_streaks_complete_independently(simulation_app):
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    predicate = _ControlledPredicate([False, False])
    requirement = TrueForConsecutiveStepsCfg(predicate=predicate, required_steps=3)
    objective = ProgressObjective(name="hold", predicate_sequence=[requirement])
    tracker = ProgressTracker([objective], num_envs=2, device="cpu")
    env = SimpleNamespace(num_envs=2, device="cpu")
    samples = [
        ([True, False], [False, False]),
        ([True, True], [False, False]),
        ([False, True], [False, False]),
        ([True, True], [False, True]),
        ([True, False], [False, True]),
        ([True, False], [True, True]),
    ]
    for step_index, (values, expected_completion) in enumerate(samples, start=1):
        predicate.values = values
        _step(tracker, env, [step_index, step_index])
        assert tracker.is_complete().tolist() == expected_completion

    assert [event.step for event in tracker.get_events()[0]] == [6]
    assert [event.step for event in tracker.get_events()[1]] == [4]
    calls_after_updates = predicate.calls
    tracker.get_state()
    tracker.get_events()
    tracker.is_complete()
    assert predicate.calls == calls_after_updates
    return True


def _test_middle_requirement_starts_when_reached(simulation_app):
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    lifted = _ControlledPredicate([False], name="lifted")
    resting = _ControlledPredicate([True], name="resting")
    placed = _ControlledPredicate([True], name="placed")
    objective = ProgressObjective(
        name="pick_and_place",
        predicate_sequence=[
            lifted,
            TrueForConsecutiveStepsCfg(resting, required_steps=2),
            placed,
        ],
    )
    tracker = ProgressTracker([objective], num_envs=1, device="cpu")
    env = SimpleNamespace(num_envs=1, device="cpu")
    _step(tracker, env, [1])
    _step(tracker, env, [2])
    lifted.values = [True]
    _step(tracker, env, [3])
    assert resting.calls == 0
    _step(tracker, env, [4])
    assert len(tracker.get_events()[0]) == 1
    _step(tracker, env, [5])
    assert placed.calls == 0
    assert not tracker.is_complete().item()

    # Completing the rest requirement is remembered even if the object moves later.
    resting.values = [False]
    _step(tracker, env, [6])
    assert tracker.is_complete().item()
    assert resting.calls == 2
    assert [(event.predicate_index, event.step) for event in tracker.get_events()[0]] == [(0, 3), (1, 5), (2, 6)]
    return True


def _test_joint_conditions_require_overlapping_steps(simulation_app):
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    object_a_resting = _ControlledPredicate([True])
    object_b_touching = _ControlledPredicate([False])

    def both_conditions_hold(env):
        return object_a_resting(env) & object_b_touching(env)

    requirement = TrueForConsecutiveStepsCfg(both_conditions_hold, required_steps=2)
    objective = ProgressObjective(name="rest_and_touch", predicate_sequence=[requirement])
    tracker = ProgressTracker([objective], num_envs=1, device="cpu")
    env = SimpleNamespace(num_envs=1, device="cpu")
    samples = [
        (True, False),
        (False, True),
        (True, True),
        (True, False),
        (True, True),
        (True, True),
    ]
    for step_index, (resting, touching) in enumerate(samples, start=1):
        object_a_resting.values = [resting]
        object_b_touching.values = [touching]
        _step(tracker, env, [step_index])
        assert tracker.is_complete().item() == (step_index == 6)
    return True


def _test_reused_requirement_has_independent_counters(simulation_app):
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    held = _ControlledPredicate([True])
    enabled = _ControlledPredicate([False])
    requirement = TrueForConsecutiveStepsCfg(held, required_steps=2)
    objectives = [
        ProgressObjective(name="twice", predicate_sequence=[requirement, requirement]),
        ProgressObjective(name="delayed", predicate_sequence=[enabled, requirement]),
    ]
    tracker = ProgressTracker(objectives, num_envs=1, device="cpu")
    assert requirement.predicate is held
    assert requirement.required_steps == 2
    assert tracker.get_predicate("twice") is held
    assert tracker.get_predicate("twice", predicate_index=1) is held
    assert tracker.get_predicate("delayed", predicate_index=1) is held
    assert held.calls == 0
    env = SimpleNamespace(num_envs=1, device="cpu")
    _step(tracker, env, [1])
    enabled.values = [True]
    _step(tracker, env, [2])
    _step(tracker, env, [3])
    assert not tracker.is_complete().item()
    assert len(tracker.get_events()[0]) == 2
    _step(tracker, env, [4])
    assert tracker.is_complete().item()
    assert [(event.progress_objective, event.predicate_index, event.step) for event in tracker.get_events()[0]] == [
        ("twice", 0, 2),
        ("delayed", 0, 2),
        ("twice", 1, 4),
        ("delayed", 1, 4),
    ]
    assert held.calls == 4

    other_objective = ProgressObjective(name="other_tracker", predicate_sequence=[requirement])
    other_tracker = ProgressTracker([other_objective], num_envs=1, device="cpu")
    _step(other_tracker, env, [4])
    assert not other_tracker.is_complete().item()
    _step(other_tracker, env, [5])
    assert other_tracker.is_complete().item()
    return True


def _test_named_sequences_keep_independent_counters(simulation_app):
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    held = _ControlledPredicate([True])
    enabled = _ControlledPredicate([True])
    requirement = TrueForConsecutiveStepsCfg(held, required_steps=2)
    objective = ProgressObjective(
        name="parallel",
        predicate_sequences={
            "immediate": [requirement],
            "delayed": [enabled, requirement],
        },
    )
    tracker = ProgressTracker([objective], num_envs=1, device="cpu")
    env = SimpleNamespace(num_envs=1, device="cpu")
    _step(tracker, env, [1])
    _step(tracker, env, [2])
    assert not tracker.is_complete().item()
    assert tracker.get_state()[0].progress_objectives["parallel"].completed_groups == 1
    _step(tracker, env, [3])
    assert tracker.is_complete().item()
    return True


def _test_partial_reset_and_duplicate_steps(simulation_app):
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    predicate = _ControlledPredicate([True, True])
    objective = ProgressObjective(name="hold", predicate_sequence=[TrueForConsecutiveStepsCfg(predicate, 2)])
    tracker = ProgressTracker([objective], num_envs=2, device="cpu")
    env = SimpleNamespace(num_envs=2, device="cpu")
    _step(tracker, env, [1, 1])
    _step(tracker, env, [1, 1])
    assert predicate.calls == 1
    assert tracker.is_complete().tolist() == [False, False]
    _step(tracker, env, [1, 2])
    assert tracker.is_complete().tolist() == [False, True]
    _step(tracker, env, [2, 2])
    assert tracker.is_complete().tolist() == [True, True]

    tracker.reset([0])
    assert tracker.get_events()[0] == []
    assert len(tracker.get_events()[1]) == 1
    _step(tracker, env, [2, 2])
    assert tracker.is_complete().tolist() == [False, True]
    _step(tracker, env, [2, 2])
    assert tracker.is_complete().tolist() == [False, True]
    _step(tracker, env, [3, 2])
    assert tracker.is_complete().tolist() == [True, True]

    tracker.reset([0, 1])
    _step(tracker, env, [3, 2])
    assert tracker.is_complete().tolist() == [False, False]
    assert tracker.get_events() == [[], []]
    tracker.reset([0])
    _step(tracker, env, [4, 3])
    assert tracker.is_complete().tolist() == [False, True]
    _step(tracker, env, [5, 3])
    assert tracker.is_complete().tolist() == [True, True]
    return True


def _test_duplicate_steps_do_not_advance_the_sequence(simulation_app):
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    instant = _ControlledPredicate([True])
    held = _ControlledPredicate([True])
    objective = ProgressObjective(
        name="sequence",
        predicate_sequence=[instant, instant, TrueForConsecutiveStepsCfg(held, 2)],
    )
    tracker = ProgressTracker([objective], num_envs=1, device="cpu")
    env = SimpleNamespace(num_envs=1, device="cpu")
    for step_index in (1, 2):
        _step(tracker, env, [step_index])
        _step(tracker, env, [step_index])
        assert len(tracker.get_events()[0]) == step_index
        assert held.calls == 0
    _step(tracker, env, [3])
    _step(tracker, env, [3])
    assert not tracker.is_complete().item()
    assert held.calls == 1
    _step(tracker, env, [4])
    assert tracker.is_complete().item()
    return True


def _test_final_requirement_loses_and_reacquires_its_streak(simulation_app):
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    resting = _ControlledPredicate([True])
    placed = _ControlledPredicate([True])
    objectives = [
        ProgressObjective(
            name="rest",
            predicate_sequence=[TrueForConsecutiveStepsCfg(resting, 2)],
            parent_subtask_idx=0,
        ),
        ProgressObjective(name="place", predicate_sequence=[placed], parent_subtask_idx=1),
    ]
    tracker = ProgressTracker(
        objectives,
        num_envs=1,
        device="cpu",
        subtasks_are_sequential=True,
        desired_subtask_success_state=[True, True],
    )
    env = SimpleNamespace(num_envs=1, device="cpu")
    _step(tracker, env, [1])
    assert tracker.get_subtask_completion().tolist() == [[False, False]]
    _step(tracker, env, [2])
    assert tracker.get_subtask_completion().tolist() == [[True, False]]
    resting.values = [False]
    _step(tracker, env, [3])
    assert tracker.get_subtask_completion().tolist() == [[True, True]]
    assert not tracker.is_complete().item()
    resting.values = [True]
    _step(tracker, env, [4])
    _step(tracker, env, [4])
    assert not tracker.is_complete().item()
    _step(tracker, env, [5])
    assert tracker.is_complete().item()
    assert resting.calls == 5

    resting.values = [False]
    _step(tracker, env, [5])
    assert tracker.is_complete().item()
    _step(tracker, env, [6])
    assert not tracker.is_complete().item()
    resting.values = [True]
    _step(tracker, env, [7])
    assert not tracker.is_complete().item()
    _step(tracker, env, [8])
    assert tracker.is_complete().item()
    assert len(tracker.get_events()[0]) == 2
    return True


def _test_final_rechecks_respect_per_environment_steps(simulation_app):
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    predicate = _ControlledPredicate([True, True])
    objective = ProgressObjective(
        name="hold",
        predicate_sequence=[TrueForConsecutiveStepsCfg(predicate, 2)],
        parent_subtask_idx=0,
    )
    tracker = ProgressTracker([objective], 2, "cpu", desired_subtask_success_state=[True])
    env = SimpleNamespace(num_envs=2, device="cpu")
    _step(tracker, env, [1, 1])
    _step(tracker, env, [2, 2])
    assert tracker.is_complete().tolist() == [True, True]
    predicate.values = [False, False]
    _step(tracker, env, [2, 3])
    assert tracker.is_complete().tolist() == [True, False]
    _step(tracker, env, [3, 3])
    assert tracker.is_complete().tolist() == [False, False]
    predicate.values = [True, True]
    _step(tracker, env, [4, 4])
    assert tracker.is_complete().tolist() == [False, False]
    _step(tracker, env, [4, 5])
    assert tracker.is_complete().tolist() == [False, True]
    assert tracker.get_subtask_completion().tolist() == [[True], [True]]
    return True


def _test_one_step_requirement_and_weighted_reporting(simulation_app):
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.progress_tracking.progress_tracking_utils import DEFAULT_GROUP_NAME
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    resting = _ControlledPredicate([False], name="object_is_resting")
    placed = _ControlledPredicate([True], name="placed")
    requirement = TrueForConsecutiveStepsCfg(resting, required_steps=1)
    objective = ProgressObjective(name="placement", predicate_sequence=[(requirement, 3.0), (placed, 1.0)])
    tracker = ProgressTracker([objective], num_envs=1, device="cpu")
    env = SimpleNamespace(num_envs=1, device="cpu")
    predicate_name = tracker.get_state()[0].progress_objectives["placement"].active_predicates[DEFAULT_GROUP_NAME]
    assert "object_is_resting" in predicate_name
    assert "TrueForConsecutiveStepsCfg" in predicate_name
    assert "1" in predicate_name
    _step(tracker, env, [1])
    assert tracker.get_state()[0].overall_score == 0.0
    resting.values = [True]
    _step(tracker, env, [2])
    assert tracker.get_state()[0].overall_score == 0.75
    assert tracker.get_events()[0][0].predicate_name == predicate_name
    _step(tracker, env, [3])
    assert tracker.is_complete().item()
    assert tracker.get_state()[0].overall_score == 1.0
    return True


def _test_requirement_validation(simulation_app):
    import torch

    import pytest

    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    predicate = _ControlledPredicate([True])
    for invalid_steps in (0, -1, True, False, 1.5, "2", None):
        with pytest.raises((AssertionError, TypeError, ValueError)):
            TrueForConsecutiveStepsCfg(predicate, required_steps=invalid_steps)
    requirement = TrueForConsecutiveStepsCfg(predicate, required_steps=2)
    for invalid_predicate in (None, 42, requirement, _ControlledPredicate):
        with pytest.raises((AssertionError, TypeError, ValueError)):
            TrueForConsecutiveStepsCfg(invalid_predicate, required_steps=2)
    assert not callable(requirement)

    objective = ProgressObjective(name="hold", predicate_sequence=[requirement])
    tracker = ProgressTracker([objective], num_envs=1, device="cpu")
    env = SimpleNamespace(num_envs=1, device="cpu", episode_length_buf=torch.tensor([1]))
    with pytest.raises(AssertionError, match="step_index"):
        tracker.step(env)
    assert predicate.calls == 0
    return True


def _test_skipped_control_steps_interrupt_the_streak(simulation_app):
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    predicate = _ControlledPredicate([True])
    objective = ProgressObjective(name="hold", predicate_sequence=[TrueForConsecutiveStepsCfg(predicate, 2)])
    tracker = ProgressTracker([objective], num_envs=1, device="cpu")
    env = SimpleNamespace(num_envs=1, device="cpu")
    _step(tracker, env, [1])
    assert not tracker.is_complete().item()
    _step(tracker, env, [3])
    assert not tracker.is_complete().item()
    _step(tracker, env, [4])
    assert tracker.is_complete().item()
    assert [event.step for event in tracker.get_events()[0]] == [4]
    return True


def _test_manager_updates_and_resets_requirement_counters(simulation_app):
    from functools import partial

    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg
    from isaaclab_arena.tests.test_task_success_from_progress import (
        _controlled_predicate,
        _make_environment_and_manager,
    )

    requirement = TrueForConsecutiveStepsCfg(partial(_controlled_predicate, predicate_name="resting"), required_steps=3)
    objective = ProgressObjective(name="rest", predicate_sequence=[requirement])
    env, manager, recorder = _make_environment_and_manager(["resting"], success_objectives=[objective])
    for step_index in (1, 2):
        env.episode_length_buf += 1
        manager.compute()
        manager.compute()
        assert manager.get_term("success").tolist() == [False, False]
        assert env.predicate_calls["resting"] == step_index
        recorder.record_post_step()
        assert env.predicate_calls["resting"] == step_index

    manager.reset(env_ids=[0])
    env.episode_length_buf[0] = 0
    manager.compute()
    assert manager.get_term("success").tolist() == [False, False]
    env.episode_length_buf += 1
    manager.compute()
    assert manager.get_term("success").tolist() == [False, True]
    env.episode_length_buf += 1
    manager.compute()
    assert manager.get_term("success").tolist() == [True, True]

    calls_after_completion = env.predicate_calls["resting"]
    recorder.record_post_step()
    manager.compute()
    assert env.predicate_calls["resting"] == calls_after_completion
    progress = env.extras["progress_tracking"]
    assert [state.all_complete for state in progress["states"]] == [True, True]
    assert [event.step for event in progress["events"][0]] == [2]
    assert [event.step for event in progress["events"][1]] == [3]
    return True


def _test_false_final_requirement_still_requires_recorded_completion(simulation_app):
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    predicate = _ControlledPredicate([False])
    objective = ProgressObjective(
        name="hold", predicate_sequence=[TrueForConsecutiveStepsCfg(predicate, 2)], parent_subtask_idx=0
    )
    tracker = ProgressTracker([objective], 1, "cpu", desired_subtask_success_state=[False])
    env = SimpleNamespace(num_envs=1, device="cpu")
    samples = [(False, False), (True, False), (True, False), (False, True), (True, True), (True, False)]
    for step_index, (condition_holds, expected_success) in enumerate(samples, start=1):
        predicate.values = [condition_holds]
        _step(tracker, env, [step_index])
        assert tracker.is_complete().item() == expected_success
        assert tracker.get_subtask_completion().tolist() == [[step_index >= 3]]
    assert len(tracker.get_events()[0]) == 1
    return True


def test_runtime_requirement_updates_only_active_environments():
    assert run_function_with_persistent_simulation_app(
        _test_runtime_requirement_updates_only_active_environments, headless=True
    )


def test_interrupted_streaks_complete_independently():
    assert run_function_with_persistent_simulation_app(_test_interrupted_streaks_complete_independently, headless=True)


def test_middle_requirement_starts_when_reached():
    assert run_function_with_persistent_simulation_app(_test_middle_requirement_starts_when_reached, headless=True)


def test_joint_conditions_require_overlapping_steps():
    assert run_function_with_persistent_simulation_app(_test_joint_conditions_require_overlapping_steps, headless=True)


def test_reused_requirement_has_independent_counters():
    assert run_function_with_persistent_simulation_app(_test_reused_requirement_has_independent_counters, headless=True)


def test_named_sequences_keep_independent_counters():
    assert run_function_with_persistent_simulation_app(_test_named_sequences_keep_independent_counters, headless=True)


def test_partial_reset_and_duplicate_steps():
    assert run_function_with_persistent_simulation_app(_test_partial_reset_and_duplicate_steps, headless=True)


def test_duplicate_steps_do_not_advance_the_sequence():
    assert run_function_with_persistent_simulation_app(_test_duplicate_steps_do_not_advance_the_sequence, headless=True)


def test_final_requirement_loses_and_reacquires_its_streak():
    assert run_function_with_persistent_simulation_app(
        _test_final_requirement_loses_and_reacquires_its_streak, headless=True
    )


def test_final_rechecks_respect_per_environment_steps():
    assert run_function_with_persistent_simulation_app(
        _test_final_rechecks_respect_per_environment_steps, headless=True
    )


def test_one_step_requirement_and_weighted_reporting():
    assert run_function_with_persistent_simulation_app(_test_one_step_requirement_and_weighted_reporting, headless=True)


def test_requirement_validation():
    assert run_function_with_persistent_simulation_app(_test_requirement_validation, headless=True)


def test_skipped_control_steps_interrupt_the_streak():
    assert run_function_with_persistent_simulation_app(_test_skipped_control_steps_interrupt_the_streak, headless=True)


def test_manager_updates_and_resets_requirement_counters():
    assert run_function_with_persistent_simulation_app(
        _test_manager_updates_and_resets_requirement_counters, headless=True
    )


def test_false_final_requirement_still_requires_recorded_completion():
    assert run_function_with_persistent_simulation_app(
        _test_false_final_requirement_still_requires_recorded_completion, headless=True
    )

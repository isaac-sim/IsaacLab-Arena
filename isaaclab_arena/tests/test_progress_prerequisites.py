# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Completion criteria prerequisites gate success sequences without contributing progress."""

from functools import partial

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_prerequisites_gate_without_scoring(simulation_app):
    from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
    from isaaclab_arena.tests.test_task_success_from_progress import (
        _controlled_predicate,
        _make_environment_and_manager,
    )

    criteria = CompletionCriteria(
        name="pick_and_place",
        prerequisites=[partial(_controlled_predicate, predicate_name=name) for name in ["settled", "ready"]],
        predicate_sequence=[partial(_controlled_predicate, predicate_name=name) for name in ["lifted", "placed"]],
    )
    env, manager, recorder = _make_environment_and_manager(
        ["settled", "ready", "lifted", "placed"], success_criteria=[criteria]
    )

    def step():
        env.episode_length_buf += 1
        manager.compute()
        recorder.record_post_step()
        return env.extras["progress_tracking"]

    env.predicate_results["settled"][:] = False
    progress = step()
    assert env.predicate_calls["lifted"] == 0
    assert all(state.overall_score == 0 for state in progress["states"])
    for state in progress["states"]:
        criteria_state = state.criteria_by_name["pick_and_place"]
        assert not criteria_state.prerequisites_met
        assert all(predicate is None for predicate in criteria_state.active_predicates.values())
    assert not any(progress["events"])
    env.predicate_results["settled"][:] = True
    env.predicate_results["ready"][:] = False
    step()
    assert env.predicate_calls["lifted"] == 0, "Prerequisites must hold together, not at different times."
    env.predicate_results["ready"][:] = True
    env.predicate_results["lifted"][:] = False
    progress = step()
    assert env.predicate_calls["lifted"] == 1, "The first predicate runs in the same update as readiness."
    assert all(state.overall_score == 0 for state in progress["states"])
    assert all(state.criteria_by_name["pick_and_place"].prerequisites_met for state in progress["states"])
    assert not any(progress["events"])
    prerequisite_calls = env.predicate_calls["settled"]
    env.predicate_results["settled"][:] = False
    env.predicate_results["lifted"][:] = True
    progress = step()
    assert all(state.overall_score == 0.5 for state in progress["states"])
    step()
    assert manager.get_term("success").all()
    assert env.predicate_calls["settled"] == prerequisite_calls
    recorder.record_post_step()
    env.progress_tracker.get_state()
    assert env.predicate_calls["settled"] == prerequisite_calls
    manager.reset([0])
    env.episode_length_buf[0] = 0
    progress = step()
    assert manager.get_term("success").tolist() == [False, True]
    assert progress["states"][0].overall_score == 0
    assert not progress["states"][0].criteria_by_name["pick_and_place"].prerequisites_met
    assert progress["states"][1].criteria_by_name["pick_and_place"].prerequisites_met
    assert not progress["events"][0]
    return True


def _test_temporal_prerequisites_follow_subtask_activation_and_reset(simulation_app):
    import pytest

    from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg
    from isaaclab_arena.tests.test_task_success_from_progress import (
        _controlled_predicate,
        _make_environment_and_manager,
    )

    criteria_sets = [
        CompletionCriteria(
            name="earlier",
            parent_subtask_idx=0,
            predicate_sequence=[partial(_controlled_predicate, predicate_name="earlier")],
        ),
        CompletionCriteria(
            name="later",
            parent_subtask_idx=1,
            prerequisites=[
                TrueForConsecutiveStepsCfg(partial(_controlled_predicate, predicate_name="ready"), required_steps=2)
            ],
            predicate_sequence=[partial(_controlled_predicate, predicate_name="later")],
        ),
    ]
    env, manager, _ = _make_environment_and_manager(
        ["earlier", "ready", "later"], success_criteria=criteria_sets, subtasks_are_sequential=True
    )

    def step():
        env.episode_length_buf += 1
        manager.compute()

    with pytest.raises(AssertionError, match="step_index"):
        env.progress_tracker.step(env)
    assert env.predicate_calls == {"earlier": 0, "ready": 0, "later": 0}
    env.predicate_results["earlier"][:] = False
    for _ in range(3):
        step()
    assert env.predicate_calls["ready"] == 0
    env.predicate_results["earlier"][0] = True
    step()
    assert env.predicate_calls["ready"] == 0
    assert not manager.get_term("success").any()
    step()
    assert not manager.get_term("success").any()
    step()
    assert manager.get_term("success").tolist() == [True, False]
    manager.reset([0])
    env.episode_length_buf[0] = 0
    step()
    assert not manager.get_term("success").any()
    step()
    assert not manager.get_term("success").any()
    step()
    assert manager.get_term("success").tolist() == [True, False]
    return True


def _test_prerequisites_validation(simulation_app):
    import pytest

    from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria

    def predicate(env):
        return True

    for invalid in (None, {}, [None], [(predicate, 1.0)]):
        with pytest.raises(AssertionError, match="prerequisites"):
            CompletionCriteria(name="invalid", predicate_sequence=[predicate], prerequisites=invalid)
    assert CompletionCriteria(name="default", predicate_sequence=[predicate]).prerequisites == []
    return True


def test_prerequisites_gate_without_scoring():
    assert run_function_with_persistent_simulation_app(_test_prerequisites_gate_without_scoring)


def test_temporal_prerequisites_follow_subtask_activation_and_reset():
    assert run_function_with_persistent_simulation_app(_test_temporal_prerequisites_follow_subtask_activation_and_reset)


def test_prerequisites_validation():
    assert run_function_with_persistent_simulation_app(_test_prerequisites_validation)

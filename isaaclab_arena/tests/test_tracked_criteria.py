# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tracked criteria record progress and events without changing task success or subtask order."""

from functools import partial

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _criteria(name, **kwargs):
    from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
    from isaaclab_arena.tests.test_task_success_from_progress import _controlled_predicate

    return CompletionCriteria(
        name=name, predicate_sequence=[partial(_controlled_predicate, predicate_name=name)], **kwargs
    )


def _test_tracked_criteria_never_end_the_episode(simulation_app):
    from isaaclab_arena.recording.progress_terms import record_progress_results
    from isaaclab_arena.tests.test_task_success_from_progress import _make_environment_and_manager

    env, manager, recorder = _make_environment_and_manager(
        ["arrived", "found", "fallen"],
        success_criteria=[
            _criteria("arrived"),
            _criteria("found", required_for_success=False),
            _criteria("fallen", required_for_success=False),
        ],
    )
    env.predicate_results["arrived"][:] = False
    env.predicate_results["fallen"][:] = False
    env.episode_length_buf += 1
    assert manager.compute().tolist() == [False, False]
    recorder.record_post_step()
    record = record_progress_results(env, env_id=0)["progress"]
    assert not record["all_complete"]
    assert record["overall_score"] == 0.0, "Completed tracked criteria must not raise the overall score."
    assert record["criteria_by_name"]["found"]["is_complete"]
    assert {name: criteria["required_for_success"] for name, criteria in record["criteria_by_name"].items()} == {
        "arrived": True,
        "found": False,
        "fallen": False,
    }
    assert [event["criteria_name"] for event in record["events"]] == ["found"]

    env.predicate_results["arrived"][:] = True
    env.episode_length_buf += 1
    assert manager.compute().tolist() == [True, True]
    recorder.record_post_step()
    record = record_progress_results(env, env_id=0)["progress"]
    assert record["all_complete"]
    assert record["overall_score"] == 1.0, "Incomplete tracked criteria must not lower the overall score."
    assert [event["criteria_name"] for event in record["events"]] == ["found", "arrived"]
    assert env.predicate_calls["found"] == 1, "Completed tracked criteria stop evaluating until reset."
    assert env.predicate_calls["fallen"] == 2

    env.predicate_results["fallen"][0] = True
    env.episode_length_buf += 1
    assert manager.compute().tolist() == [True, True]
    states = env.progress_tracker.get_state()
    assert [state.criteria_by_name["fallen"].is_complete for state in states] == [True, False]
    assert [state.overall_score for state in states] == [1.0, 1.0]

    manager.reset(env_ids=[0])
    env.episode_length_buf[0] = 0
    states = env.progress_tracker.get_state()
    assert [state.criteria_by_name["found"].is_complete for state in states] == [False, True]
    assert env.progress_tracker.get_events()[0] == []
    assert len(env.progress_tracker.get_events()[1]) == 2
    env.predicate_results["arrived"][0] = False
    env.episode_length_buf += 1
    assert manager.compute().tolist() == [False, True]
    assert [event.criteria_name for event in env.progress_tracker.get_events()[0]] == ["found", "fallen"]
    return True


def _test_tracked_criteria_complete_while_success_waits(simulation_app):
    import torch
    from types import SimpleNamespace

    from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg
    from isaaclab_arena.tests.test_task_success_from_progress import _controlled_predicate

    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        predicate_results={"gate": torch.ones(2, dtype=torch.bool), "held": torch.ones(2, dtype=torch.bool)},
        predicate_calls={"gate": 0, "held": 0},
    )
    held = TrueForConsecutiveStepsCfg(predicate=partial(_controlled_predicate, predicate_name="held"), required_steps=2)
    gate = partial(_controlled_predicate, predicate_name="gate")
    tracker = ProgressTracker(
        [
            CompletionCriteria(name="success", predicate_sequence=[gate, held]),
            CompletionCriteria(name="tracked", predicate_sequence=[held], required_for_success=False),
        ],
        num_envs=2,
        device="cpu",
    )
    # The success sequence reaches the held condition one step after the tracked sequence and
    # counts its own streak from there.
    for step in (1, 2):
        tracker.step(env, torch.full((2,), step, dtype=torch.long))
    assert [state.criteria_by_name["tracked"].is_complete for state in tracker.get_state()] == [True, True]
    overall_scores = [state.overall_score for state in tracker.get_state()]
    assert overall_scores == [0.5, 0.5], "The tracked set must not carry weight in overall_score."
    assert not tracker.is_complete().any()
    tracker.step(env, torch.full((2,), 3, dtype=torch.long))
    assert tracker.is_complete().all()
    assert env.predicate_calls["held"] == 3, "Both streaks must reuse one evaluation per step."
    return True


def _test_tracked_criteria_ignore_subtask_order(simulation_app):
    from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase
    from isaaclab_arena.tasks.no_task import NoTask
    from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
    from isaaclab_arena.tests.test_task_success_from_progress import _make_environment_and_manager

    class ObservedTask(NoTask):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def get_termination_cfg(self):
            return TaskTerminationCfg(
                timeout_s=2.0,
                success=[_criteria(self.name), _criteria("event", required_for_success=False)],
            )

    cfg = CompositeTaskBase(
        [ObservedTask("first"), ObservedTask("second")],
        subtasks_are_sequential=True,
        desired_subtask_success_state=[False, True],
    ).get_termination_cfg()
    assert [criteria.name for criteria in cfg.success] == [
        "subtask_0/first",
        "subtask_0/event",
        "subtask_1/second",
        "subtask_1/event",
    ]
    assert [criteria.required_for_success for criteria in cfg.success] == [True, False, True, False]
    env, manager, _ = _make_environment_and_manager(
        ["first", "second", "event"],
        success_criteria=cfg.success,
        subtasks_are_sequential=cfg.subtasks_are_sequential,
        desired_subtask_success_state=cfg.desired_subtask_success_state,
    )
    env.predicate_results["first"][:] = False
    env.predicate_results["event"][1] = False
    env.episode_length_buf += 1
    manager.compute()
    assert env.predicate_calls["second"] == 0, "The second subtask must wait for the first."
    assert env.progress_tracker.get_subtask_completion().tolist() == [[False, False], [False, False]]
    assert {event.criteria_name for event in env.progress_tracker.get_events()[0]} == {
        "subtask_0/event",
        "subtask_1/event",
    }
    assert env.progress_tracker.get_events()[1] == []

    env.predicate_results["first"][:] = True
    env.episode_length_buf += 1
    manager.compute()
    assert env.predicate_calls["second"] == 0
    # Neither the tracked set's completion nor its absence changes the first subtask's final condition.
    env.predicate_results["first"][:] = False
    env.episode_length_buf += 1
    manager.compute()
    assert manager.get_term("success").tolist() == [True, True]
    assert env.progress_tracker.get_subtask_completion().tolist() == [[True, True], [True, True]]
    return True


def _test_success_requires_a_required_criteria_set(simulation_app):
    import pytest

    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase
    from isaaclab_arena.tasks.no_task import NoTask
    from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

    tracked_only = [_criteria("found", required_for_success=False)]
    with pytest.raises(AssertionError, match="required_for_success=True"):
        ProgressTracker(tracked_only, num_envs=2, device="cpu")

    class TrackedOnlyTask(NoTask):
        def get_termination_cfg(self):
            return TaskTerminationCfg(timeout_s=1.0, success=tracked_only)

    with pytest.raises(AssertionError, match="Subtask 0 must define success criteria with required_for_success=True"):
        CompositeTaskBase([TrackedOnlyTask()]).get_termination_cfg()
    return True


def _test_event_details_are_recorded_at_the_transition(simulation_app):
    import json
    import torch
    from types import SimpleNamespace

    from isaaclab.managers import TerminationTermCfg

    from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.recording.progress_terms import record_progress_results
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    class ReasonPredicate:
        def __init__(self, cfg, env):
            self.reason = ["visible"]

        def __call__(self, env):
            return env.predicate_results["found"]

        def event_details(self, env_idx):
            return {"env": env_idx, "reason": self.reason}

    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        scene={},
        extras={},
        predicate_results={"found": torch.ones(2, dtype=torch.bool)},
        predicate_calls={"found": 0},
    )
    requirement = TrueForConsecutiveStepsCfg(predicate=TerminationTermCfg(func=ReasonPredicate), required_steps=2)
    tracker = ProgressTracker(
        [
            _criteria("found"),
            CompletionCriteria(name="reason", predicate_sequence=[requirement], required_for_success=False),
        ],
        num_envs=2,
        device="cpu",
        env=env,
    )
    for step in (1, 2):
        tracker.step(env, torch.full((2,), step, dtype=torch.long))
    # Later predicate state must not change events that were already recorded.
    tracker.get_predicate("reason").reason.append("changed")
    env.extras["progress_tracking"] = {"states": tracker.get_state(), "events": tracker.get_events()}
    for env_idx in (0, 1):
        found, reason = record_progress_results(env, env_idx)["progress"]["events"]
        assert found == {
            "step": 1,
            "criteria_name": "found",
            "sequence_name": "default_sequence",
            "predicate_index": 0,
            "predicate_name": found["predicate_name"],
            "score_delta": 1.0,
        }, "Events without details keep the plain record format."
        assert reason["step"] == 2
        assert reason["details"] == {"env": env_idx, "reason": ["visible"]}
        assert json.loads(json.dumps(reason)) == reason
    return True


def test_tracked_criteria_never_end_the_episode():
    assert run_function_with_persistent_simulation_app(_test_tracked_criteria_never_end_the_episode)


def test_tracked_criteria_complete_while_success_waits():
    assert run_function_with_persistent_simulation_app(_test_tracked_criteria_complete_while_success_waits)


def test_tracked_criteria_ignore_subtask_order():
    assert run_function_with_persistent_simulation_app(_test_tracked_criteria_ignore_subtask_order)


def test_success_requires_a_required_criteria_set():
    assert run_function_with_persistent_simulation_app(_test_success_requires_a_required_criteria_set)


def test_event_details_are_recorded_at_the_transition():
    assert run_function_with_persistent_simulation_app(_test_event_details_are_recorded_at_the_transition)

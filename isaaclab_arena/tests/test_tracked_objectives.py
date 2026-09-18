# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Recorded events must not change task success, ordering, or timeout behavior."""

from functools import partial

from isaaclab_arena.tests.test_task_success_from_progress import _controlled_predicate, _make_environment_and_manager
from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _objective(name):
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective

    return ProgressObjective(
        name=name,
        predicate_sequence=[partial(_controlled_predicate, predicate_name=name)],
    )


def _test_tracked_events_do_not_gate_success_and_reset_independently(simulation_app):
    from dataclasses import replace

    import pytest

    from isaaclab_arena.recording.progress_terms import record_progress_results

    env, manager, recorder = _make_environment_and_manager(
        ["arrived", "found", "fallen"],
        success_objectives=[replace(_objective("arrived"), score=0.5)],
        tracked_objectives=[_objective("found"), _objective("fallen")],
    )
    env.predicate_results["arrived"][:] = False
    env.predicate_results["fallen"][:] = False
    env.episode_length_buf[:] = 1
    assert manager.compute().tolist() == [False, False]
    recorder.record_post_step()
    record = record_progress_results(env, env_id=0)["progress"]
    assert not record["all_complete"]
    assert record["overall_score"] == pytest.approx(1.0 / 2.5)
    assert [event["objective"] for event in record["events"]] == ["found"]

    env.predicate_results["arrived"][:] = True
    env.episode_length_buf[:] = 2
    assert manager.compute().tolist() == [True, True]
    recorder.record_post_step()
    record = record_progress_results(env, env_id=0)["progress"]
    assert record["all_complete"]
    assert record["overall_score"] == pytest.approx(1.5 / 2.5)
    assert set(record["objectives"]) == {"arrived", "found", "fallen"}
    assert [event["objective"] for event in record["events"]] == ["found", "arrived"]
    assert env.predicate_calls["found"] == 1
    assert env.predicate_calls["fallen"] == 2

    manager.reset(env_ids=[0])
    states = env.progress_tracker.get_state()
    assert not states[0].progress_objectives["found"].is_complete
    assert states[1].progress_objectives["found"].is_complete
    assert env.progress_tracker.get_events()[0] == []
    assert len(env.progress_tracker.get_events()[1]) == 2
    env.predicate_results["arrived"][0] = False
    env.predicate_results["found"][0] = False
    env.episode_length_buf[0] = 1
    assert manager.compute().tolist() == [False, True]
    return True


def _test_tracked_only_builder_records_until_timeout(simulation_app):
    import torch

    from isaaclab.managers import TerminationManager

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.recording.progress_terms import record_progress_results
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.no_task import NoTask
    from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

    class TrackedOnlyTask(NoTask):
        def get_termination_cfg(self):
            return TaskTerminationCfg(timeout_s=1.0, tracked=[_objective("found")])

    definition = IsaacLabArenaEnvironment(name="tracked_only", scene=Scene(), task=TrackedOnlyTask())
    cfg, _ = ArenaEnvBuilder(
        definition, ArenaEnvBuilderCfg(num_envs=2, device="cpu", solve_relations=False)
    ).compose_manager_cfg()
    assert cfg.episode_length_s == 1.0
    assert set(cfg.terminations.to_dict()) == {"success", "time_out"}
    env, _, _ = _make_environment_and_manager(["found"])
    env._progress_tracker = None
    env.max_episode_length = 2
    manager = TerminationManager(cfg.terminations, env)
    env.termination_manager = manager
    recorder = cfg.recorders.progress_tracking.class_type(cfg.recorders.progress_tracking, env)
    env.episode_length_buf[:] = 1
    assert not manager.compute().any()
    recorder.record_post_step()
    record = record_progress_results(env, env_id=0)["progress"]
    assert record["overall_score"] == 1.0
    assert not record["all_complete"]
    assert [event["objective"] for event in record["events"]] == ["found"]
    env.episode_length_buf[:] = torch.tensor([2, 1])
    assert manager.compute().tolist() == [True, False]
    assert manager.time_outs.tolist() == [True, False]
    assert not manager.terminated.any()
    assert not manager.get_term("success").any()
    return True


def _test_composite_tracked_events_ignore_order_and_final_conditions(simulation_app):
    from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase
    from isaaclab_arena.tasks.no_task import NoTask
    from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

    class ObservedTask(NoTask):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def get_termination_cfg(self):
            return TaskTerminationCfg(timeout_s=2.0, success=[_objective(self.name)], tracked=[_objective("event")])

    cfg = CompositeTaskBase(
        [ObservedTask("first"), ObservedTask("second")],
        subtasks_are_sequential=True,
        desired_subtask_success_state=[False, True],
    ).get_termination_cfg()
    assert [objective.name for objective in cfg.tracked] == ["subtask_0/event", "subtask_1/event"]
    assert [objective.parent_subtask_idx for objective in cfg.tracked] == [0, 1]
    env, manager, _ = _make_environment_and_manager(
        ["first", "second", "event"],
        success_objectives=cfg.success,
        tracked_objectives=cfg.tracked,
        subtasks_are_sequential=cfg.subtasks_are_sequential,
        desired_subtask_success_state=cfg.desired_subtask_success_state,
    )
    env.predicate_results["first"][:] = False
    env.predicate_results["event"][1] = False
    manager.compute()
    assert env.predicate_calls["second"] == 0
    assert env.progress_tracker.get_subtask_completion().tolist() == [[False, False], [False, False]]
    assert {event.progress_objective for event in env.progress_tracker.get_events()[0]} == {
        "subtask_0/event",
        "subtask_1/event",
    }
    assert env.progress_tracker.get_events()[1] == []

    env.predicate_results["first"][:] = True
    manager.compute()
    assert env.predicate_calls["second"] == 0
    env.predicate_results["first"][:] = False
    manager.compute()
    assert manager.get_term("success").tolist() == [True, True]
    assert env.progress_tracker.get_subtask_completion().tolist() == [[True, True], [True, True]]
    # The recorded event is true in one environment and false in the other.
    # Neither value changes the requested false final condition for the first task.
    return True


def _test_objective_names_are_unique_across_both_lists(simulation_app):
    import pytest

    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker

    with pytest.raises(AssertionError, match="names must be unique"):
        ProgressTracker([_objective("same")], 2, "cpu", tracked_objectives=[_objective("same")])
    return True


def test_tracked_events_do_not_gate_success_and_reset_independently():
    assert run_function_with_persistent_simulation_app(_test_tracked_events_do_not_gate_success_and_reset_independently)


def test_tracked_only_builder_records_until_timeout():
    assert run_function_with_persistent_simulation_app(_test_tracked_only_builder_records_until_timeout)


def test_composite_tracked_events_ignore_order_and_final_conditions():
    assert run_function_with_persistent_simulation_app(_test_composite_tracked_events_ignore_order_and_final_conditions)


def test_objective_names_are_unique_across_both_lists():
    assert run_function_with_persistent_simulation_app(_test_objective_names_are_unique_across_both_lists)


def _test_shared_stateless_predicate_runs_once_for_both_roles(simulation_app):
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective

    shared = partial(_controlled_predicate, predicate_name="shared")
    env, manager, _ = _make_environment_and_manager(
        ["shared"],
        success_objectives=[ProgressObjective(name="success", predicate_sequence=[shared])],
        tracked_objectives=[ProgressObjective(name="tracked", predicate_sequence=[shared])],
    )
    env.predicate_results["shared"][1] = False
    manager.compute()
    assert env.predicate_calls["shared"] == 1
    assert [len(events) for events in env.progress_tracker.get_events()] == [2, 0]
    manager.reset(env_ids=[0])
    env.predicate_results["shared"][:] = True
    manager.compute()
    assert env.predicate_calls["shared"] == 2
    assert manager.get_term("success").tolist() == [True, True]
    assert [len(events) for events in env.progress_tracker.get_events()] == [2, 2]
    return True


def _test_shared_managed_instance_is_rejected_across_roles(simulation_app):
    import pytest
    from isaaclab.managers import TerminationTermCfg

    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.predicates.composite import CompositePredicate
    from isaaclab_arena.tasks.predicates.consecutive import ConsecutivePredicate

    class ConsecutiveEvent(ConsecutivePredicate):
        def __call__(self, env, consecutive_steps=2, active_mask=None):
            return self._update_consecutive_and_get_completion_mask(
                env.predicate_results["shared"], active_mask=active_mask
            )

    env, _, _ = _make_environment_and_manager(["gate", "shared"])
    cfg = TerminationTermCfg(func=ConsecutiveEvent, params={"consecutive_steps": 2})
    shared = partial(ConsecutiveEvent(cfg, env), consecutive_steps=2)

    def composite(child):
        # Assign after config construction to deliberately share the runtime child.
        child_cfg = TerminationTermCfg(func=child)
        child_cfg.func = child
        composite_cfg = TerminationTermCfg(func=CompositePredicate)
        composite_cfg.params["predicates"] = [child_cfg]
        instance = CompositePredicate(composite_cfg, env)
        assert instance.predicates[0].func is child
        return partial(instance, **composite_cfg.params)

    for success_predicate, tracked_predicate in (
        (shared, shared),
        (composite(shared), shared),
        (composite(composite(shared)), shared),
        (composite(shared), composite(shared)),
    ):
        success = ProgressObjective(
            name="success",
            predicate_sequence=[partial(_controlled_predicate, predicate_name="gate"), success_predicate],
        )
        tracked = ProgressObjective(name="tracked", predicate_sequence=[tracked_predicate])
        with pytest.raises(AssertionError, match="must not share managed predicate instances"):
            ProgressTracker([success], 2, "cpu", env=env, tracked_objectives=[tracked])

    # Reusing a configuration gives each role its own counter.
    success = ProgressObjective(
        name="success", predicate_sequence=[partial(_controlled_predicate, predicate_name="gate"), cfg]
    )
    tracked = ProgressObjective(name="tracked", predicate_sequence=[cfg])
    tracker = ProgressTracker([success], 2, "cpu", env=env, tracked_objectives=[tracked])
    env.predicate_results["gate"][:] = False
    tracker.step(env)
    tracker.step(env)
    assert not tracker.is_complete().any()
    assert tracker.get_state()[0].progress_objectives["tracked"].is_complete
    env.predicate_results["gate"][:] = True
    tracker.step(env)
    tracker.step(env)
    assert not tracker.is_complete().any()
    tracker.step(env)
    assert tracker.is_complete().all()
    tracker.reset([0])
    tracker.step(env)
    assert tracker.is_complete().tolist() == [False, True]
    assert not tracker.get_state()[0].progress_objectives["tracked"].is_complete
    assert tracker.get_state()[1].progress_objectives["tracked"].is_complete
    return True


def test_shared_stateless_predicate_runs_once_for_both_roles():
    assert run_function_with_persistent_simulation_app(_test_shared_stateless_predicate_runs_once_for_both_roles)


def test_shared_managed_instance_is_rejected_across_roles():
    assert run_function_with_persistent_simulation_app(_test_shared_managed_instance_is_rejected_across_roles)

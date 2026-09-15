# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_sequential_progress_requires_order(simulation_app):
    from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase
    from isaaclab_arena.tests.test_composite_task_base import _ControlledPredicate, _ControlledTask, _make_tracker

    task = SequentialTaskBase([_ControlledTask(_ControlledPredicate(index)) for index in range(3)])
    env, tracker = _make_tracker(task, [[False, True, True], [True, True, True]])
    tracker.step(env)
    assert tracker.get_child_completion("task").tolist() == [[False, False, False], [True, False, False]]
    env.conditions[0, 0] = True
    tracker.step(env)
    assert tracker.get_child_completion("task").tolist() == [[True, False, False], [True, True, False]]
    env.conditions[:, 0] = False
    tracker.step(env)
    assert tracker.get_child_completion("task").tolist() == [[True, True, False], [True, True, True]]
    assert tracker.is_complete().tolist() == [False, True]
    tracker.step(env)
    assert tracker.is_complete().tolist() == [True, True]
    assert [len(events) for events in tracker.get_events()] == [3, 3]
    return True


def _test_sequential_final_states_use_current_conditions(simulation_app):
    from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase
    from isaaclab_arena.tests.test_composite_task_base import _ControlledPredicate, _ControlledTask, _make_tracker

    task = SequentialTaskBase(
        [_ControlledTask(_ControlledPredicate(index)) for index in range(2)],
        desired_subtask_success_state=[False, True],
    )
    env, tracker = _make_tracker(task, [[True, True]])
    tracker.step(env)
    tracker.step(env)
    assert tracker.get_child_completion("task").tolist() == [[True, True]]
    assert tracker.is_complete().tolist() == [False]
    env.conditions[0, 0] = False
    tracker.step(env)
    assert tracker.is_complete().tolist() == [True]
    assert len(tracker.get_events()[0]) == 2
    return True


def _test_sequential_reset_preserves_other_environments(simulation_app):
    import torch

    from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase
    from isaaclab_arena.tests.test_composite_task_base import _ControlledPredicate, _ControlledTask, _make_tracker

    task = SequentialTaskBase([_ControlledTask(_ControlledPredicate(index)) for index in range(2)])
    env, tracker = _make_tracker(task, [[True, True], [True, True]])
    tracker.step(env)
    tracker.step(env)
    previous_completion = tracker.is_complete()
    tracker.reset(torch.tensor([0]))
    assert previous_completion.tolist() == [True, True]
    assert tracker.get_child_completion("task").tolist() == [[False, False], [True, True]]
    assert [len(events) for events in tracker.get_events()] == [0, 2]
    tracker.step(env)
    assert tracker.get_child_completion("task").tolist() == [[True, False], [True, True]]
    tracker.reset(slice(None))
    assert tracker.get_child_completion("task").tolist() == [[False, False], [False, False]]
    tracker.step(env)
    tracker.reset()
    assert tracker.get_child_completion("task").tolist() == [[False, False], [False, False]]
    return True


def test_sequential_progress_requires_order():
    assert run_function_with_persistent_simulation_app(_test_sequential_progress_requires_order)


def test_sequential_final_states_use_current_conditions():
    assert run_function_with_persistent_simulation_app(_test_sequential_final_states_use_current_conditions)


def test_sequential_reset_preserves_other_environments():
    assert run_function_with_persistent_simulation_app(_test_sequential_reset_preserves_other_environments)

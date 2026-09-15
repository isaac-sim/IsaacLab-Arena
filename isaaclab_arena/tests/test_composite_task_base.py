# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import traceback

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

HEADLESS = True


def _test_add_suffix_configclass_transform(simulation_app) -> bool:
    """Test that _add_suffix_configclass_transform correctly renames fields with suffix."""

    from functools import partial

    from isaaclab.utils.configclass import configclass

    from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase
    from isaaclab_arena.utils.configclass import transform_configclass_instance

    @configclass
    class FooCfg:
        int_field: int = 123
        str_field: str = "123"
        float_field: float = 1.23
        bool_field: bool = True

    try:
        original_cfg = FooCfg()
        edited_cfg = transform_configclass_instance(
            original_cfg,
            partial(CompositeTaskBase._add_suffix_configclass_transform, suffix="_suffix"),
        )

        # Check that new fields exist with suffix
        assert hasattr(edited_cfg, "int_field_suffix")
        assert hasattr(edited_cfg, "str_field_suffix")
        assert hasattr(edited_cfg, "float_field_suffix")
        assert hasattr(edited_cfg, "bool_field_suffix")

        # Check that values are preserved
        assert edited_cfg.int_field_suffix == 123
        assert edited_cfg.str_field_suffix == "123"
        assert edited_cfg.float_field_suffix == 1.23
        assert edited_cfg.bool_field_suffix is True

        # Check types are preserved
        assert isinstance(edited_cfg.int_field_suffix, int)
        assert isinstance(edited_cfg.str_field_suffix, str)
        assert isinstance(edited_cfg.float_field_suffix, float)
        assert isinstance(edited_cfg.bool_field_suffix, bool)

        # Check that old field names don't exist
        assert not hasattr(edited_cfg, "int_field")
        assert not hasattr(edited_cfg, "str_field")
        assert not hasattr(edited_cfg, "float_field")
        assert not hasattr(edited_cfg, "bool_field")

        # Test None input
        edited_cfg = transform_configclass_instance(
            None,
            partial(CompositeTaskBase._add_suffix_configclass_transform, suffix="_suffix"),
        )
        assert edited_cfg is None

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    return True


class _ControlledPredicate:
    """Read one independently controlled condition from the test environment."""

    def __init__(self, index):
        self.index = index
        self.calls = 0

    def __call__(self, env):
        self.calls += 1
        return env.conditions[:, self.index]


class _ControlledTask:
    def __init__(self, predicate, timeout_s=1.0):
        self.predicate = predicate
        self.timeout_s = timeout_s

    def get_termination_cfg(self):
        from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
        from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

        return TaskTerminationCfg(
            timeout_s=self.timeout_s,
            success=[ProgressObjective(name="condition", predicate_sequences=[self.predicate])],
        )

    def get_metrics(self):
        return []


def _make_tracker(task, conditions):
    import torch
    from types import SimpleNamespace

    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker

    env = SimpleNamespace(
        conditions=torch.tensor(conditions, dtype=torch.bool), extras={}, num_envs=len(conditions), device="cpu"
    )
    tracker = ProgressTracker(task.get_termination_cfg().success, len(conditions), "cpu")
    env._progress_tracker = tracker
    return env, tracker


def _test_composite_tracks_each_child_history(simulation_app):
    from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase

    task = CompositeTaskBase([_ControlledTask(_ControlledPredicate(index)) for index in range(2)])
    env, tracker = _make_tracker(task, [[True, False], [False, True]])
    tracker.step(env)
    assert tracker.is_complete().tolist() == [False, False]
    assert tracker.get_child_completion("task").tolist() == [[True, False], [False, True]]
    env.conditions.logical_not_()
    tracker.step(env)
    assert tracker.is_complete().tolist() == [True, True]
    assert not hasattr(env, "_subtask_ever_succeeded")
    assert not hasattr(env, "_current_subtask_idx")
    return True


def _test_composite_requires_history_and_desired_current_states(simulation_app):
    from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase

    predicates = [_ControlledPredicate(index) for index in range(3)]
    task = CompositeTaskBase(
        [_ControlledTask(predicate) for predicate in predicates],
        desired_subtask_success_state=[False, True, None],
    )
    env, tracker = _make_tracker(task, [[True, True, False]])
    tracker.step(env)
    assert tracker.is_complete().tolist() == [False]
    env.conditions[0, 0] = False
    tracker.step(env)
    assert tracker.is_complete().tolist() == [False], "None does not waive the child's completion history."
    env.conditions[0, 2] = True
    tracker.step(env)
    assert tracker.is_complete().tolist() == [True]
    return True


def _test_final_conditions_are_evaluated_once_per_step(simulation_app):
    from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase

    predicates = [_ControlledPredicate(index) for index in range(2)]
    task = CompositeTaskBase(
        [_ControlledTask(predicate) for predicate in predicates], desired_subtask_success_state=[True, True]
    )
    env, tracker = _make_tracker(task, [[True, True]])
    tracker.step(env)
    assert tracker.is_complete().tolist() == [True]
    assert [predicate.calls for predicate in predicates] == [1, 1]
    tracker.get_state()
    tracker.get_child_completion("task")
    assert [predicate.calls for predicate in predicates] == [1, 1]
    return True


def _test_nested_composition_preserves_order_and_records_subtasks(simulation_app):
    from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase, SubtaskSuccessStateRecorderCfg
    from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase

    children = [_ControlledTask(_ControlledPredicate(index)) for index in range(3)]
    task = SequentialTaskBase([CompositeTaskBase(children[:2]), children[2]])
    env, tracker = _make_tracker(task, [[False, True, True]])
    tracker.step(env)
    assert tracker.get_child_completion("task").tolist() == [[False, False]]
    env.conditions[0, 0] = True
    env.conditions[0, 1] = False
    tracker.step(env)
    assert tracker.get_child_completion("task").tolist() == [[True, False]]
    assert tracker.get_child_completion("subtask_0/task").tolist() == [[True, True]]
    tracker.step(env)
    assert tracker.is_complete().tolist() == [True]

    recorder_cfg = SubtaskSuccessStateRecorderCfg(objective_name="subtask_0/task")
    recorder = recorder_cfg.class_type(recorder_cfg, env)
    name, recorded_completion = recorder.record_post_step()
    assert name == "subtask_success_rate"
    assert recorded_completion.tolist() == [[True, True]]
    return True


def _test_multiple_objective_subtask_retains_completion_history(simulation_app):
    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase
    from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

    class MultipleObjectiveTask(_ControlledTask):
        def get_termination_cfg(self):
            return TaskTerminationCfg(
                timeout_s=self.timeout_s,
                success=[
                    ProgressObjective(name=f"condition_{index}", predicate_sequences=[_ControlledPredicate(index)])
                    for index in range(2)
                ],
            )

    for desired_state in (True, False):
        task = CompositeTaskBase(
            [MultipleObjectiveTask(None), _ControlledTask(_ControlledPredicate(2))],
            desired_subtask_success_state=[desired_state, None],
        )
        env, tracker = _make_tracker(task, [[True, True, False]])
        tracker.step(env)
        assert tracker.get_child_completion("task").tolist() == [[True, False]]
        env.conditions.logical_not_()
        tracker.step(env)
        assert tracker.get_child_completion("task").tolist() == [[True, True]]
        assert tracker.is_complete().tolist() == [desired_state]
    return True


def _test_composite_preserves_each_child_failure_condition(simulation_app):
    import pytest
    from isaaclab.managers import TerminationTermCfg

    from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase
    from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

    class FailingTask(_ControlledTask):
        def get_termination_cfg(self):
            termination = super().get_termination_cfg()
            termination.failures["object_dropped"] = TerminationTermCfg(func=self.predicate)
            return termination

    children = [FailingTask(_ControlledPredicate(index)) for index in range(2)]
    task = CompositeTaskBase(children)
    env, _ = _make_tracker(task, [[True, False]])
    terms = task.get_termination_cfg()
    assert terms.failures["object_dropped_subtask_0"].func(env).tolist() == [True]
    assert terms.failures["object_dropped_subtask_1"].func(env).tolist() == [False]
    assert len(terms.success) == 1
    assert terms.timeout_s == 2.0

    class InvalidTask(_ControlledTask):
        def get_termination_cfg(self):
            return TaskTerminationCfg(timeout_s=self.timeout_s)

    invalid_task = CompositeTaskBase([InvalidTask(_ControlledPredicate(0))])
    with pytest.raises(AssertionError, match="success objectives"):
        invalid_task.get_termination_cfg()
    return True


def _test_composed_task_has_one_overall_timeout(simulation_app):
    from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase
    from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase

    children = [
        _ControlledTask(_ControlledPredicate(0), timeout_s=2.0),
        _ControlledTask(_ControlledPredicate(1), timeout_s=3.0),
    ]
    for task_type in (CompositeTaskBase, SequentialTaskBase):
        default_task = task_type(children)
        default_termination = default_task.get_termination_cfg()
        assert default_termination.timeout_s == 5.0
        assert default_termination.failures == {}
        assert default_termination.success[0].sequential == (task_type is SequentialTaskBase)

        overridden_task = task_type(children, episode_length_s=8.0)
        assert overridden_task.get_termination_cfg().timeout_s == 8.0

        nested_task = task_type([default_task, _ControlledTask(_ControlledPredicate(2), timeout_s=4.0)])
        assert nested_task.get_termination_cfg().timeout_s == 9.0
    return True


def test_composed_task_has_one_overall_timeout():
    assert run_function_with_persistent_simulation_app(_test_composed_task_has_one_overall_timeout, headless=HEADLESS)


def test_multiple_objective_subtask_retains_completion_history():
    assert run_function_with_persistent_simulation_app(
        _test_multiple_objective_subtask_retains_completion_history, headless=HEADLESS
    )


def test_composite_preserves_each_child_failure_condition():
    assert run_function_with_persistent_simulation_app(
        _test_composite_preserves_each_child_failure_condition, headless=HEADLESS
    )


def test_add_suffix_configclass_transform():
    assert run_function_with_persistent_simulation_app(_test_add_suffix_configclass_transform, headless=HEADLESS)


def test_composite_tracks_each_child_history():
    assert run_function_with_persistent_simulation_app(_test_composite_tracks_each_child_history, headless=HEADLESS)


def test_composite_requires_history_and_desired_current_states():
    assert run_function_with_persistent_simulation_app(
        _test_composite_requires_history_and_desired_current_states, headless=HEADLESS
    )


def test_final_conditions_are_evaluated_once_per_step():
    assert run_function_with_persistent_simulation_app(
        _test_final_conditions_are_evaluated_once_per_step, headless=HEADLESS
    )


def test_nested_composition_preserves_order_and_records_subtasks():
    assert run_function_with_persistent_simulation_app(
        _test_nested_composition_preserves_order_and_records_subtasks, headless=HEADLESS
    )

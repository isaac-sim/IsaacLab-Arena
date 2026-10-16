# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from types import SimpleNamespace

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _condition(env, index=0):
    return env.conditions[:, index]


def _never_complete(env):
    return env.conditions[:, 0] & False


def _success_criteria():
    from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria

    return [CompletionCriteria("success", predicate_sequence=[_never_complete])]


def _env(values):
    import torch

    return SimpleNamespace(
        num_envs=len(values),
        device="cpu",
        conditions=torch.tensor(values),
        step_dt=0.02,
        episode_length_buf=torch.zeros(len(values), dtype=torch.long),
        extras={},
    )


def _step(tracker, env):
    env.episode_length_buf += 1
    tracker.step(env, env.episode_length_buf)


def _events(tracker, env_id=0):
    return [(event.name, event.step) for event in tracker.get_tracked_event_occurrences()[env_id]]


def _test_recurring_events_and_success(_app):
    from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.progress_tracking.tracked_event_cfg import TrackedEventCfg
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    env = _env([[True], [True]])
    criteria = [CompletionCriteria("success", predicate_sequence=[TrueForConsecutiveStepsCfg(_condition, 3)])]
    baseline = ProgressTracker(criteria, 2, "cpu")
    tracker = ProgressTracker(
        criteria,
        2,
        "cpu",
        tracked_events=[
            TrackedEventCfg("instant", predicate_sequence=[_condition]),
            TrackedEventCfg("held", predicate_sequence=[TrueForConsecutiveStepsCfg(_condition, 2)]),
        ],
    )
    for step in range(1, 7):
        _step(tracker, env)
        baseline.step(env, env.episode_length_buf)
        assert tracker.get_state() == baseline.get_state()
        assert tracker.get_events() == baseline.get_events()
    assert _events(tracker) == [("instant", 1), ("held", 2)]
    for value in (False, True, True):
        env.conditions[:] = value
        _step(tracker, env)
        baseline.step(env, env.episode_length_buf)
        assert tracker.get_state() == baseline.get_state()
        assert tracker.get_events() == baseline.get_events()
    assert _events(tracker) == [("instant", 1), ("held", 2), ("instant", 8), ("held", 9)]
    snapshot = tracker.get_tracked_event_occurrences()
    tracker.reset([0])
    env.episode_length_buf[0] = 0
    assert len(snapshot[0]) == 4
    assert tracker.get_tracked_event_occurrences()[0] == []
    _step(tracker, env)
    assert _events(tracker, 0) == [("instant", 1)]
    assert _events(tracker, 1)[-1] == ("held", 9)
    _step(tracker, env)
    assert _events(tracker, 0)[-1] == ("held", 2)
    assert _events(tracker, 1)[-1] == ("held", 9)
    return True


def _test_events_preserve_subtask_success(_app):
    import torch

    from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.progress_tracking.tracked_event_cfg import TrackedEventCfg
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    env = _env([[True, True, True], [True, True, True]])
    criteria = [
        CompletionCriteria(
            f"subtask_{index}",
            [TrueForConsecutiveStepsCfg(partial(_condition, index=index), 2)],
            parent_subtask_idx=index,
        )
        for index in range(2)
    ]
    baseline = ProgressTracker(
        criteria, 2, "cpu", subtasks_are_sequential=True, desired_subtask_success_state=[True, True]
    )
    tracker = ProgressTracker(
        criteria,
        2,
        "cpu",
        subtasks_are_sequential=True,
        desired_subtask_success_state=[True, True],
        tracked_events=[
            TrackedEventCfg(
                "observed", predicate_sequence=[TrueForConsecutiveStepsCfg(partial(_condition, index=2), 2)]
            )
        ],
    )
    for step in range(1, 9):
        env.conditions[:, 0] = step not in (3, 6)
        env.conditions[:, 2] = step not in (3, 7)
        env.episode_length_buf += 1
        tracker.step(env, env.episode_length_buf)
        baseline.step(env, env.episode_length_buf)
        assert tracker.get_state() == baseline.get_state()
        assert tracker.get_events() == baseline.get_events()
        assert torch.equal(tracker.get_subtask_completion(), baseline.get_subtask_completion())
        # Event completion and resets do not alter subtask history or current final-state checks.
        assert tracker.is_complete().tolist() == [step in (5, 8)] * 2
    assert _events(tracker, 0) == [("observed", 2), ("observed", 5)]
    assert _events(tracker, 1) == [("observed", 2), ("observed", 5)]
    return True


def _test_order_and_fresh_history(_app):
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.progress_tracking.tracked_event_cfg import TrackedEventCfg

    env = _env([[False, True]])
    predicates = [partial(_condition, index=0), partial(_condition, index=1)]
    tracker = ProgressTracker(
        _success_criteria(), 1, "cpu", tracked_events=[TrackedEventCfg("ordered", predicate_sequence=predicates)]
    )
    _step(tracker, env)
    assert _events(tracker) == [], "The final predicate alone must not record an event."
    env.conditions[:] = env.conditions.logical_not()
    _step(tracker, env)
    assert _events(tracker) == []
    env.conditions[:] = False
    env.conditions[:, 1] = True
    _step(tracker, env)
    assert _events(tracker) == [("ordered", 3)], "Earlier predicates need not remain true."
    env.conditions[:] = True
    _step(tracker, env)
    _step(tracker, env)
    assert _events(tracker) == [("ordered", 3)], "A persistent final condition records once."
    env.conditions[:] = False
    _step(tracker, env)
    env.conditions[:, 1] = True
    _step(tracker, env)
    assert _events(tracker) == [("ordered", 3)], "A new cycle must observe the first predicate again."
    env.conditions[:] = True
    _step(tracker, env)
    assert _events(tracker) == [("ordered", 3)], "At most one predicate advances per step."
    env.conditions[:, 0] = False
    _step(tracker, env)
    assert _events(tracker) == [("ordered", 3), ("ordered", 9)]
    assert not tracker.is_complete().any(), "Observations alone never make the task succeed."
    return True


def _test_demo_failure_sequences(_app):
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.progress_tracking.tracked_event_cfg import TrackedEventCfg
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    # Columns: open near object, closed empty, lifted and grasped, grasped, resting outside bowl.
    env = _env([[False, False, False, False, True]])

    def released_and_resting(env):
        return ~env.conditions[:, 3] & env.conditions[:, 4]

    tracker = ProgressTracker(
        _success_criteria(),
        1,
        "cpu",
        tracked_events=[
            TrackedEventCfg(
                "failed_grasp",
                predicate_sequence=[_condition, TrueForConsecutiveStepsCfg(partial(_condition, index=1), 2)],
            ),
            TrackedEventCfg(
                "dropped_object",
                predicate_sequence=[
                    TrueForConsecutiveStepsCfg(partial(_condition, index=2), 2),
                    TrueForConsecutiveStepsCfg(released_and_resting, 2),
                ],
            ),
        ],
    )
    phases = [
        ([False, False, False, False, True], 3),  # Initial rest is not a drop.
        ([True, False, False, False, True], 1),
        ([False, True, False, False, True], 3),  # One failed grasp despite sustained closure.
        ([False, False, True, True, False], 2),
        ([False, False, False, False, True], 3),  # One drop despite sustained rest.
        ([True, False, False, False, True], 1),
        ([False, True, False, False, True], 2),  # A second failed grasp.
        ([False, False, True, True, False], 3),  # Clear the drop and satisfy a fresh lift.
        ([False, False, False, False, True], 2),  # A second drop.
    ]
    for conditions, steps in phases:
        for _ in range(steps):
            env.conditions[0] = env.conditions.new_tensor(conditions)
            _step(tracker, env)
    assert _events(tracker) == [
        ("failed_grasp", 6),
        ("dropped_object", 11),
        ("failed_grasp", 15),
        ("dropped_object", 20),
    ]
    assert not tracker.is_complete().any()
    return True


def _test_temporal_event_isolation(_app):
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.progress_tracking.tracked_event_cfg import TrackedEventCfg
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    env = _env([[False, True], [True, True]])
    counter = TrueForConsecutiveStepsCfg(partial(_condition, index=1), required_steps=2)
    tracker = ProgressTracker(
        _success_criteria(),
        2,
        "cpu",
        env,
        tracked_events=[
            TrackedEventCfg("gated", predicate_sequence=[_condition, counter]),
            TrackedEventCfg("independent", predicate_sequence=[counter]),
        ],
    )
    for _ in range(3):
        _step(tracker, env)
    assert _events(tracker, 0) == [("independent", 2)]
    assert _events(tracker, 1) == [("independent", 2), ("gated", 3)]
    # Reusing a configuration must not share counters or count while waiting for the gate.
    env.conditions[0, 0] = True
    _step(tracker, env)
    _step(tracker, env)
    assert _events(tracker, 0) == [("independent", 2)]
    _step(tracker, env)
    assert _events(tracker, 0) == [("independent", 2), ("gated", 6)]
    assert _events(tracker, 1) == [("independent", 2), ("gated", 3)]
    env.conditions[1, 1] = False
    _step(tracker, env)
    env.conditions[1, 1] = True
    # Reset one episode while the other has just cleared its failure condition.
    tracker.reset([0])
    env.episode_length_buf[0] = 0
    _step(tracker, env)
    assert _events(tracker, 0) == []
    assert _events(tracker, 1)[-1] == ("gated", 3)
    _step(tracker, env)
    assert _events(tracker, 0) == [("independent", 2)]
    assert _events(tracker, 1)[-1] == ("independent", 9)
    _step(tracker, env)
    assert _events(tracker, 0)[-1] == ("gated", 3)
    assert _events(tracker, 1)[-1] == ("gated", 10)
    return True


def _test_callable_events(_app):
    from threading import Lock

    from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.progress_tracking.tracked_event_cfg import TrackedEventCfg
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    class ReadCondition:
        def __init__(self):
            self.lock = Lock()

        def read(self, env):
            with self.lock:
                return _condition(env)

        def __call__(self, env):
            return self.read(env)

    source = ReadCondition()
    for predicate in (source, source.read, partial(source.read)):
        env = _env([[True]])
        held = TrueForConsecutiveStepsCfg(predicate, required_steps=2)
        criteria = [CompletionCriteria("success", [held])]
        baseline = ProgressTracker(criteria, 1, "cpu")
        tracker = ProgressTracker(
            criteria,
            1,
            "cpu",
            tracked_events=[
                TrackedEventCfg("instant", predicate_sequence=[predicate]),
                TrackedEventCfg("held", predicate_sequence=[held]),
                TrackedEventCfg("ordered", predicate_sequence=[held, held]),
            ],
        )
        for _ in range(5):
            _step(tracker, env)
            baseline.step(env, env.episode_length_buf)
            assert tracker.get_state() == baseline.get_state()
        for name, expected_steps in (("instant", [1]), ("held", [2]), ("ordered", [4])):
            assert [step for event_name, step in _events(tracker) if event_name == name] == expected_steps
    return True


def _test_events_share_predicate_cache(_app):
    from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.progress_tracking.tracked_event_cfg import TrackedEventCfg
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    calls = 0

    def condition(env):
        nonlocal calls
        calls += 1
        return _condition(env)

    env = _env([[True], [True]])
    held = TrueForConsecutiveStepsCfg(condition, 2)
    tracker = ProgressTracker(
        [CompletionCriteria("success", [held])],
        2,
        "cpu",
        tracked_events=[
            TrackedEventCfg("instant", predicate_sequence=[condition]),
            TrackedEventCfg("held", predicate_sequence=[held]),
            TrackedEventCfg("ordered", predicate_sequence=[held, held]),
        ],
    )
    for step, value in enumerate([True] * 5 + [False, True, True], start=1):
        env.conditions[0, 0] = value
        _step(tracker, env)
        assert calls == step, "Success, advancing events, and waiting events share one predicate cache."
    first_occurrences = [("instant", 1), ("held", 2), ("ordered", 4)]
    assert _events(tracker, 0) == first_occurrences + [("instant", 7), ("held", 8)]
    assert _events(tracker, 1) == first_occurrences
    assert tracker.is_complete().all(), "Clearing an event must not clear task success."
    return True


def _test_rearm_requires_fresh_temporal_sequence(_app):
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.progress_tracking.tracked_event_cfg import TrackedEventCfg
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    env = _env([[True, True]])
    tracker = ProgressTracker(
        _success_criteria(),
        1,
        "cpu",
        tracked_events=[
            TrackedEventCfg(
                "failure",
                predicate_sequence=[
                    TrueForConsecutiveStepsCfg(_condition, 2),
                    TrueForConsecutiveStepsCfg(partial(_condition, index=1), 3),
                ],
            )
        ],
    )
    for step in range(1, 16):
        env.conditions[0, 1] = step != 10
        _step(tracker, env)
        expected = [("failure", 5)] if step >= 5 else []
        if step == 15:
            expected.append(("failure", 15))
        assert _events(tracker) == expected
    return True


def _test_finished_episode_records_preserve_events(_app, output_dir):
    import json
    import torch

    from isaaclab_arena.embodiments.no_embodiment import NoEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
    from isaaclab_arena.progress_tracking.tracked_event_cfg import TrackedEventCfg
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.no_task import NoTask
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg
    from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

    def always(env):
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def initially_true(env):
        return env.episode_length_buf <= 2

    def finished(env):
        return env.episode_length_buf >= torch.tensor([2, 5], device=env.device)

    class ObservingTask(NoTask):
        def get_termination_cfg(self):
            return TaskTerminationCfg(timeout_s=None, success=[CompletionCriteria("finished", [finished])])

        def get_tracked_events(self):
            held = TrueForConsecutiveStepsCfg(always, required_steps=2)
            return [
                TrackedEventCfg("instant", predicate_sequence=[always]),
                TrackedEventCfg("held", predicate_sequence=[held]),
                TrackedEventCfg("ordered", predicate_sequence=[held, held]),
                TrackedEventCfg(
                    "cleared",
                    predicate_sequence=[
                        TrueForConsecutiveStepsCfg(initially_true, required_steps=2),
                        lambda env: ~initially_true(env),
                    ],
                ),
            ]

    description = IsaacLabArenaEnvironment(
        name="tracked_event_episode_limit", scene=Scene(), task=ObservingTask(), embodiment=NoEmbodiment()
    )
    builder = ArenaEnvBuilder(description, ArenaEnvBuilderCfg(num_envs=2, solve_relations=False))
    env = builder.make_registered()
    base = env.unwrapped
    results_path = output_dir / "episode_results.jsonl"
    try:
        base.configure_episode_limit(2)
        base.episode_recorder_manager.set_output_path(results_path)
        env.reset()
        actions = torch.zeros(env.action_space.shape, device=base.device)
        first_record = None
        for step in range(1, 8):
            env.step(actions)
            records = [json.loads(line) for line in results_path.read_text().splitlines()]
            if step == 2:
                assert base.active_episode_mask.tolist() == [False, True]
                assert len(records) == 1
                first_record = records[0]
            if step >= 2:
                assert records[0] == first_record, "Later steps must not alter a finished episode's saved result."
        assert not base.active_episode_mask.any()
        assert len(records) == 2
        expected_events = {
            0: [("instant", 1), ("held", 2)],
            1: [("instant", 1), ("held", 2), ("cleared", 3), ("ordered", 4)],
        }
        for record in records:
            events = record["progress"]["tracked_events"]
            assert [(event["name"], event["step"]) for event in events] == expected_events[record["env_id"]]
    finally:
        env.close()
    return True


def _test_recording_and_validation(_app):
    import json

    import pytest

    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker, ProgressTrackingRecorderCfg
    from isaaclab_arena.progress_tracking.tracked_event_cfg import TrackedEventCfg
    from isaaclab_arena.recording.progress_terms import record_progress_results

    env = _env([[True]])
    event_cfg = TrackedEventCfg("observed", predicate_sequence=[_condition])
    env.progress_tracker = ProgressTracker(_success_criteria(), 1, "cpu", tracked_events=[event_cfg])
    tracker = env.progress_tracker
    with pytest.raises(AssertionError, match="step_index"):
        tracker.step(env)
    with pytest.raises(AssertionError, match="unique"):
        ProgressTracker(_success_criteria(), 1, "cpu", tracked_events=[event_cfg, event_cfg])
    with pytest.raises(AssertionError, match="Task success requires"):
        ProgressTracker([], 1, "cpu", tracked_events=[event_cfg])
    with pytest.raises(TypeError, match="predicate_sequence"):
        TrackedEventCfg("invalid")
    for invalid in (
        [],
        {"a": [_condition]},
        {"empty": []},
        {},
        None,
        _condition,
        (_condition,),
        {"invalid": _condition},
        [True],
        [(_condition, 1.0)],
    ):
        with pytest.raises(AssertionError):
            TrackedEventCfg("invalid", predicate_sequence=invalid)
    for name in ("", "  ", None):
        with pytest.raises(AssertionError, match="nonempty name"):
            TrackedEventCfg(name, predicate_sequence=[_condition])
    _step(tracker, env)
    with pytest.raises(AssertionError, match="exactly one"):
        tracker.step(env, env.episode_length_buf)
    assert _events(tracker) == [("observed", 1)]
    cfg = ProgressTrackingRecorderCfg()
    cfg.class_type(cfg, env).record_post_step()
    tracker.reset([0])
    result = json.loads(json.dumps(record_progress_results(env, 0)))
    assert result["tracked_event_names"] == ["observed"]
    assert result["tracked_events"] == [{"name": "observed", "step": 1}]
    assert result["control_step_dt"] == 0.02
    assert result["overall_score"] == 0.0
    assert not result["all_complete"]
    assert list(result["criteria_by_name"]) == ["success"]
    assert result["events"] == []
    return True


def _test_builder_and_composition(_app):
    import pytest

    from isaaclab_arena.embodiments.no_embodiment import NoEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.progress_tracking.tracked_event_cfg import TrackedEventCfg
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase
    from isaaclab_arena.tasks.no_task import NoTask
    from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

    class ObservingTask(NoTask):
        def get_tracked_events(self):
            return [TrackedEventCfg("observed", predicate_sequence=[_condition])]

    description = IsaacLabArenaEnvironment(
        name="observing_task",
        scene=Scene(),
        task=NoTask(),
        embodiment=NoEmbodiment(),
    )
    builder = ArenaEnvBuilder(description, ArenaEnvBuilderCfg(solve_relations=False, device="cpu"))
    cfg, _ = builder.build_env_cfg()
    assert cfg.terminations.to_dict() == {}
    assert cfg.task_progress is None
    description.task = ObservingTask()
    with pytest.raises(AssertionError, match="Tracked events require task success criteria"):
        builder.build_env_cfg()

    class ObservingTaskWithSuccess(ObservingTask):
        def get_termination_cfg(self):
            return TaskTerminationCfg(timeout_s=None, success=_success_criteria())

    task = ObservingTaskWithSuccess()
    description.task = task
    cfg, _ = builder.build_env_cfg()
    assert cfg.task_progress.success_criteria[0].name == "success"
    assert cfg.task_progress.tracked_events[0].name == "observed"
    composite = CompositeTaskBase([task, task], episode_length_s=10, subtasks_are_sequential=True)
    assert [event_cfg.name for event_cfg in composite.get_tracked_events()] == [
        "subtask_0/observed",
        "subtask_1/observed",
    ]
    assert task.get_tracked_events()[0].name == "observed"
    return True


def test_recurring_events_and_success():
    assert run_function_with_persistent_simulation_app(_test_recurring_events_and_success)


def test_events_preserve_subtask_success():
    assert run_function_with_persistent_simulation_app(_test_events_preserve_subtask_success)


def test_order_and_fresh_history():
    assert run_function_with_persistent_simulation_app(_test_order_and_fresh_history)


def test_temporal_event_isolation():
    assert run_function_with_persistent_simulation_app(_test_temporal_event_isolation)


def test_callable_events():
    assert run_function_with_persistent_simulation_app(_test_callable_events)


def test_events_share_predicate_cache():
    assert run_function_with_persistent_simulation_app(_test_events_share_predicate_cache)


def test_rearm_requires_fresh_temporal_sequence():
    assert run_function_with_persistent_simulation_app(_test_rearm_requires_fresh_temporal_sequence)


def test_finished_episode_records_preserve_events(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_finished_episode_records_preserve_events, output_dir=tmp_path
    )


def test_recording_and_validation():
    assert run_function_with_persistent_simulation_app(_test_recording_and_validation)


def test_builder_and_composition():
    assert run_function_with_persistent_simulation_app(_test_builder_and_composition)


def test_demo_failure_sequences():
    assert run_function_with_persistent_simulation_app(_test_demo_failure_sequences)

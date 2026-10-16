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
            TrackedEventCfg("instant", predicates=[_condition]),
            TrackedEventCfg("held", predicates=[TrueForConsecutiveStepsCfg(_condition, 2)]),
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


def _test_event_activity_does_not_change_success(_app):
    import torch

    import pytest

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
            TrackedEventCfg("observed", predicates=[TrueForConsecutiveStepsCfg(partial(_condition, index=2), 2)])
        ],
    )
    for step in range(1, 9):
        env.conditions[:, 0] = step not in (3, 6)
        env.conditions[:, 2] = step not in (3, 7)
        env.episode_length_buf += 1
        tracking_active = torch.tensor([step < 2, step < 6])
        tracker.step(env, env.episode_length_buf, event_active_envs=tracking_active)
        baseline.step(env, env.episode_length_buf)
        assert tracker.get_state() == baseline.get_state()
        assert tracker.get_events() == baseline.get_events()
        assert torch.equal(tracker.get_subtask_completion(), baseline.get_subtask_completion())
        # Success continues to advance and recheck final conditions even when observations pause.
        assert tracker.is_complete().tolist() == [step in (5, 8)] * 2
    assert _events(tracker, 0) == []
    assert _events(tracker, 1) == [("observed", 2), ("observed", 5)]
    # Pausing observations must not bypass the existing control-step validation.
    with pytest.raises(AssertionError, match="exactly one"):
        tracker.step(env, env.episode_length_buf, event_active_envs=torch.zeros(2, dtype=torch.bool))
    return True


def _test_order_and_fresh_history(_app):
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.progress_tracking.tracked_event_cfg import TrackedEventCfg

    env = _env([[False, True]])
    predicates = [partial(_condition, index=0), partial(_condition, index=1)]
    tracker = ProgressTracker(
        _success_criteria(),
        1,
        "cpu",
        tracked_events=[
            TrackedEventCfg("ordered", predicates=predicates),
            TrackedEventCfg(
                "unordered",
                predicates={f"member_{index}": [predicate] for index, predicate in enumerate(predicates)},
            ),
        ],
    )
    _step(tracker, env)
    env.conditions[:] = env.conditions.logical_not()
    _step(tracker, env)
    assert _events(tracker) == [("unordered", 2)]
    env.conditions[:] = False
    env.conditions[:, 1] = True
    _step(tracker, env)
    assert _events(tracker)[-1] == ("ordered", 3)
    _step(tracker, env)
    assert len(_events(tracker)) == 2, "A new cycle must observe the first member again."
    env.conditions[:] = True
    _step(tracker, env)
    assert _events(tracker)[-1] == ("unordered", 5)
    _step(tracker, env)
    assert ("ordered", 6) not in _events(tracker), "The final condition has not cleared."
    env.conditions[:] = False
    _step(tracker, env)
    env.conditions[:, 0] = True
    _step(tracker, env)
    env.conditions[:, 1] = True
    _step(tracker, env)
    assert _events(tracker)[-2:] == [("ordered", 9), ("unordered", 9)]
    assert not tracker.is_complete().any(), "Observations alone never make the task succeed."
    return True


def _test_desired_states_and_none_gating(_app):
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.progress_tracking.tracked_event_cfg import TrackedEventCfg

    env = _env([[False, False]])
    predicates = [partial(_condition, index=0), partial(_condition, index=1)]
    tracker = ProgressTracker(
        _success_criteria(),
        1,
        "cpu",
        tracked_events=[
            TrackedEventCfg("cleared", predicates=[predicates[0]], desired_predicate_state=[False]),
            TrackedEventCfg("gated", predicates=predicates, desired_predicate_state=[None, True]),
            TrackedEventCfg(
                "ignored",
                predicates={f"member_{index}": [predicate] for index, predicate in enumerate(predicates)},
                desired_predicate_state=[None, True],
            ),
        ],
    )
    _step(tracker, env)
    assert not _events(tracker), "False requires prior satisfaction."
    env.conditions[:, 1] = True
    _step(tracker, env)
    assert _events(tracker) == [("ignored", 2)]
    env.conditions[:] = True
    _step(tracker, env)
    assert ("gated", 3) not in _events(tracker), "The next member starts on the next step."
    env.conditions[:, 0] = False
    _step(tracker, env)
    assert ("cleared", 4) in _events(tracker)
    assert ("gated", 4) in _events(tracker)
    _step(tracker, env)
    assert ("cleared", 5) not in _events(tracker)
    assert ("gated", 5) not in _events(tracker)
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
            TrackedEventCfg("gated", predicates=[_condition, counter]),
            TrackedEventCfg("independent", predicates=[counter]),
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
                TrackedEventCfg("instant", predicates=[predicate]),
                TrackedEventCfg("held", predicates=[held]),
                TrackedEventCfg("ordered", predicates=[held, held]),
            ],
        )
        for _ in range(5):
            _step(tracker, env)
            baseline.step(env, env.episode_length_buf)
            assert tracker.get_state() == baseline.get_state()
        for name, expected_steps in (("instant", [1]), ("held", [2]), ("ordered", [4])):
            assert [step for event_name, step in _events(tracker) if event_name == name] == expected_steps
    return True


def _test_single_tracker_and_shared_cache(_app):
    from unittest.mock import patch

    from isaaclab_arena.progress_tracking import progress_tracker
    from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
    from isaaclab_arena.progress_tracking.tracked_event_cfg import TrackedEventCfg
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    calls = 0

    def condition(env):
        nonlocal calls
        calls += 1
        return _condition(env)

    env = _env([[True], [True]])
    held = TrueForConsecutiveStepsCfg(condition, 2)
    with (
        patch.object(progress_tracker, "ProgressTracker", wraps=progress_tracker.ProgressTracker) as constructor,
        patch.object(
            progress_tracker, "CompletionCriteriaRunner", wraps=progress_tracker.CompletionCriteriaRunner
        ) as criteria_constructor,
    ):
        tracker = progress_tracker.ProgressTracker(
            [CompletionCriteria("success", [held])],
            2,
            "cpu",
            tracked_events=[
                TrackedEventCfg("instant", predicates=[condition]),
                TrackedEventCfg("held", predicates=[held]),
                TrackedEventCfg("ordered", predicates=[held, held]),
                TrackedEventCfg("unordered", predicates={"member_0": [held], "member_1": [held]}),
            ],
        )
        assert constructor.call_count == 1, "Events must not construct nested ProgressTrackers."
        assert criteria_constructor.call_count == 5, "Use one success runner and one runner per event, not per member."
    for step, value in enumerate([True] * 5 + [False, True, True], start=1):
        env.conditions[0, 0] = value
        _step(tracker, env)
        assert calls == step, "Success, advancing events, and waiting events share one predicate cache."
    first_occurrences = [("instant", 1), ("held", 2), ("unordered", 2), ("ordered", 4)]
    assert _events(tracker, 0) == first_occurrences + [("instant", 7), ("held", 8), ("unordered", 8)]
    assert _events(tracker, 1) == first_occurrences
    assert tracker.is_complete().all(), "Clearing an event must not clear task success."
    return True


def _test_desired_states_recheck_completed_temporal_members(_app):
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.progress_tracking.tracked_event_cfg import TrackedEventCfg
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    env = _env([[False, False]])
    requirements = [
        TrueForConsecutiveStepsCfg(_condition, 2),
        TrueForConsecutiveStepsCfg(partial(_condition, index=1), 2),
    ]
    tracker = ProgressTracker(
        _success_criteria(),
        1,
        "cpu",
        tracked_events=[
            TrackedEventCfg("ordered", predicates=requirements, desired_predicate_state=[True, False]),
            TrackedEventCfg(
                "unordered",
                predicates={f"member_{index}": [predicate] for index, predicate in enumerate(requirements)},
                desired_predicate_state=[True, False],
            ),
        ],
    )
    timeline = [
        [True, False],
        [True, False],  # First member completes.
        [False, True],
        [False, True],  # Both histories complete; current desired states still fail.
        [True, False],  # First member must rebuild its entire two-step streak.
        [True, False],  # First occurrence.
        [False, False],  # The desired combination clears; reset both histories.
        [True, True],
        [True, True],
        [True, True],
        [True, True],
        [True, False],  # Second occurrence requires observing the second member true again.
    ]
    for step, values in enumerate(timeline, start=1):
        env.conditions[0] = env.conditions.new_tensor(values)
        _step(tracker, env)
        expected = [("ordered", 6), ("unordered", 6)] if step >= 6 else []
        if step == 12:
            expected += [("ordered", 12), ("unordered", 12)]
        assert _events(tracker) == expected
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
                predicates=[
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


def _test_temporal_desired_states_while_waiting(_app):
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.progress_tracking.tracked_event_cfg import TrackedEventCfg
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    env = _env([[False, True]])
    tracker = ProgressTracker(
        _success_criteria(),
        1,
        "cpu",
        tracked_events=[
            TrackedEventCfg(
                "contact_lost_in_region",
                predicates=[
                    TrueForConsecutiveStepsCfg(_condition, 2),
                    partial(_condition, index=1),
                ],
                desired_predicate_state=[False, True],
            )
        ],
    )
    # The object stays in the region. Brief contact recovery must not clear the
    # event: the full two-step contact requirement must become true again.
    contact = [True, True, True, False, True, False, True, True, True, False, True, True, False]
    for step, value in enumerate(contact, start=1):
        env.conditions[0, 0] = value
        _step(tracker, env)
        expected = [("contact_lost_in_region", 4)] if step >= 4 else []
        if step == 13:
            expected.append(("contact_lost_in_region", 13))
        assert _events(tracker) == expected
        if 4 <= step <= 7:
            assert tracker._tracked_event_runners[0]._waiting_for_clear.item()
        if step == 8:
            assert not tracker._tracked_event_runners[0]._waiting_for_clear.item()
    # The recovery on steps 7-8 is discarded at reset: step 9 alone cannot
    # establish the next occurrence's two-step contact history.
    return True


def _test_named_sequences_and_desired_state_order(_app):
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.progress_tracking.tracked_event_cfg import TrackedEventCfg

    env = _env([[False, False, True]])
    first, second, independent = [_condition, partial(_condition, index=1), partial(_condition, index=2)]
    tracker = ProgressTracker(
        _success_criteria(),
        1,
        "cpu",
        tracked_events=[
            TrackedEventCfg(
                "observed",
                predicates={"z_sequence": [first, second], "a_sequence": [independent]},
                desired_predicate_state=[False, True, True],
            )
        ],
    )
    timeline = [
        [False, False, True],  # Independent sequence can finish before the other starts.
        [True, True, True],  # Only the first member of z_sequence can complete now.
        [True, True, True],  # Both sequences complete, but the desired states do not match.
        [False, True, True],  # Desired states follow insertion order, not sorted sequence names.
        [False, True, True],
        [False, True, False],  # Clear and reset both sequences.
        [True, True, True],
        [False, True, True],  # Fresh completion of both sequences.
    ]
    for step, values in enumerate(timeline, start=1):
        env.conditions[0] = env.conditions.new_tensor(values)
        _step(tracker, env)
        expected = [("observed", 4)] if step >= 4 else []
        if step == 8:
            expected.append(("observed", 8))
        assert _events(tracker) == expected
    return True


def _test_any_order_rearm_and_desired_states(_app):
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.progress_tracking.tracked_event_cfg import TrackedEventCfg

    env = _env([[True, False]])
    first, second = _condition, partial(_condition, index=1)
    tracker = ProgressTracker(
        _success_criteria(),
        1,
        "cpu",
        tracked_events=[
            TrackedEventCfg("forward", predicates={"member_0": [first], "member_1": [second]}),
            TrackedEventCfg("reverse", predicates={"member_0": [second], "member_1": [first]}),
            TrackedEventCfg("cleared", predicates=[first], desired_predicate_state=[False]),
            TrackedEventCfg("ignored_tail", predicates=[first, second], desired_predicate_state=[True, None]),
        ],
    )
    for values in (
        [True, False],
        [False, True],
        [True, True],
        [False, True],
        [False, False],
        [True, False],
        [False, True],
    ):
        env.conditions[0] = env.conditions.new_tensor(values)
        _step(tracker, env)
    for name in ("forward", "reverse"):
        assert [step for event_name, step in _events(tracker) if event_name == name] == [2, 7]
    assert [step for name, step in _events(tracker) if name == "cleared"] == [2, 7]
    assert [step for name, step in _events(tracker) if name == "ignored_tail"] == [1, 3, 6]
    # A single true clearing step is not reused as the next cycle's history.
    env.conditions[0, 0] = True
    _step(tracker, env)
    env.conditions[0, 0] = False
    _step(tracker, env)
    assert [step for name, step in _events(tracker) if name == "cleared"] == [2, 7]
    env.conditions[0, 0] = True
    _step(tracker, env)
    env.conditions[0, 0] = False
    _step(tracker, env)
    assert [step for name, step in _events(tracker) if name == "cleared"] == [2, 7, 11]
    return True


def _test_finished_episode_events(_app, output_dir):
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
                TrackedEventCfg("instant", predicates=[always]),
                TrackedEventCfg("held", predicates=[held]),
                TrackedEventCfg("ordered", predicates=[held, held]),
                TrackedEventCfg(
                    "cleared",
                    predicates=[TrueForConsecutiveStepsCfg(initially_true, required_steps=2)],
                    desired_predicate_state=[False],
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
        finished_events = None
        for step in range(1, 8):
            env.step(actions)
            if step == 2:
                assert base.active_episode_mask.tolist() == [False, True]
                finished_events = base.progress_tracker.get_tracked_event_occurrences()[0]
            if step >= 2:
                assert base.progress_tracker.get_tracked_event_occurrences()[0] == finished_events
                assert base.extras["progress_tracking"]["tracked_events"][0] == finished_events
        assert not base.active_episode_mask.any()
        assert _events(base.progress_tracker, 0) == [("instant", 1), ("held", 2)]
        assert _events(base.progress_tracker, 1) == [
            ("instant", 1),
            ("held", 2),
            ("cleared", 3),
            ("ordered", 4),
        ]
        records = [json.loads(line) for line in results_path.read_text().splitlines()]
        assert len(records) == 2
        for record in records:
            events = record["progress"]["tracked_events"]
            assert [(event["name"], event["step"]) for event in events] == _events(
                base.progress_tracker, record["env_id"]
            )
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
    event_cfg = TrackedEventCfg("observed", predicates=[_condition])
    env.progress_tracker = ProgressTracker(_success_criteria(), 1, "cpu", tracked_events=[event_cfg])
    tracker = env.progress_tracker
    with pytest.raises(AssertionError, match="step_index"):
        tracker.step(env)
    with pytest.raises(AssertionError, match="unique"):
        ProgressTracker(_success_criteria(), 1, "cpu", tracked_events=[event_cfg, event_cfg])
    with pytest.raises(AssertionError, match="Task success requires"):
        ProgressTracker([], 1, "cpu", tracked_events=[event_cfg])
    with pytest.raises(TypeError, match="predicates"):
        TrackedEventCfg("invalid")
    with pytest.raises(AssertionError, match="one desired state"):
        TrackedEventCfg("invalid", predicates={"a": [_condition, _condition]}, desired_predicate_state=[True])
    for invalid in ([], {"empty": []}, {}, None, _condition, (_condition,), {"invalid": _condition}):
        with pytest.raises(AssertionError):
            TrackedEventCfg("invalid", predicates=invalid)
    with pytest.raises(AssertionError, match="At least one"):
        TrackedEventCfg("invalid", predicates=[_condition], desired_predicate_state=[None])
    _step(tracker, env)
    with pytest.raises(AssertionError, match="exactly one"):
        tracker.step(env, env.episode_length_buf)
    assert _events(tracker) == [("observed", 1)]
    cfg = ProgressTrackingRecorderCfg()
    cfg.class_type(cfg, env).record_post_step()
    tracker.reset([0])
    result = json.loads(json.dumps(record_progress_results(env, 0)))
    assert result["progress"]["tracked_event_names"] == ["observed"]
    assert result["progress"]["tracked_events"] == [{"name": "observed", "step": 1}]
    assert result["progress"]["control_step_dt"] == 0.02
    assert result["progress"]["overall_score"] == 0.0
    assert not result["progress"]["all_complete"]
    assert list(result["progress"]["criteria_by_name"]) == ["success"]
    assert result["progress"]["events"] == []
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
            return [TrackedEventCfg("observed", predicates=[_condition])]

    description = IsaacLabArenaEnvironment(
        name="observing_task",
        scene=Scene(),
        task=NoTask(),
        embodiment=NoEmbodiment(),
    )
    builder = ArenaEnvBuilder(description, ArenaEnvBuilderCfg(solve_relations=False, device="cpu"))
    cfg, _ = builder.compose_manager_cfg()
    assert cfg.terminations.to_dict() == {}
    assert cfg.task_progress is None
    description.task = ObservingTask()
    with pytest.raises(AssertionError, match="Tracked events require task success criteria"):
        builder.compose_manager_cfg()

    class ObservingTaskWithSuccess(ObservingTask):
        def get_termination_cfg(self):
            return TaskTerminationCfg(timeout_s=None, success=_success_criteria())

    task = ObservingTaskWithSuccess()
    description.task = task
    cfg, _ = builder.compose_manager_cfg()
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


def test_event_activity_does_not_change_success():
    assert run_function_with_persistent_simulation_app(_test_event_activity_does_not_change_success)


def test_order_and_fresh_history():
    assert run_function_with_persistent_simulation_app(_test_order_and_fresh_history)


def test_desired_states_and_none_gating():
    assert run_function_with_persistent_simulation_app(_test_desired_states_and_none_gating)


def test_temporal_event_isolation():
    assert run_function_with_persistent_simulation_app(_test_temporal_event_isolation)


def test_callable_events():
    assert run_function_with_persistent_simulation_app(_test_callable_events)


def test_single_tracker_and_shared_cache():
    assert run_function_with_persistent_simulation_app(_test_single_tracker_and_shared_cache)


def test_desired_states_recheck_completed_temporal_members():
    assert run_function_with_persistent_simulation_app(_test_desired_states_recheck_completed_temporal_members)


def test_rearm_requires_fresh_temporal_sequence():
    assert run_function_with_persistent_simulation_app(_test_rearm_requires_fresh_temporal_sequence)


def test_temporal_desired_states_while_waiting():
    assert run_function_with_persistent_simulation_app(_test_temporal_desired_states_while_waiting)


def test_named_sequences_and_desired_state_order():
    assert run_function_with_persistent_simulation_app(_test_named_sequences_and_desired_state_order)


def test_any_order_rearm_and_desired_states():
    assert run_function_with_persistent_simulation_app(_test_any_order_rearm_and_desired_states)


def test_finished_episode_events(tmp_path):
    assert run_function_with_persistent_simulation_app(_test_finished_episode_events, output_dir=tmp_path)


def test_recording_and_validation():
    assert run_function_with_persistent_simulation_app(_test_recording_and_validation)


def test_builder_and_composition():
    assert run_function_with_persistent_simulation_app(_test_builder_and_composition)

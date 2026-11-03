# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.tests.test_tracked_events import _condition, _env, _events, _step, _success_criteria
from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_latched_events_reset_independently(_app):
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.progress_tracking.tracked_event_cfg import TrackedEventCfg

    env = _env([[True], [False]])
    baseline = ProgressTracker(_success_criteria(), 2, "cpu")
    tracker = ProgressTracker(
        _success_criteria(), 2, "cpu", tracked_events=[TrackedEventCfg("first_seen", [_condition], latch=True)]
    )
    for values in ([[True], [False]], [[False], [True]], [[True], [True]]):
        env.conditions[:] = env.conditions.new_tensor(values)
        _step(tracker, env)
        baseline.step(env, env.episode_length_buf)
        assert tracker.get_state() == baseline.get_state()
        assert tracker.get_events() == baseline.get_events()
    assert _events(tracker, 0) == [("first_seen", 1)]
    assert _events(tracker, 1) == [("first_seen", 2)]

    tracker.reset([0])
    baseline.reset([0])
    env.episode_length_buf[0] = 0
    _step(tracker, env)
    baseline.step(env, env.episode_length_buf)
    assert tracker.get_state() == baseline.get_state()
    assert tracker.get_events() == baseline.get_events()
    assert _events(tracker, 0) == [("first_seen", 1)]
    assert _events(tracker, 1) == [("first_seen", 2)]
    return True


def test_latched_events_reset_independently():
    assert run_function_with_persistent_simulation_app(_test_latched_events_reset_independently)


def _test_latched_temporal_event_retains_first_completion(_app):
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.progress_tracking.tracked_event_cfg import TrackedEventCfg
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    env = _env([[True]])
    tracker = ProgressTracker(
        _success_criteria(),
        1,
        "cpu",
        tracked_events=[TrackedEventCfg("held_once", [TrueForConsecutiveStepsCfg(_condition, 2)], latch=True)],
    )
    _step(tracker, env)
    assert _events(tracker) == []
    _step(tracker, env)
    assert _events(tracker) == [("held_once", 2)]
    for value in (False, True, True):
        env.conditions[:] = value
        _step(tracker, env)
    assert _events(tracker) == [("held_once", 2)]

    tracker.reset([0])
    env.episode_length_buf[0] = 0
    _step(tracker, env)
    assert _events(tracker) == []
    _step(tracker, env)
    assert _events(tracker) == [("held_once", 2)]
    return True


def test_latched_temporal_event_retains_first_completion():
    assert run_function_with_persistent_simulation_app(_test_latched_temporal_event_retains_first_completion)


def _test_fully_latched_event_stops_evaluating(_app):
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.progress_tracking.tracked_event_cfg import TrackedEventCfg

    calls = []

    def counting_condition(env):
        calls.append(True)
        return _condition(env)

    env = _env([[True], [True]])
    tracker = ProgressTracker(
        _success_criteria(), 2, "cpu", tracked_events=[TrackedEventCfg("once", [counting_condition], latch=True)]
    )
    for value in (True, False, True, True):
        env.conditions[:] = value
        _step(tracker, env)
    assert len(calls) == 1
    assert _events(tracker, 0) == [("once", 1)]
    assert _events(tracker, 1) == [("once", 1)]
    return True


def test_fully_latched_event_stops_evaluating():
    assert run_function_with_persistent_simulation_app(_test_fully_latched_event_stops_evaluating)


def _test_latched_and_recurring_events_share_predicate_cache(_app):
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.progress_tracking.tracked_event_cfg import TrackedEventCfg

    calls = []

    def shared_condition(env):
        calls.append(True)
        return _condition(env)

    env = _env([[True]])
    tracker = ProgressTracker(
        _success_criteria(),
        1,
        "cpu",
        tracked_events=[
            TrackedEventCfg("latched", [shared_condition], latch=True),
            TrackedEventCfg("recurring", [shared_condition]),
        ],
    )
    for value in (True, False, True):
        env.conditions[:] = value
        _step(tracker, env)
    assert len(calls) == 3
    assert _events(tracker) == [("latched", 1), ("recurring", 1), ("recurring", 3)]
    return True


def test_latched_and_recurring_events_share_predicate_cache():
    assert run_function_with_persistent_simulation_app(_test_latched_and_recurring_events_share_predicate_cache)

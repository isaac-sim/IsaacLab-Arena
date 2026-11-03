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


def _test_details_capture_completed_occurrences_and_copy_values(_app):
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.progress_tracking.tracked_event_cfg import TrackedEventCfg

    env = _env([[False], [True]])
    measurements = [{"height_m": [0.1]}, {"height_m": [0.2]}]
    calls = []

    def capture_details(current_env, env_idx):
        assert current_env is env
        calls.append((env_idx, int(env.episode_length_buf[env_idx])))
        return measurements[env_idx]

    tracker = ProgressTracker(
        _success_criteria(),
        2,
        "cpu",
        tracked_events=[TrackedEventCfg("arrival", [_condition], event_details=capture_details)],
    )
    _step(tracker, env)
    assert calls == [(1, 1)]
    measurements[1]["height_m"][0] = 0.9
    snapshot = tracker.get_tracked_event_occurrences()
    assert snapshot[1][0].details == {"height_m": [0.2]}
    snapshot[1][0].details["height_m"][0] = -1
    assert tracker.get_tracked_event_occurrences()[1][0].details == {"height_m": [0.2]}

    env.conditions[:] = True
    _step(tracker, env)
    assert calls == [(1, 1), (0, 2)]
    _step(tracker, env)
    assert calls == [(1, 1), (0, 2)], "A sustained condition must not recapture details."
    env.conditions[1] = False
    _step(tracker, env)
    env.conditions[1] = True
    _step(tracker, env)
    assert calls[-1] == (1, 5)
    assert [event.details for event in tracker.get_tracked_event_occurrences()[1]] == [
        {"height_m": [0.2]},
        {"height_m": [0.9]},
    ]

    tracker.reset([0])
    env.episode_length_buf[0] = 0
    measurements[0]["height_m"][0] = 0.4
    _step(tracker, env)
    assert calls[-1] == (0, 1)
    assert tracker.get_tracked_event_occurrences()[0][0].details == {"height_m": [0.4]}
    assert len(tracker.get_tracked_event_occurrences()[1]) == 2
    return True


def test_details_capture_completed_occurrences_and_copy_values():
    assert run_function_with_persistent_simulation_app(_test_details_capture_completed_occurrences_and_copy_values)


def _test_details_wait_for_sequence_completion(_app):
    from functools import partial

    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.progress_tracking.tracked_event_cfg import TrackedEventCfg
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg

    env = _env([[True, False]])
    calls = []

    def capture_details(env, env_idx):
        calls.append(int(env.episode_length_buf[env_idx]))
        return {"step": calls[-1]}

    tracker = ProgressTracker(
        _success_criteria(),
        1,
        "cpu",
        tracked_events=[
            TrackedEventCfg(
                "held_arrival",
                [_condition, TrueForConsecutiveStepsCfg(partial(_condition, index=1), 2)],
                latch=True,
                event_details=capture_details,
            )
        ],
    )
    _step(tracker, env)
    env.conditions[:, 1] = True
    _step(tracker, env)
    assert calls == []
    _step(tracker, env)
    assert calls == [3]
    env.conditions[:] = False
    _step(tracker, env)
    assert calls == [3]
    assert tracker.get_tracked_event_occurrences()[0][0].details == {"step": 3}
    return True


def test_details_wait_for_sequence_completion():
    assert run_function_with_persistent_simulation_app(_test_details_wait_for_sequence_completion)


def _test_recorded_details_are_optional_and_independent(_app):
    import json

    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker, ProgressTrackingRecorderCfg
    from isaaclab_arena.progress_tracking.tracked_event_cfg import TrackedEventCfg
    from isaaclab_arena.recording.progress_terms import record_progress_results

    for callback in (None, lambda env, env_idx: {}, lambda env, env_idx: {"height_m": [0.3]}):
        env = _env([[True]])
        tracker = ProgressTracker(
            _success_criteria(),
            1,
            "cpu",
            tracked_events=[TrackedEventCfg("arrival", [_condition], event_details=callback)],
        )
        env.progress_tracker = tracker
        _step(tracker, env)
        cfg = ProgressTrackingRecorderCfg()
        cfg.class_type(cfg, env).record_post_step()
        recorded = record_progress_results(env, 0)["progress"]["tracked_events"]
        if callback is None or not callback(env, 0):
            assert json.dumps(recorded) == '[{"name": "arrival", "step": 1}]'
        else:
            assert json.loads(json.dumps(recorded)) == [{"name": "arrival", "step": 1, "details": {"height_m": [0.3]}}]
            recorded[0]["details"]["height_m"][0] = -1
            assert record_progress_results(env, 0)["progress"]["tracked_events"][0]["details"] == {"height_m": [0.3]}
            assert tracker.get_tracked_event_occurrences()[0][0].details == {"height_m": [0.3]}
    return True


def test_recorded_details_are_optional_and_independent():
    assert run_function_with_persistent_simulation_app(_test_recorded_details_are_optional_and_independent)


def _test_details_configuration_rejects_invalid_callbacks(_app):
    import pytest

    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.progress_tracking.tracked_event_cfg import TrackedEventCfg

    with pytest.raises(AssertionError, match="callable"):
        TrackedEventCfg("arrival", [_condition], event_details={})
    env = _env([[True]])
    tracker = ProgressTracker(
        _success_criteria(),
        1,
        "cpu",
        tracked_events=[TrackedEventCfg("arrival", [_condition], event_details=lambda env, env_idx: [])],
    )
    with pytest.raises(AssertionError, match="dictionary"):
        _step(tracker, env)
    return True


def test_details_configuration_rejects_invalid_callbacks():
    assert run_function_with_persistent_simulation_app(_test_details_configuration_rejects_invalid_callbacks)


def _test_occurrences_with_details_are_hashable(_app):
    from isaaclab_arena.progress_tracking.tracked_event_cfg import TrackedEventOccurrence

    occurrence = TrackedEventOccurrence(0, "arrival", 1, {"height_m": 0.3})
    assert isinstance(hash(occurrence), int)
    assert occurrence != TrackedEventOccurrence(0, "arrival", 1, {"height_m": 0.4})
    return True


def test_occurrences_with_details_are_hashable():
    assert run_function_with_persistent_simulation_app(_test_occurrences_with_details_are_hashable)

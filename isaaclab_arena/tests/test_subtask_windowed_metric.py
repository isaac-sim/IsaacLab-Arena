# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pure-mock tests for ``SubtaskWindowedMetric``.

The windowing logic is independent of Isaac Sim. These tests stand up a fake inner
metric whose ``compute_metric_from_recording`` returns the windowed slices it received
so we can directly assert the per-episode windows produced by the wrapper.
"""

import numpy as np
import traceback

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

HEADLESS = True


def _make_recording_inner_metric():
    """Inner metric stub whose ``compute_metric_from_recording`` returns the windowed
    slices passed in. We use this to inspect what the wrapper fed downstream. Built
    inside a function so the ``MetricBase`` import happens after the simulation app
    starts (deferred imports are required for tests using ``run_simulation_app_function``).
    """
    from isaaclab_arena.metrics.metric_base import MetricBase

    class _RecordingInnerMetric(MetricBase):
        name = "inner"
        recorder_term_name = "inner_term"

        def get_recorder_term_cfg(self):
            return None

        def compute_metric_from_recording(self, recorded_metric_data, context=None):
            return [arr.tolist() for arr in recorded_metric_data]

    return _RecordingInnerMetric()


def _state(rows: list[list[bool]]) -> np.ndarray:
    return np.array(rows, dtype=bool)


def _data(values: list[int]) -> np.ndarray:
    return np.array(values, dtype=int)


def _test_windowed_metric_from_episode_start(simulation_app) -> bool:
    """Composite (unordered) scope: window is [0, T_i] inclusive of the latching step."""

    from isaaclab_arena.metrics.subtask_windowed_metric import (
        SUBTASK_SUCCESS_STATE_RECORDER_NAME,
        WINDOW_SCOPE_FROM_EPISODE_START,
        SubtaskWindowedMetric,
    )

    try:
        inner = _make_recording_inner_metric()
        wrapper = SubtaskWindowedMetric(
            inner_metric=inner,
            subtask_idx=1,
            scope=WINDOW_SCOPE_FROM_EPISODE_START,
        )
        assert wrapper.extra_recorder_term_dependencies == [SUBTASK_SUCCESS_STATE_RECORDER_NAME]
        assert wrapper.name == inner.name
        assert wrapper.recorder_term_name == inner.recorder_term_name

        # Episode A: 5 steps; subtask 1 latches True at step 2 → window = [0, 2].
        # Episode B: 4 steps; subtask 1 never latches → window = [0, 3] (whole episode).
        # Episode C: 3 steps; subtask 1 latches at step 0 → window = [0, 0].
        episode_state = [
            _state([[False, False], [False, False], [True, True], [True, True], [True, True]]),
            _state([[False, False], [True, False], [True, False], [True, False]]),
            _state([[False, True], [True, True], [True, True]]),
        ]
        episode_data = [
            _data([10, 11, 12, 13, 14]),
            _data([20, 21, 22, 23]),
            _data([30, 31, 32]),
        ]
        context = {SUBTASK_SUCCESS_STATE_RECORDER_NAME: episode_state}

        windowed = wrapper.compute_metric_from_recording(episode_data, context=context)
        assert windowed == [[10, 11, 12], [20, 21, 22, 23], [30]], f"got {windowed}"

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    return True


def _test_windowed_metric_from_prev_done(simulation_app) -> bool:
    """Sequential scope: window starts the step AFTER the previous subtask latches."""

    from isaaclab_arena.metrics.subtask_windowed_metric import (
        SUBTASK_SUCCESS_STATE_RECORDER_NAME,
        WINDOW_SCOPE_FROM_PREV_DONE,
        SubtaskWindowedMetric,
    )

    try:
        inner = _make_recording_inner_metric()

        # subtask_idx == 0 → start always 0 (same as composite scope).
        wrapper0 = SubtaskWindowedMetric(
            inner_metric=inner,
            subtask_idx=0,
            scope=WINDOW_SCOPE_FROM_PREV_DONE,
        )
        episode_state = [
            _state([[False, False], [True, False], [True, True], [True, True]]),
        ]
        episode_data = [_data([100, 101, 102, 103])]
        context = {SUBTASK_SUCCESS_STATE_RECORDER_NAME: episode_state}
        windowed = wrapper0.compute_metric_from_recording(episode_data, context=context)
        # subtask 0 latches at step 1 → window = [0, 1].
        assert windowed == [[100, 101]], f"subtask 0 window mismatch: {windowed}"

        # subtask_idx == 1 → starts at step after subtask 0 latched.
        wrapper1 = SubtaskWindowedMetric(
            inner_metric=inner,
            subtask_idx=1,
            scope=WINDOW_SCOPE_FROM_PREV_DONE,
        )
        # Episode A: 5 steps; subtask 0 latches step 1, subtask 1 latches step 3
        #   → window = [2, 3].
        # Episode B: 4 steps; subtask 0 latches step 2, subtask 1 never latches
        #   → window = [3, 3] (open-ended to end of episode).
        # Episode C: 3 steps; subtask 0 never latches → subtask 1 never active → empty.
        # Episode D: 4 steps; subtask 0 latches on final step → empty (no room for 1).
        episode_state = [
            _state([
                [False, False],
                [True, False],
                [True, False],
                [True, True],
                [True, True],
            ]),
            _state([
                [False, False],
                [False, False],
                [True, False],
                [True, False],
            ]),
            _state([[False, False], [False, False], [False, False]]),
            _state([
                [False, False],
                [False, False],
                [False, False],
                [True, False],
            ]),
        ]
        episode_data = [
            _data([10, 11, 12, 13, 14]),
            _data([20, 21, 22, 23]),
            _data([30, 31, 32]),
            _data([40, 41, 42, 43]),
        ]
        context = {SUBTASK_SUCCESS_STATE_RECORDER_NAME: episode_state}
        windowed = wrapper1.compute_metric_from_recording(episode_data, context=context)
        assert windowed == [[12, 13], [23], [], []], f"got {windowed}"

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    return True


def _test_windowed_metric_passes_through_to_inner(simulation_app) -> bool:
    """The wrapper must feed sliced data into the inner metric and forward its result."""

    from isaaclab.managers.recorder_manager import RecorderTermCfg

    from isaaclab_arena.metrics.metric_base import MetricBase
    from isaaclab_arena.metrics.subtask_windowed_metric import (
        SUBTASK_SUCCESS_STATE_RECORDER_NAME,
        WINDOW_SCOPE_FROM_EPISODE_START,
        SubtaskWindowedMetric,
    )

    try:
        class _SumMetric(MetricBase):
            name = "sum"
            recorder_term_name = "sum_term"

            def get_recorder_term_cfg(self) -> RecorderTermCfg | None:
                return None

            def compute_metric_from_recording(self, recorded_metric_data, context=None):
                return float(sum(int(arr.sum()) for arr in recorded_metric_data))

        wrapper = SubtaskWindowedMetric(
            inner_metric=_SumMetric(),
            subtask_idx=0,
            scope=WINDOW_SCOPE_FROM_EPISODE_START,
        )

        # Episode: subtask 0 latches at step 2; data is [1, 2, 3, 4, 5];
        # windowed = [1, 2, 3]; sum = 6.
        episode_state = [_state([[False], [False], [True], [True], [True]])]
        episode_data = [_data([1, 2, 3, 4, 5])]
        context = {SUBTASK_SUCCESS_STATE_RECORDER_NAME: episode_state}
        result = wrapper.compute_metric_from_recording(episode_data, context=context)
        assert result == 6.0, f"expected 6.0, got {result}"

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    return True


def _test_composite_wraps_metrics_when_enabled(simulation_app) -> bool:
    """``_combine_subtask_metrics`` wraps subtask metrics in ``SubtaskWindowedMetric``
    only when ``window_subtask_metrics=True``, and uses the class-level scope."""

    from isaaclab_arena.metrics.metric_base import MetricBase
    from isaaclab_arena.metrics.subtask_windowed_metric import (
        WINDOW_SCOPE_FROM_EPISODE_START,
        WINDOW_SCOPE_FROM_PREV_DONE,
        SubtaskWindowedMetric,
    )
    from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase
    from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase

    class _FakeMetric(MetricBase):
        name = "object_moved_rate"
        recorder_term_name = "object_linear_velocity"

        def get_recorder_term_cfg(self):
            return None

        def compute_metric_from_recording(self, recorded_metric_data, context=None):
            return 0.0

    class _FakeSubtask:
        def __init__(self):
            self._metric = _FakeMetric()

        def get_metrics(self):
            # Return a fresh instance per call (matches expected TaskBase contract that
            # ``_combine_subtask_metrics`` mutates the returned objects).
            m = _FakeMetric()
            m.name = "object_moved_rate"
            m.recorder_term_name = "object_linear_velocity"
            return [m]

    def _make_composite(cls, *, window: bool):
        # Bypass __init__ to avoid building real Isaac Lab event/termination configs —
        # ``_combine_subtask_metrics`` only needs ``subtasks`` and the flag.
        obj = cls.__new__(cls)
        obj.subtasks = [_FakeSubtask(), _FakeSubtask()]
        obj.window_subtask_metrics = window
        return obj

    try:
        # Flag off: metrics are renamed but not wrapped — one entry per subtask metric.
        unwrapped = _make_composite(CompositeTaskBase, window=False)._combine_subtask_metrics([0, 1])
        assert len(unwrapped) == 2
        for m in unwrapped:
            assert not isinstance(m, SubtaskWindowedMetric)
        assert {m.name for m in unwrapped} == {"object_moved_rate_subtask_0", "object_moved_rate_subtask_1"}

        # Flag on: BOTH the episode-wide bare metric and the windowed wrapper appear,
        # side-by-side, under distinct ``name``s but sharing ``recorder_term_name``.
        both = _make_composite(CompositeTaskBase, window=True)._combine_subtask_metrics([0, 1])
        assert len(both) == 4
        bare = [m for m in both if not isinstance(m, SubtaskWindowedMetric)]
        wrapped = [m for m in both if isinstance(m, SubtaskWindowedMetric)]
        assert len(bare) == 2 and len(wrapped) == 2
        assert {m.name for m in bare} == {
            "object_moved_rate_subtask_0",
            "object_moved_rate_subtask_1",
        }
        assert {m.name for m in wrapped} == {
            "object_moved_rate_subtask_0_windowed",
            "object_moved_rate_subtask_1_windowed",
        }
        for m in wrapped:
            assert m.scope == WINDOW_SCOPE_FROM_EPISODE_START
            assert m.recorder_term_name in {
                "object_linear_velocity_subtask_0",
                "object_linear_velocity_subtask_1",
            }

        # Sequential: same shape, but wrappers use FROM_PREV_DONE scope.
        both_seq = _make_composite(SequentialTaskBase, window=True)._combine_subtask_metrics([0, 1])
        assert len(both_seq) == 4
        wrapped_seq = [m for m in both_seq if isinstance(m, SubtaskWindowedMetric)]
        assert len(wrapped_seq) == 2
        for m in wrapped_seq:
            assert m.scope == WINDOW_SCOPE_FROM_PREV_DONE

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    return True


def test_composite_wraps_metrics_when_enabled():
    result = run_simulation_app_function(
        _test_composite_wraps_metrics_when_enabled,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_composite_wraps_metrics_when_enabled.__name__} failed"


def test_windowed_metric_from_episode_start():
    result = run_simulation_app_function(
        _test_windowed_metric_from_episode_start,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_windowed_metric_from_episode_start.__name__} failed"


def test_windowed_metric_from_prev_done():
    result = run_simulation_app_function(
        _test_windowed_metric_from_prev_done,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_windowed_metric_from_prev_done.__name__} failed"


def test_windowed_metric_passes_through_to_inner():
    result = run_simulation_app_function(
        _test_windowed_metric_passes_through_to_inner,
        headless=HEADLESS,
    )
    assert result, f"Test {_test_windowed_metric_passes_through_to_inner.__name__} failed"


if __name__ == "__main__":
    test_windowed_metric_from_episode_start()
    test_windowed_metric_from_prev_done()
    test_windowed_metric_passes_through_to_inner()
    test_composite_wraps_metrics_when_enabled()

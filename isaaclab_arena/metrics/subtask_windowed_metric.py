# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from isaaclab.managers.recorder_manager import RecorderTermCfg

from isaaclab_arena.metrics.metric_base import MetricBase

# Scope of a subtask's evaluation window.
WINDOW_SCOPE_FROM_EPISODE_START = "from_episode_start"
WINDOW_SCOPE_FROM_PREV_DONE = "from_prev_done"

# Recorder term name written by ``SubtaskSuccessStateRecorder`` (in
# ``isaaclab_arena.tasks.composite_task_base``). Per-step shape: (num_subtasks,) bool,
# True wherever a subtask has ever latched success during the episode so far.
SUBTASK_SUCCESS_STATE_RECORDER_NAME = "subtask_success_rate"


class SubtaskWindowedMetric(MetricBase):
    """Wraps a per-subtask metric so it is computed only over the subtask's relevant
    window of an episode, rather than across the entire episode.

    The window per episode is derived from the per-step latched subtask success state
    recorded under ``SUBTASK_SUCCESS_STATE_RECORDER_NAME``. The two supported scopes:

    - ``WINDOW_SCOPE_FROM_EPISODE_START``: window is ``[0, T_i]`` where ``T_i`` is the
      first step where subtask ``i`` latches True (else end of episode). Intended for
      unordered composite tasks.
    - ``WINDOW_SCOPE_FROM_PREV_DONE``: window is ``[T_{i-1}+1, T_i]`` for ``i > 0`` and
      ``[0, T_0]`` for ``i == 0``. If the previous subtask never latched, the subtask
      was never active and the window is empty. Intended for sequential composite tasks.
    """

    def __init__(
        self,
        inner_metric: MetricBase,
        subtask_idx: int,
        scope: str,
        name: str | None = None,
        state_recorder_name: str = SUBTASK_SUCCESS_STATE_RECORDER_NAME,
    ):
        assert scope in (WINDOW_SCOPE_FROM_EPISODE_START, WINDOW_SCOPE_FROM_PREV_DONE), (
            f"Unknown subtask window scope: {scope}"
        )
        self.inner_metric = inner_metric
        self.subtask_idx = subtask_idx
        self.scope = scope
        self.state_recorder_name = state_recorder_name
        # The wrapper reuses the inner's recorder_term_name (it reads the same HDF5
        # stream), but takes a distinct ``name`` so it can coexist in the metric list
        # alongside the unwrapped (episode-wide) version under a different output key.
        self.name = name if name is not None else inner_metric.name
        self.recorder_term_name = inner_metric.recorder_term_name

    @property
    def extra_recorder_term_dependencies(self) -> list[str]:
        return [self.state_recorder_name]

    def get_recorder_term_cfg(self) -> RecorderTermCfg:
        return self.inner_metric.get_recorder_term_cfg()

    def compute_metric_from_recording(
        self,
        recorded_metric_data: list[np.ndarray],
        context: dict[str, list[np.ndarray]] | None = None,
    ):
        assert (
            context is not None and self.state_recorder_name in context
        ), f"SubtaskWindowedMetric requires '{self.state_recorder_name}' in context"
        subtask_state_per_demo = context[self.state_recorder_name]
        assert len(subtask_state_per_demo) == len(recorded_metric_data), (
            "Mismatched episode counts between metric data and subtask state recording"
        )

        windowed_data: list[np.ndarray] = []
        for ep_metric_data, ep_state in zip(recorded_metric_data, subtask_state_per_demo):
            start, end_exclusive = self._compute_window(ep_state)
            if start is None or start >= end_exclusive:
                # Subtask was never active during this episode — empty slice.
                windowed_data.append(ep_metric_data[:0])
            else:
                windowed_data.append(ep_metric_data[start:end_exclusive])

        # The inner metric is fed already-windowed data and reduces it as-is — it never
        # needs ``context`` itself (the wrapper consumed it for slicing).
        return self.inner_metric.compute_metric_from_recording(windowed_data)

    def _compute_window(self, ep_state: np.ndarray) -> tuple[int | None, int]:
        """Returns ``(start, end_exclusive)`` for slicing this subtask's window.

        ``ep_state`` is shape ``(T, num_subtasks)`` of bool. ``start = None`` signals
        that the subtask was never active in this episode.
        """
        assert ep_state.ndim == 2, f"Expected (T, num_subtasks) state; got shape {ep_state.shape}"
        T = ep_state.shape[0]
        if T == 0:
            return None, 0

        # End of window: include the step at which this subtask first latches True. If
        # it never latches, the window extends to the end of the recorded episode.
        done_steps = np.where(ep_state[:, self.subtask_idx])[0]
        end_exclusive = int(done_steps[0]) + 1 if len(done_steps) > 0 else T

        if self.scope == WINDOW_SCOPE_FROM_EPISODE_START:
            start: int | None = 0
        else:  # WINDOW_SCOPE_FROM_PREV_DONE
            if self.subtask_idx == 0:
                start = 0
            else:
                prev_done_steps = np.where(ep_state[:, self.subtask_idx - 1])[0]
                if len(prev_done_steps) == 0:
                    start = None  # Previous subtask never completed → this one never active.
                else:
                    start = int(prev_done_steps[0]) + 1
                    if start >= T:
                        start = None  # Prev finished on the final step → no room for this one.

        return start, end_exclusive

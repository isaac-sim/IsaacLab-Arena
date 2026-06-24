# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Episode recorder term that captures the progress-tracking state of each finishing episode."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from isaaclab.utils import configclass

from isaaclab_arena.progress_tracking.progress_tracker import get_progress_tracker
from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg


def record_progress_results(env, env_id: int) -> dict[str, Any]:
    """Record the progress-tracking state for ``env_id``.

    Fired at episode end (pre-reset), while the tracker still holds the finished episode's state.
    Returns ``{}`` when progress tracking is not active on the env. All fields are nested under a
    single ``progress`` key (JSON-serializable) so they cannot collide with other recorder terms.
    """
    tracker = get_progress_tracker(env)
    if tracker is None:
        return {}

    state = tracker.get_state()[env_id]
    events = tracker.get_events()[env_id]
    return {
        "progress": {
            "overall_score": state.overall_score,
            "all_complete": state.all_complete,
            "objectives": {
                name: {
                    "score": obj.score,
                    "is_complete": obj.is_complete,
                    "completed_groups": obj.completed_groups,
                    "total_groups": obj.total_groups,
                }
                for name, obj in state.progress_objectives.items()
            },
            # Per-episode predicate transitions, in the order they fired (step = episode-local step).
            "events": [
                {
                    "step": event.step,
                    "objective": event.progress_objective,
                    "group": event.group,
                    "predicate_index": event.predicate_index,
                    "predicate_name": event.predicate_name,
                    "score_delta": event.score_delta,
                }
                for event in events
            ],
        }
    }


@configclass
class ProgressEpisodeRecorderTermCfg(EpisodeRecorderTermCfg):
    """Term recording each finishing episode's final progress-tracking state and predicate events."""

    func: Callable[..., dict[str, Any]] = record_progress_results

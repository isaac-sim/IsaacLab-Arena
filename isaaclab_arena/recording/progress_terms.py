# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Episode recorder term that captures the progress-tracking state of each finishing episode."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from isaaclab.utils.configclass import configclass

from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg


def record_progress_results(env, env_id: int) -> dict[str, Any]:
    """Record the progress-tracking state for ``env_id``."""
    progress = env.extras.get("progress_tracking")
    if not progress:
        return {}

    state = progress["states"][env_id]
    events = progress["events"][env_id]
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
                    "active_predicates": obj.active_predicates,
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
    """Term recording each episode's final progress-tracking state and predicate events."""

    func: Callable[..., dict[str, Any]] = record_progress_results

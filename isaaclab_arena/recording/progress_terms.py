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
    objectives = {}
    for name, objective in state.progress_objectives.items():
        objective_record = {
            "score": objective.score,
            "is_complete": objective.is_complete,
            "completed_groups": objective.completed_groups,
            "total_groups": objective.total_groups,
            "active_predicates": objective.active_predicates,
        }
        if objective.diagnostic_predicates:
            first_pass_events = [
                {"step": check["first_true_step"], "check": check_name}
                for check_name, check in objective.diagnostic_predicates.items()
                if check["first_true_step"] is not None
            ]
            first_pass_events.sort(key=lambda event: event["step"])
            objective_record["intermediate_checks"] = {
                "best_simultaneous": objective.best_simultaneous_checks,
                "total": len(objective.diagnostic_predicates),
                "checks": objective.diagnostic_predicates,
                "first_pass_events": first_pass_events,
            }
        objectives[name] = objective_record
    return {
        "progress": {
            "overall_score": state.overall_score,
            "all_complete": state.all_complete,
            "objectives": objectives,
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

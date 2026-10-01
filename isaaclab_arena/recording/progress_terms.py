# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Episode recorder term that captures the progress-tracking state of each finishing episode."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from isaaclab.utils.configclass import configclass

from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg


def record_progress_results(env, env_id: int) -> dict[str, Any]:
    """Record the progress-tracking state for ``env_id``."""
    progress = env.extras.get("progress_tracking")
    if not progress:
        return {}

    # Use the tracker's published snapshot of this episode. Re-evaluating a predicate
    # here could observe an already-reset scene or accidentally advance a stateful check.
    state = progress["states"][env_id]
    events = progress["events"][env_id]
    objectives = {}
    for name, objective in state.progress_objectives.items():
        # Milestone scores describe completed sequence entries. For [object(10),
        # gripper(10)], the score is 0.5 once only the object requirement has completed.
        objective_record = {
            "score": objective.score,
            "is_complete": objective.is_complete,
            "completed_groups": objective.completed_groups,
            "total_groups": objective.total_groups,
            "active_predicates": objective.active_predicates,
        }
        if objective.consecutive_step_progress:
            # Store live counters beside milestone scores, preserving group and sequence
            # index. Example: object completed 10/10, gripper active 6/10, score still 0.5.
            # asdict converts each snapshot to JSON-compatible fields, with no tensors.
            consecutive_step_progress = {}
            for group, requirements in objective.consecutive_step_progress.items():
                consecutive_step_progress[group] = [asdict(requirement) for requirement in requirements]
            objective_record["consecutive_step_progress"] = consecutive_step_progress
        if objective.tracked_predicates:
            # These optional instantaneous checks have a separate history. "First true
            # at step 2" means the check held then, not that its ten-step streak finished.
            first_true_events = [
                {"step": status["first_true_step"], "predicate": predicate_name}
                for predicate_name, status in objective.tracked_predicates.items()
                if status["first_true_step"] is not None
            ]
            # Present first observations chronologically even if configuration order differs.
            first_true_events.sort(key=lambda event: event["step"])
            objective_record["predicate_progress"] = {
                "max_simultaneous_true": objective.max_simultaneous_true,
                "total": len(objective.tracked_predicates),
                "predicates": objective.tracked_predicates,
                "first_true_events": first_true_events,
            }
        objectives[name] = objective_record
    return {
        "progress": {
            "overall_score": state.overall_score,
            "all_complete": state.all_complete,
            "objectives": objectives,
            # Completion events remain milestones, e.g. object at step 10 and gripper at
            # step 20. Intermediate streak counts belong to the objective snapshot above.
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

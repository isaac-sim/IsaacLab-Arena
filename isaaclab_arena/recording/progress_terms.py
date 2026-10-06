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
    criteria_by_name = {}
    for name, criteria_state in state.criteria_by_name.items():
        # Milestone scores describe completed sequence entries. For [object(10),
        # gripper(10)], the score is 0.5 once only the object requirement has completed.
        criteria_record = {
            "score": criteria_state.score,
            "is_complete": criteria_state.is_complete,
            "completed_sequences": criteria_state.completed_sequences,
            "total_sequences": criteria_state.total_sequences,
            "active_predicates": criteria_state.active_predicates,
        }
        counter_snapshots = state.consecutive_step_progress.get(name, {})
        if counter_snapshots:
            # Store live counters beside milestone scores, preserving sequence name and predicate
            # index. Example: object completed 10/10, gripper active 6/10, score still 0.5.
            # asdict converts each snapshot to JSON-compatible fields, with no tensors.
            consecutive_step_progress = {}
            for sequence_name, requirements in counter_snapshots.items():
                consecutive_step_progress[sequence_name] = [asdict(requirement) for requirement in requirements]
            criteria_record["consecutive_step_progress"] = consecutive_step_progress
        criteria_by_name[name] = criteria_record
    return {
        "progress": {
            "overall_score": state.overall_score,
            "all_complete": state.all_complete,
            "criteria_by_name": criteria_by_name,
            # Completion events remain milestones, e.g. object at step 10 and gripper at
            # step 20. Intermediate streak counts are copied from the progress snapshot above.
            "events": [
                {
                    "step": event.step,
                    "criteria_name": event.criteria_name,
                    "sequence_name": event.sequence_name,
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

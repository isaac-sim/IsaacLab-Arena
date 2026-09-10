# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Classify how a datagen episode ended, from the env's termination state."""

from __future__ import annotations

from typing import Any, Literal

EpisodeOutcome = Literal["success", "failure", "timeout"]


def classify_outcome(env: Any, env_id: int) -> EpisodeOutcome:
    """Classify env_id's just-finished episode from its active termination terms.

    Mirrors record_core_episode_results in isaaclab_arena/recording/common_terms.py,
    which reads the same "success" termination term for its own per-episode record.

    Args:
        env: IsaacLab environment instance (must have a termination_manager).
        env_id: Index of the env whose episode just ended.

    Returns:
        "success" if the success termination term fired, "timeout" if the time_out
        term fired, otherwise "failure".
    """
    active_terms = env.termination_manager.active_terms
    if "success" in active_terms and bool(env.termination_manager.get_term("success")[env_id]):
        return "success"
    if "time_out" in active_terms and bool(env.termination_manager.get_term("time_out")[env_id]):
        return "timeout"
    return "failure"

# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Map the termination term that ended a datagen episode to an outcome label."""

from __future__ import annotations


def classify_outcome(ended_by: str | None) -> str:
    """Classify an episode outcome from the termination term that ended it.

    Args:
        ended_by: Name of the stashed termination term that fired, or ``None``
            when the episode ended on the ``max_episode_length`` cap.

    Returns:
        ``"success"`` if the success term fired, ``"timeout"`` if nothing fired
        (length cap), otherwise ``"failure"``.
    """
    if ended_by is None:
        return "timeout"
    if ended_by == "success":
        return "success"
    return "failure"

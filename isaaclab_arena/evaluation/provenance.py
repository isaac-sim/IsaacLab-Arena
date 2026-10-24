# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Read the Arena commit stamp baked into the Docker image."""

from __future__ import annotations

import os

UNKNOWN_PROVENANCE_VALUE = "unknown"


def read_provenance() -> dict[str, str]:
    """Return the Arena git SHA and branch from the environment, ``"unknown"`` when unset or empty.

    Returns:
        Mapping with ``git_sha`` (from ``ARENA_GIT_SHA``) and ``git_branch`` (from ``ARENA_GIT_BRANCH``).
    """
    return {
        "git_sha": os.environ.get("ARENA_GIT_SHA") or UNKNOWN_PROVENANCE_VALUE,
        "git_branch": os.environ.get("ARENA_GIT_BRANCH") or UNKNOWN_PROVENANCE_VALUE,
    }

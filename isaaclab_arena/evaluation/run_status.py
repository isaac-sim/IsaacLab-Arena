# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The execution status of an Arena Run.

Kept in its own dependency-free module, and re-exported from ``arena_run``, so that reporting and
analysis tools can describe Run outcomes without importing the evaluation stack.
"""

from __future__ import annotations

from enum import Enum


class RunStatus(Enum):
    """Describe whether a run completed or failed."""

    COMPLETED = "completed"
    FAILED = "failed"

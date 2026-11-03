# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run status enum describing whether an Arena Run completed or failed."""

from enum import Enum


class RunStatus(Enum):
    """Describe whether a run completed or failed."""

    COMPLETED = "completed"
    FAILED = "failed"

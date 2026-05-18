# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Policy type definitions for Isaac Lab Arena OSMO workflows."""

from enum import Enum


class PolicyType(str, Enum):
    """Registered policy types supported by the Arena OSMO policy-runner workflow."""

    ZERO_ACTION = "zero_action"
    REPLAY = "replay"
    # TODO: Add more registered policy types here as their OSMO inputs are modeled.

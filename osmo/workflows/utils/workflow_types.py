# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Workflow type definitions for Isaac Lab Arena OSMO workflows."""

from enum import Enum


class WorkflowType(str, Enum):
    """Workflow types supported by the Arena OSMO workflow generation."""

    POLICY_RUNNER = "policy_runner"
    OPENPI_SERVER = "openpi_server"
    GR00T_SERVER = "gr00t_server"
    GR00T_POLICY_RUNNER = "gr00t_policy_runner"

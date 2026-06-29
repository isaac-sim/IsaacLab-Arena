# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass


@dataclass
class PhysicsSettleParams:
    """Configuration for the in-sim physics settle check."""

    num_steps: int = 5
    """Number of env steps to advance before reading back object state in the settle check. The settle
    check converts this to ``num_steps * decimation`` physics substeps internally."""

    lin_vel_thresh: float = 0.1
    """Max per-object linear speed (m/s) after settling. Above this the layout is considered unsettled."""

    ang_vel_thresh: float = 0.1
    """Max per-object angular speed (rad/s) after settling. Above this the layout is considered unsettled."""

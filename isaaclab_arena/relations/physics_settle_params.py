# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass


@dataclass
class PhysicsSettleParams:
    """Configuration for the post-reset, in-sim physics settle check."""

    num_steps: int = 10
    """Number of physics steps to advance before reading back object velocities in the settle check."""

    lin_vel_thresh: float = 0.1
    """Max per-object linear speed (m/s) after settling. Above this the layout is considered unsettled."""

    ang_vel_thresh: float = 0.1
    """Max per-object angular speed (rad/s) after settling. Above this the layout is considered unsettled."""
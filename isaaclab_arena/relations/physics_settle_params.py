# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass


@dataclass
class PhysicsSettleParams:
    """Configuration for the post-reset, in-sim physics settle check and re-selection.

    The check runs after ``env.reset()`` while the Sim App is live (see
    ``placement_events.verify_in_sim_and_reselect``). It is a distinct concern from the geometric
    placement solve configured by ``ObjectPlacerParams`` -- it validates layouts physically rather
    than geometrically -- so it is kept as its own parameter group. Whether the check runs at all is
    decided by the caller (the ``--enable_physics_settle_check`` CLI flag); this holds only its tuning.
    """

    num_steps: int = 10
    """Number of physics steps to advance before reading back object velocities in the settle check."""

    lin_vel_thresh: float = 0.1
    """Max per-object linear speed (m/s) after settling. Above this the layout is considered unsettled."""

    ang_vel_thresh: float = 0.1
    """Max per-object angular speed (rad/s) after settling. Above this the layout is considered unsettled."""

    max_retries: int = 3
    """Re-selection budget per env: how many replacement layouts to try when the drawn layout does not
    settle. The last layout tried is kept as a soft fallback even if it never settles."""

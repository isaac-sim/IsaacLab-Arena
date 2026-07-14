# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


def ensure_openpi_policy_registered() -> None:
    """Register the first-party OpenPI policy for typed Experiment composition."""
    from isaaclab_arena_openpi.policy import pi0_remote_policy  # noqa: F401

# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GR00T embodiment adapter for GR1 tabletop manipulation with arm and hand joint control."""

from __future__ import annotations

from isaaclab_arena_gr00t.policy.embodiment_adapters.joint_space_adapter import Gr00tJointSpaceAdapter


class Gr00tGr1ArmsAdapter(Gr00tJointSpaceAdapter):
    """Joint-space adapter for GR1, whose GR00T state and actions are the arm and hand joint groups.

    Each env action is the sim-ordered joint targets from the policy config's action joint YAML.
    """

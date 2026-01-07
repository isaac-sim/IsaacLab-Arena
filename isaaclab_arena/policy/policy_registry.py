# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.policy.replay_action_policy import ReplayActionPolicy
from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicy
from isaaclab_arena_gr00t.gr00t_closedloop_policy import Gr00tClosedloopPolicy
from isaaclab_arena_gr00t.replay_lerobot_action_policy import ReplayLerobotActionPolicy

POLICIES = {
    "zero_action": ZeroActionPolicy,
    "replay": ReplayActionPolicy,
    "replay_lerobot": ReplayLerobotActionPolicy,
    "gr00t_closedloop": Gr00tClosedloopPolicy,
}


def get_policy_cls(policy_type: str) -> type:
    """Get the policy class for the given policy type."""
    return POLICIES[policy_type]

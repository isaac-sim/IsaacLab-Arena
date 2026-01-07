# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


def get_policy_cls(policy_type: str) -> type:
    """Get the policy class for the given policy type.

    Uses lazy imports to avoid requiring optional dependencies like gr00t
    unless they're actually being used.
    """
    if policy_type == "zero_action":
        from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicy

        return ZeroActionPolicy
    elif policy_type == "replay":
        from isaaclab_arena.policy.replay_action_policy import ReplayActionPolicy

        return ReplayActionPolicy
    elif policy_type == "replay_lerobot":
        from isaaclab_arena_gr00t.replay_lerobot_action_policy import ReplayLerobotActionPolicy

        return ReplayLerobotActionPolicy
    elif policy_type == "gr00t_closedloop":
        from isaaclab_arena_gr00t.gr00t_closedloop_policy import Gr00tClosedloopPolicy

        return Gr00tClosedloopPolicy
    else:
        raise ValueError(
            f"Unknown policy type: {policy_type}. Available types: zero_action, replay, replay_lerobot,"
            " gr00t_closedloop"
        )

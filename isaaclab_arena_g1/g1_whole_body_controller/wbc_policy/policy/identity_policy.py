# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Policy that passes a target configuration through as the action."""

import numpy as np

from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy.base import WBCPolicy


class IdentityPolicy(WBCPolicy):
    """Identity policy that passes through the target pose."""

    def __init__(self):
        """Initialize the identity policy."""
        self.reset()

    def get_action(self, target_pose: np.ndarray) -> dict[str, np.ndarray]:
        """Return the target pose as the action.

        Args:
            target_pose: The target joint configuration.

        Returns:
            Dictionary with the target pose under the key `q`.
        """
        return {"q": target_pose}

    def reset(self):
        """Reset the policy. The identity policy has no state, so this does nothing."""
        pass

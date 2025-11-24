# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy.base import WBCPolicy


class IdentityPolicy(WBCPolicy):
    """Identity policy that passes through the target pose."""

    def __init__(self):
        """Initialize the identity policy."""
        self.reset()

    def get_action(self, target_pose: np.ndarray) -> dict[str, np.ndarray]:
        """Get the action for the identity policy."""
        return {"q": target_pose}

    def reset(self):
        pass

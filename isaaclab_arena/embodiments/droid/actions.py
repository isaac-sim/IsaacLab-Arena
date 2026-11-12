# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Action terms for the DROID embodiment."""

import torch

from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction


class BinaryJointPositionZeroToOneAction(BinaryJointPositionAction):
    """Binary joint position action that treats inputs above one half (or true) as close."""

    # override
    def process_actions(self, actions: torch.Tensor):
        """Map raw actions to the open or close joint command and clip them when clipping is configured.

        Args:
            actions: Raw actions per environment. Boolean true, or a numeric value above one half, selects close.
        """
        # store the raw actions
        self._raw_actions[:] = actions
        # compute the binary mask
        if actions.dtype == torch.bool:
            # true: close, false: open
            binary_mask = actions == 1
        else:
            # true: close, false: open
            binary_mask = actions > 0.5
        # compute the command
        self._processed_actions = torch.where(binary_mask, self._close_command, self._open_command)
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )

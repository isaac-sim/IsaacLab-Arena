# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction

DROID_GRIPPER_MIMIC_SIGNS = {
    "finger_joint": 1.0,
    "left_inner_finger_joint": -1.0,
    "left_inner_finger_knuckle_joint": -1.0,
    "right_outer_knuckle_joint": 1.0,
    "right_inner_finger_joint": 1.0,
    "right_inner_finger_knuckle_joint": -1.0,
}
DROID_GRIPPER_JOINT_NAMES = tuple(DROID_GRIPPER_MIMIC_SIGNS)
DROID_GRIPPER_OPEN_COMMAND = dict.fromkeys(DROID_GRIPPER_JOINT_NAMES, 0.0)
DROID_GRIPPER_CLOSE_COMMAND = {name: sign * 0.7 for name, sign in DROID_GRIPPER_MIMIC_SIGNS.items()}


class BinaryJointPositionZeroToOneAction(BinaryJointPositionAction):
    # override
    def process_actions(self, actions: torch.Tensor):
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

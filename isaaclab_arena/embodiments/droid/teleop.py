# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""SE(3) teleoperation devices using DROID's zero-to-one gripper commands."""

import torch

from isaaclab.devices import Se3Keyboard, Se3SpaceMouse


class _DroidGripperCommands:
    def advance(self) -> torch.Tensor:
        """Return the device's arm command with DROID's binary gripper control."""
        action = super().advance()
        if self.gripper_term:
            action[-1] = (action[-1] < 0).to(action.dtype)
        return action


class DroidSe3Keyboard(_DroidGripperCommands, Se3Keyboard):
    """Keyboard teleoperation with one to close and zero to open the DROID gripper."""


class DroidSe3SpaceMouse(_DroidGripperCommands, Se3SpaceMouse):
    """SpaceMouse teleoperation with one to close and zero to open the DROID gripper."""

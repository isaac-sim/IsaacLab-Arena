# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Task-specific actions for Gear Assembly."""

import torch
from collections.abc import Sequence

from isaaclab_arena.embodiments.droid.actions import BinaryJointPositionZeroToOneAction


class GearAssemblyBinaryJointPositionAction(BinaryJointPositionZeroToOneAction):
    """Apply gear-specific close widths with a Newton-safe target slew rate."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._gear_close_commands = torch.tensor(
            [env.cfg.hand_close_width[name] for name in ("gear_small", "gear_medium", "gear_large")],
            device=self.device,
        ).unsqueeze(-1) * torch.sign(self._close_command).unsqueeze(0)
        self._gear_open_commands = torch.tensor(
            [env.cfg.hand_grasp_width[name] for name in ("gear_small", "gear_medium", "gear_large")],
            device=self.device,
        ).unsqueeze(-1) * torch.sign(self._close_command).unsqueeze(0)
        self._gear_type_manager = env._gear_type_manager
        self._max_command_step = 0.5 * env.step_dt

    def process_actions(self, actions: torch.Tensor) -> None:
        previous_commands = self._processed_actions.clone()
        super().process_actions(actions)
        close_mask = actions if actions.dtype == torch.bool else actions > 0.5
        gear_indices = self._gear_type_manager.get_all_gear_type_indices()
        open_commands = self._gear_open_commands[gear_indices]
        close_commands = self._gear_close_commands[gear_indices]
        desired_commands = torch.where(close_mask, close_commands, open_commands)
        command_delta = torch.clamp(
            desired_commands - previous_commands,
            min=-self._max_command_step,
            max=self._max_command_step,
        )
        self._processed_actions = previous_commands + command_delta

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._processed_actions[env_ids] = self._asset.data.joint_pos.torch[env_ids][:, self._joint_ids]

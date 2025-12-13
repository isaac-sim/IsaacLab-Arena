# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv


def joint_acc(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint accelerations of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their accelerations returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_acc[:, asset_cfg.joint_ids]


def gripper_pos_by_joint_names(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    gripper_joint_names: list[str] = ["gripper_joint1", "gripper_joint2"],
) -> torch.Tensor:
    """
    Obtain the gripper position of the specified joint names.
    """
    robot: Articulation = env.scene[robot_cfg.name]

    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        # Handle multiple surface grippers by concatenating their states
        gripper_states = []
        for gripper_name, surface_gripper in env.scene.surface_grippers.items():
            gripper_states.append(surface_gripper.state.view(-1, 1))

        if len(gripper_states) == 1:
            return gripper_states[0]
        else:
            return torch.cat(gripper_states, dim=1)

    else:
        gripper_joint_ids, _ = robot.find_joints(gripper_joint_names)
        assert len(gripper_joint_ids) == 2, "Observation gripper_pos only support parallel gripper for now"
        finger_joint_1 = robot.data.joint_pos[:, gripper_joint_ids[0]].clone().unsqueeze(1)
        finger_joint_2 = -1 * robot.data.joint_pos[:, gripper_joint_ids[1]].clone().unsqueeze(1)
        return torch.cat((finger_joint_1, finger_joint_2), dim=1)

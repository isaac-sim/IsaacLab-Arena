# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Actions for the bimanual YAM cable-routing embodiment."""

from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils.configclass import configclass

from .config import ARM_JOINT_NAMES, GRIPPER_CLOSED_POSITION, GRIPPER_JOINT_NAME, GRIPPER_OPEN_POSITION


class FiniteJointPositionAction(JointPositionAction):
    """Keep absolute joint targets finite and inside the soft limits."""

    def process_actions(self, actions: torch.Tensor) -> None:
        """Sanitize and constrain absolute joint-position commands."""
        finite_actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
        super().process_actions(finite_actions)
        default = self._asset.data.default_joint_pos.torch[:, self._joint_ids]
        limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids]
        target = torch.where(torch.isfinite(self._processed_actions), self._processed_actions, default)
        self._processed_actions = torch.maximum(torch.minimum(target, limits[..., 1]), limits[..., 0])


class NormalizedFiniteJointPositionAction(FiniteJointPositionAction):
    """Map a finite command in ``[0, 1]`` through the joint transform."""

    def process_actions(self, actions: torch.Tensor) -> None:
        """Clamp the normalized command before applying its configured transform."""
        normalized_actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        super().process_actions(normalized_actions)


@configclass
class FiniteJointPositionActionCfg(JointPositionActionCfg):
    """Configure finite absolute joint-position actions."""

    class_type: type[FiniteJointPositionAction] = FiniteJointPositionAction


@configclass
class NormalizedFiniteJointPositionActionCfg(JointPositionActionCfg):
    """Configure normalized finite joint-position actions."""

    class_type: type[NormalizedFiniteJointPositionAction] = NormalizedFiniteJointPositionAction


def _arm_action(asset_name: str) -> FiniteJointPositionActionCfg:
    return FiniteJointPositionActionCfg(
        asset_name=asset_name,
        joint_names=ARM_JOINT_NAMES,
        preserve_order=True,
        use_default_offset=False,
    )


def _gripper_action(asset_name: str) -> NormalizedFiniteJointPositionActionCfg:
    return NormalizedFiniteJointPositionActionCfg(
        asset_name=asset_name,
        joint_names=[GRIPPER_JOINT_NAME],
        scale=GRIPPER_CLOSED_POSITION - GRIPPER_OPEN_POSITION,
        offset=GRIPPER_OPEN_POSITION,
        preserve_order=True,
        use_default_offset=False,
    )


@configclass
class BimanualYamActionsCfg:
    """Left arm/gripper followed by right arm/gripper in policy order."""

    left_arm_action: FiniteJointPositionActionCfg = _arm_action("left_robot")
    left_gripper_action: NormalizedFiniteJointPositionActionCfg = _gripper_action("left_robot")
    right_arm_action: FiniteJointPositionActionCfg = _arm_action("right_robot")
    right_gripper_action: NormalizedFiniteJointPositionActionCfg = _gripper_action("right_robot")

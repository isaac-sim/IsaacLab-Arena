# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Finite joint-position actions used by the YAM embodiment."""

from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils.configclass import configclass


class FiniteJointPositionAction(JointPositionAction):
    """Apply finite absolute joint targets constrained to articulation soft limits."""

    def process_actions(self, actions: torch.Tensor) -> None:
        """Sanitize and constrain absolute joint-position commands."""
        finite_actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
        super().process_actions(finite_actions)

        default = self._asset.data.default_joint_pos.torch[:, self._joint_ids]
        limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids]
        target = torch.where(torch.isfinite(self._processed_actions), self._processed_actions, default)
        self._processed_actions = torch.maximum(torch.minimum(target, limits[..., 1]), limits[..., 0])


class NormalizedFiniteJointPositionAction(FiniteJointPositionAction):
    """Map a finite command in [0, 1] through the configured affine joint transform."""

    def process_actions(self, actions: torch.Tensor) -> None:
        """Sanitize the normalized command before applying its joint transform."""
        normalized_actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        super().process_actions(normalized_actions)


@configclass
class FiniteJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for finite absolute joint-position actions."""

    class_type: type = FiniteJointPositionAction


@configclass
class NormalizedFiniteJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for finite normalized joint-position actions."""

    class_type: type = NormalizedFiniteJointPositionAction

# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""MDP2 terms of Arena embodiments: end-effector observations and joint-target resets."""

from __future__ import annotations

import torch

from isaaclab_experimental.mdp2 import Term

end_effector_pos_w = Term(
    torch=lambda body_link_pose_w, body_ids: body_link_pose_w[:, body_ids][:, 0, :3], shape=("worlds", 3)
)
"""Position of the body of ``body_ids`` (one body) in the world frame [m], like the DROID ``ee_pos`` observation."""

end_effector_quat_w = Term(
    torch=lambda body_link_pose_w, body_ids: body_link_pose_w[:, body_ids][:, 0, 3:7], shape=("worlds", 4)
)
"""Orientation ``(x, y, z, w)`` of the body of ``body_ids`` (one body) in the world frame, like ``ee_quat``."""


def _reset_joint_targets(mask: torch.Tensor, default_joint_pos: torch.Tensor, joint_pos_target: torch.Tensor) -> None:
    joint_pos_target.copy_(torch.where(mask[:, None], default_joint_pos, joint_pos_target))


reset_joint_targets = Term(torch=_reset_joint_targets)
"""Set the joint position targets of the reset worlds to the default joint positions, as
``reset_scene_to_default(reset_joint_targets=True)`` does. Bind ``joint_pos_target`` as an output."""

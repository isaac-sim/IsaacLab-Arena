# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Convert measured and desired link poses into relative IK action terms."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg


def pose_to_relative_ik_action(
    T_B_L: torch.Tensor,
    T_B_L_target: torch.Tensor,
    action_cfg: DifferentialInverseKinematicsActionCfg,
) -> torch.Tensor:
    """Convert absolute link targets into a raw relative IK action term.

    Args:
        T_B_L: Measured link L in articulation root B, shaped (N, 7), meters and XYZW.
        T_B_L_target: Desired pose of the same link and shape, before controller offsets.
        action_cfg: Relative pose IK term configuration, including scale and body offset.

    Returns:
        Tensor shaped (N, 6) on the measured pose's device. Applying the configured
        action scale and relative controller recovers the target controller-frame pose.
        Callers supply bounded references and assemble other action terms, such as grippers.
        This function does not plan trajectories, limit speed, or avoid collisions.
    """
    from isaaclab.utils.math import combine_frame_transforms, compute_pose_error

    cfg = action_cfg
    assert cfg.controller.command_type == "pose" and cfg.controller.use_relative_mode, "Expected relative pose IK"
    assert cfg.clip is None, "Clipped action terms are not supported"
    assert T_B_L.ndim == 2 and T_B_L.shape[1] == 7, "Expected batched measured poses shaped (N, 7)"
    target = T_B_L_target.to(device=T_B_L.device, dtype=T_B_L.dtype)
    assert target.shape == T_B_L.shape, "Expected one target pose per measured pose"
    for pose in (T_B_L, target):
        assert torch.isfinite(pose).all(), "Poses must be finite"
        quaternion_norm = torch.linalg.vector_norm(pose[:, 3:], dim=-1)
        assert torch.allclose(
            quaternion_norm, torch.ones_like(quaternion_norm), atol=1e-4
        ), "Expected unit XYZW quaternions"

    # C is the controller frame; its optional offset is fixed relative to link L.
    t_B_C, q_B_C = T_B_L[:, :3], T_B_L[:, 3:]
    t_B_C_target, q_B_C_target = target[:, :3], target[:, 3:]
    if cfg.body_offset is not None:
        t_L_C = T_B_L.new_tensor(cfg.body_offset.pos).expand(T_B_L.shape[0], -1)
        q_L_C = T_B_L.new_tensor(cfg.body_offset.rot).expand(T_B_L.shape[0], -1)
        t_B_C, q_B_C = combine_frame_transforms(t_B_C, q_B_C, t_L_C, q_L_C)
        t_B_C_target, q_B_C_target = combine_frame_transforms(t_B_C_target, q_B_C_target, t_L_C, q_L_C)
    translation, rotation = compute_pose_error(t_B_C, q_B_C, t_B_C_target, q_B_C_target, rot_error_type="axis_angle")
    scale = T_B_L.new_tensor(cfg.scale)
    assert scale.ndim == 0 or scale.shape == (6,), "IK scale must be a scalar or six components"
    assert torch.isfinite(scale).all() and (scale != 0).all(), "IK scale must be finite and nonzero"
    return torch.cat((translation, rotation), dim=-1) / scale

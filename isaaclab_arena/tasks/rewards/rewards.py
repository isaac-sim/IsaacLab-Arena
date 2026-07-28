# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

import warp as wp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

from isaaclab_arena.scene.object_geometry import object_geometry


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # End-effector position: (num_envs, 3)
    ee_w = wp.to_torch(ee_frame.data.target_pos_w)[..., 0, :]
    nearest_point_w = object_geometry(env, object_cfg.name).nearest_point_w(ee_w)
    object_ee_distance = torch.norm(nearest_point_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)

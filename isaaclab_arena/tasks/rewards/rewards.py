# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv


def object_ee_distance(
    env: IsaacLabArenaManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    object_position_w = env.arena_world.get_position_w(object_cfg.name)

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    end_effector_position_w = ee_frame.data.target_pos_w.torch[..., 0, :]
    distance_to_object = torch.norm(object_position_w - end_effector_position_w, dim=1)

    return 1 - torch.tanh(distance_to_object / std)

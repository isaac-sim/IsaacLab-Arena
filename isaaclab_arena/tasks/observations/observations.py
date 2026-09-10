# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv


def object_position_in_frame(
    env: IsaacLabArenaManagerBasedRLEnv,
    root_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return the object's position in the requested root frame."""
    arena_world = env.arena_world
    T_W_R = arena_world.get_pose_w(root_frame_cfg.name)
    object_position_w = arena_world.get_position_w(object_cfg.name)
    object_position_in_root_frame, _ = subtract_frame_transforms(T_W_R[:, :3], T_W_R[:, 3:], object_position_w)
    return object_position_in_root_frame

# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Generic observation terms for gear insertion tasks.

This module intentionally contains reusable task observations. The packed
OSC policy observation lives in the environment MDP package because it depends
on OSC action-term state.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import warp as wp
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_unique

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _resolve_body_id(robot: Articulation, body_name: str) -> int:
    """Return the single body id matching ``body_name``."""
    body_ids, body_names = robot.find_bodies([body_name])
    if len(body_ids) != 1:
        raise ValueError(f"Expected one body matching '{body_name}', found {len(body_ids)}: {body_names}.")
    return body_ids[0]


def body_pos_in_env_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "panda_fingertip_centered",
) -> torch.Tensor:
    """Return a robot body position in each environment frame.

    Isaac Lab body positions are world-frame values; subtracting environment
    origins makes the observation invariant to scene cloning offsets.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    body_id = _resolve_body_id(robot, body_name)
    return wp.to_torch(robot.data.body_pos_w)[:, body_id] - env.scene.env_origins


def body_quat_canonical(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "panda_fingertip_centered",
) -> torch.Tensor:
    """Return a robot body quaternion with a unique sign convention."""
    robot: Articulation = env.scene[robot_cfg.name]
    body_id = _resolve_body_id(robot, body_name)
    quat = wp.to_torch(robot.data.body_quat_w)[:, body_id]
    return quat_unique(quat)

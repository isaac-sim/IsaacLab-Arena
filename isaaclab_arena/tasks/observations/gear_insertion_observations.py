# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Observation terms for the NIST gear insertion task."""

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def gear_pos_in_env_frame(
    env: ManagerBasedRLEnv,
    gear_cfg: SceneEntityCfg = SceneEntityCfg("medium_nist_gear"),
) -> torch.Tensor:
    """Position of the held gear relative to the environment origin."""
    gear: RigidObject = env.scene[gear_cfg.name]
    return gear.data.root_pos_w - env.scene.env_origins


def gear_quat_canonical(
    env: ManagerBasedRLEnv,
    gear_cfg: SceneEntityCfg = SceneEntityCfg("medium_nist_gear"),
) -> torch.Tensor:
    """Orientation of the held gear, canonicalized so w >= 0."""
    gear: RigidObject = env.scene[gear_cfg.name]
    quat = gear.data.root_quat_w
    sign = torch.where(quat[:, 0:1] < 0, -1.0, 1.0)
    return quat * sign


def board_pos_in_env_frame(
    env: ManagerBasedRLEnv,
    board_cfg: SceneEntityCfg = SceneEntityCfg("gears_and_base"),
) -> torch.Tensor:
    """Position of the board/base relative to the environment origin."""
    board: RigidObject = env.scene[board_cfg.name]
    return board.data.root_pos_w - env.scene.env_origins


def peg_pos_in_env_frame(
    env: ManagerBasedRLEnv,
    board_cfg: SceneEntityCfg = SceneEntityCfg("gears_and_base"),
    peg_offset: list[float] = [0.0, 0.0, 0.0],
) -> torch.Tensor:
    """Target peg position: fixed asset pose + offset in its local frame."""
    board: RigidObject = env.scene[board_cfg.name]
    pos = board.data.root_pos_w - env.scene.env_origins
    quat = board.data.root_quat_w
    offset = torch.tensor(peg_offset, device=env.device, dtype=torch.float32).unsqueeze(0).expand(env.num_envs, 3)
    return pos + math_utils.quat_apply(quat, offset)


def board_quat_canonical(
    env: ManagerBasedRLEnv,
    board_cfg: SceneEntityCfg = SceneEntityCfg("gears_and_base"),
) -> torch.Tensor:
    """Orientation of the assembled board / peg, canonicalized so w >= 0."""
    board: RigidObject = env.scene[board_cfg.name]
    quat = board.data.root_quat_w
    sign = torch.where(quat[:, 0:1] < 0, -1.0, 1.0)
    return quat * sign


def held_gear_base_pos_in_env_frame(
    env: ManagerBasedRLEnv,
    gear_cfg: SceneEntityCfg = SceneEntityCfg("medium_nist_gear"),
    held_gear_base_offset: list[float] = [2.025e-2, 0.0, 0.0],
) -> torch.Tensor:
    """Position of the held gear's insertion point (root + offset in gear frame) in env frame."""
    gear: RigidObject = env.scene[gear_cfg.name]
    gear_pos = gear.data.root_pos_w - env.scene.env_origins
    gear_quat = gear.data.root_quat_w
    held_off = torch.tensor(
        held_gear_base_offset, device=env.device, dtype=torch.float32
    ).unsqueeze(0).expand(env.num_envs, 3)
    return gear_pos + math_utils.quat_apply(gear_quat, held_off)


def peg_delta_from_held_gear_base(
    env: ManagerBasedRLEnv,
    gear_cfg: SceneEntityCfg = SceneEntityCfg("medium_nist_gear"),
    board_cfg: SceneEntityCfg = SceneEntityCfg("gears_and_base"),
    peg_offset: list[float] = [0.0, 0.0, 0.0],
    held_gear_base_offset: list[float] = [2.025e-2, 0.0, 0.0],
) -> torch.Tensor:
    """Vector from held gear insertion point to peg (peg_pos - held_gear_base_pos). Positive = peg is ahead in that axis."""
    held_base = held_gear_base_pos_in_env_frame(env, gear_cfg, held_gear_base_offset)
    peg_pos = peg_pos_in_env_frame(env, board_cfg, peg_offset)
    return peg_pos - held_base

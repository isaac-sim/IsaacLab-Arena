# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared geometry utilities for gear insertion.

The frame math follows the Isaac Lab Factory/Forge insertion tasks: represent
the peg and held-gear insertion points as local offsets on their owning assets,
then compare those points in each environment frame.
"""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
import warp as wp
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def resolve_offset_tensor(
    values: list[float] | tuple[float, ...] | None,
    cached_values: tuple[float, ...],
    cached_tensor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Return an offset tensor on ``device``.

    Reward and termination terms cache their default offsets at construction.
    Manager call-time parameters may still override those values, so this function
    keeps the common "use cached tensor unless explicitly overridden" path in
    one place.
    """
    if values is None or tuple(values) == cached_values:
        return cached_tensor
    return torch.tensor(values, device=device, dtype=torch.float32)


def compute_asset_local_offset_pose(
    env: ManagerBasedRLEnv,
    asset: RigidObject,
    offset: tuple[float, ...] | torch.Tensor,
    num_envs: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an asset-local offset pose in each environment frame.

    Isaac Lab stores rigid-object root poses in world coordinates. Gear
    insertion rewards and terminations compare geometry in the per-environment
    frame, so this function subtracts ``env.scene.env_origins`` before rotating
    the local offset into the asset root frame.
    """
    num_envs = env.num_envs if num_envs is None else num_envs
    root_pos = wp.to_torch(asset.data.root_pos_w)[:num_envs] - env.scene.env_origins[:num_envs]
    root_quat = wp.to_torch(asset.data.root_quat_w)[:num_envs]
    offset_tensor = torch.as_tensor(offset, device=env.device, dtype=torch.float32)
    offset_tensor = offset_tensor.unsqueeze(0).expand(num_envs, 3)
    return root_pos + math_utils.quat_apply(root_quat, offset_tensor), root_quat


def compute_asset_local_offset_pos(
    env: ManagerBasedRLEnv,
    asset: RigidObject,
    offset: tuple[float, ...] | torch.Tensor,
) -> torch.Tensor:
    """Return an asset-local offset position in each environment frame."""
    pos, _ = compute_asset_local_offset_pose(env, asset, offset)
    return pos


def peg_pos_in_env_frame(
    env: ManagerBasedRLEnv,
    board_cfg: SceneEntityCfg,
    peg_offset: tuple[float, ...] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Return the target peg position in each environment frame.

    ``peg_offset`` is measured in the fixed board or gear-base asset frame.
    """
    board: RigidObject = env.scene[board_cfg.name]
    return compute_asset_local_offset_pos(env, board, peg_offset)


def held_gear_base_pos_in_env_frame(
    env: ManagerBasedRLEnv,
    gear_cfg: SceneEntityCfg,
    held_gear_base_offset: tuple[float, ...] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Return the held gear insertion point in each environment frame.

    The held gear's root is not necessarily the point that should align with
    the peg, so insertion logic compares this offset point against the peg.
    """
    gear: RigidObject = env.scene[gear_cfg.name]
    return compute_asset_local_offset_pos(env, gear, held_gear_base_offset)


def peg_delta_from_held_gear_base(
    env: ManagerBasedRLEnv,
    gear_cfg: SceneEntityCfg,
    board_cfg: SceneEntityCfg,
    peg_offset: tuple[float, ...] = (0.0, 0.0, 0.0),
    held_gear_base_offset: tuple[float, ...] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Return the vector from the held gear insertion point to the peg."""
    held_base = held_gear_base_pos_in_env_frame(env, gear_cfg, held_gear_base_offset)
    peg_pos = peg_pos_in_env_frame(env, board_cfg, peg_offset)
    return peg_pos - held_base


def check_gear_insertion_geometry(
    held_base_pos: torch.Tensor,
    peg_pos: torch.Tensor,
    gear_peg_height: float,
    z_fraction: float,
    xy_threshold: float,
) -> torch.Tensor:
    """Return whether the gear is centered and inserted on the peg.

    Success is geometric: the insertion point must be within the XY tolerance
    and below the configured fraction of the peg height. Lower Z values mean the
    gear has moved farther down the peg.
    """
    xy_dist = torch.norm(held_base_pos[:, :2] - peg_pos[:, :2], dim=-1)
    z_diff = held_base_pos[:, 2] - peg_pos[:, 2]
    return (xy_dist < xy_threshold) & (z_diff < gear_peg_height * z_fraction)


def compute_gear_insertion_success(
    env: ManagerBasedRLEnv,
    gear_cfg: SceneEntityCfg,
    board_cfg: SceneEntityCfg,
    peg_offset: tuple[float, ...] | torch.Tensor,
    held_gear_base_offset: tuple[float, ...] | torch.Tensor,
    gear_peg_height: float,
    z_fraction: float,
    xy_threshold: float,
) -> torch.Tensor:
    """Return whether the held gear meets the insertion success geometry.

    This is the shared implementation used by rewards, terminations, and
    OSC-specific auxiliary losses so they agree on the success label.
    """
    gear: RigidObject = env.scene[gear_cfg.name]
    board: RigidObject = env.scene[board_cfg.name]
    held_base_pos, _ = compute_asset_local_offset_pose(env, gear, held_gear_base_offset)
    peg_pos, _ = compute_asset_local_offset_pose(env, board, peg_offset)
    return check_gear_insertion_geometry(held_base_pos, peg_pos, gear_peg_height, z_fraction, xy_threshold)

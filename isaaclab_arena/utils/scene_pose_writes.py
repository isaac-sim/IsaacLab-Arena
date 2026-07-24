# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for writing scene root poses into the sim."""

from __future__ import annotations

import torch
from typing import Any


def write_scene_root_poses_to_sim(
    scene_asset: Any,
    scene_name: str,
    pose_tensor: torch.Tensor,
    env_ids: torch.Tensor,
    device: torch.device,
) -> None:
    """Write world-frame root poses for articulations, rigid bodies, or AssetBase extras.

    Args:
        scene_asset: Scene entity to write (articulation, rigid object, or XForm prim view).
        scene_name: Isaac Lab scene key, used only for error messages.
        pose_tensor: ``(N, 7)`` poses in world frame with env origins already applied.
        env_ids: Environment indices being written.
        device: Torch device for zero-velocity tensors on the articulation path.
    """
    num_envs = pose_tensor.shape[0]
    zero_velocity = torch.zeros(num_envs, 6, device=device)

    write_root_pose = getattr(scene_asset, "write_root_pose_to_sim", None)
    if write_root_pose is not None:
        write_root_pose(pose_tensor, env_ids=env_ids)
        scene_asset.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)
        return

    set_world_poses = getattr(scene_asset, "set_world_poses", None)
    if set_world_poses is not None:
        set_world_poses(
            positions=pose_tensor[:, :3],
            orientations=pose_tensor[:, 3:7],
            indices=env_ids.detach().cpu(),
        )
        return

    assert False, f"Scene asset '{scene_name}' does not support root pose writes"

# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_arena.utils.pose import Pose


@configclass
class PlacementEventsCfg:
    """Reset event: set object pose per env from layout dicts (relation placement)."""

    set_object_pose_per_env_from_layouts: EventTermCfg = dataclasses.MISSING  # type: ignore[assignment]


def make_placement_event_cfg(
    positions_all_envs_by_name: list[dict[str, tuple[float, float, float]]],
    object_names: list[str],
    anchor_names: list[str] | None = None,
    placement_valid_per_env: list[bool] | None = None,
) -> PlacementEventsCfg:
    """Build event cfg for applying placement layouts at reset (one layout per env).

    Anchor names are applied first so supports (e.g. table) are set before objects on them.
    """
    params: dict = {
        "positions_all_envs_by_name": positions_all_envs_by_name,
        "object_names": object_names,
        "anchor_names": anchor_names or [],
    }
    if placement_valid_per_env is not None:
        params["placement_valid_per_env"] = placement_valid_per_env
    return PlacementEventsCfg(
        set_object_pose_per_env_from_layouts=EventTermCfg(
            func=set_object_pose_per_env_from_layouts,
            mode="reset",
            params=params,
        )
    )


def _resolve_env_ids(env: ManagerBasedEnv, env_ids) -> list[int] | None:
    """Convert env_ids to a list of int indices for use in reset events.

    Event manager may pass env_ids as slice(None) when reset applies to all envs,
    or as a tensor/sequence when only a subset of envs reset. Normalize to list[int].
    """
    if env_ids is None:
        return None
    if isinstance(env_ids, slice):
        if env_ids == slice(None):
            return list(range(env.num_envs))
        start, stop, step = env_ids.indices(env.num_envs)
        return list(range(start, stop, step))
    if hasattr(env_ids, "tolist"):
        return env_ids.tolist()
    return list(env_ids)


def set_object_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose: Pose,
) -> None:
    if env_ids is None:
        return
    # Grab the object
    asset = env.scene[asset_cfg.name]
    num_envs = len(env_ids)
    # Convert the pose to the env frame
    pose_t_xyz_q_wxyz = pose.to_tensor(device=env.device).repeat(num_envs, 1)
    pose_t_xyz_q_wxyz[:, :3] += env.scene.env_origins[env_ids]
    # Set the pose and velocity
    asset.write_root_pose_to_sim(pose_t_xyz_q_wxyz, env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device), env_ids=env_ids)


def set_object_pose_per_env(
    env: ManagerBasedEnv,
    env_ids,
    asset_cfg: SceneEntityCfg,
    pose_list: list[Pose],
) -> None:
    """Set root pose per env from pose_list. Uses one batched write per asset (no per-env loop)."""
    if env_ids is None:
        return

    resolved = _resolve_env_ids(env, env_ids)
    if not resolved:
        return

    asset = env.scene[asset_cfg.name]
    if not hasattr(asset, "write_root_pose_to_sim"):
        return

    valid_envs = [e for e in resolved if e < len(pose_list)]
    if not valid_envs:
        return

    device = env.device
    dtype = torch.float32
    poses_xyz_q = []
    for e in valid_envs:
        t = pose_list[e].to_tensor(device=device)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        poses_xyz_q.append(t)
    batched = torch.cat(poses_xyz_q, dim=0).to(device=device, dtype=dtype)
    # Add env origins so positions are in simulation world frame (same as default reset).
    origins = env.scene.env_origins[valid_envs, :].clone().to(device=device, dtype=dtype)
    batched[:, :3] += origins

    # Set full root state (pose + zero velocity) in one call so order matches reset_scene_to_default
    # and no intermediate state can cause one env to blow. Shape (N, 13) = (N, 7) pose + (N, 6) vel.
    env_ids_t = torch.tensor(valid_envs, device=device, dtype=torch.int64)
    zero_vel = torch.zeros(len(valid_envs), 6, device=device, dtype=dtype)
    root_state = torch.cat([batched, zero_vel], dim=-1)
    asset.write_root_state_to_sim(root_state, env_ids=env_ids_t)


def set_object_pose_per_env_from_layouts(
    env: ManagerBasedEnv,
    env_ids,
    positions_all_envs_by_name: list[dict[str, tuple[float, float, float]]],
    object_names: list[str],
    anchor_names: list[str] | None = None,
    placement_valid_per_env: list[bool] | None = None,
) -> None:
    """Set object pose per env for each object from layout dicts; calls set_object_pose_per_env per object.

    Applies anchors first (e.g. table), then non-anchors.
    """
    resolved = _resolve_env_ids(env, env_ids)
    if not resolved:
        return
    anchor_set = set(anchor_names or [])
    ordered_names = [n for n in object_names if n in anchor_set]
    ordered_names += [n for n in object_names if n not in anchor_set]
    identity_wxyz = (1.0, 0.0, 0.0, 0.0)
    for name in ordered_names:
        if name not in env.scene.keys():
            continue
        asset = env.scene[name]
        if not hasattr(asset, "write_root_pose_to_sim"):
            continue
        pose_list = []
        for e in range(len(positions_all_envs_by_name)):
            xyz = positions_all_envs_by_name[e].get(name) if e < len(positions_all_envs_by_name) else None
            if xyz is not None:
                x, y, z = xyz
                pose_list.append(Pose(position_xyz=(x, y, z), rotation_wxyz=identity_wxyz))
            else:
                pose_list.append(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_wxyz=identity_wxyz))
        set_object_pose_per_env(env, resolved, SceneEntityCfg(name), pose_list)

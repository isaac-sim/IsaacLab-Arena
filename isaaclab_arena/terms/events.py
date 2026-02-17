# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.usd_helpers import get_asset_usd_path_from_prim_path


def _resolve_object_name_for_usd_path(usd_path: str, objects: list) -> str:
    """Resolve a referenced USD path to an object name. Raises if no object matches."""
    ref_basename = os.path.basename(usd_path)
    for obj in objects:
        obj_name = getattr(obj, "name", None)
        obj_usd = getattr(obj, "usd_path", None)
        if obj_name is None:
            continue
        if obj_usd and ref_basename == os.path.basename(obj_usd):
            return obj_name
        if ref_basename == obj_name + ".usd" or ref_basename.startswith(obj_name + "_"):
            return obj_name
    raise AssertionError(f"No object name for USD path {usd_path!r} (basename {ref_basename!r})")


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
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_list: list[Pose],
) -> None:
    if env_ids is None:
        return

    # Grab the object
    asset = env.scene[asset_cfg.name]

    # Set the objects pose in each environment independently
    assert env_ids.ndim == 1
    for cur_env in env_ids.tolist():
        # Convert the pose to the env frame
        pose = pose_list[cur_env]
        pose_t_xyz_q_wxyz = pose.to_tensor(device=env.device)
        pose_t_xyz_q_wxyz[:3] += env.scene.env_origins[cur_env, :].squeeze()
        # Set the pose and velocity
        asset.write_root_pose_to_sim(pose_t_xyz_q_wxyz, env_ids=torch.tensor([cur_env], device=env.device))
        asset.write_root_velocity_to_sim(
            torch.zeros(1, 6, device=env.device), env_ids=torch.tensor([cur_env], device=env.device)
        )


def set_object_set_pose_by_usd(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_by_object: dict[str, Pose],
    objects: list,
) -> None:
    """Reset object_set instances to poses by object name. Prim must have a USD reference; object name must be in pose_by_object."""
    if env_ids is None or env_ids.numel() == 0:
        return
    asset = env.scene[asset_cfg.name]
    stage = env.scene.stage
    matching_prims = sim_utils.find_matching_prims(asset.cfg.prim_path, stage)
    poses_list = []
    for i in range(env_ids.numel()):
        env_id = int(env_ids[i].item())
        prim = matching_prims[env_id]
        prim_path = prim.GetPath().pathString
        usd_path = get_asset_usd_path_from_prim_path(prim_path, stage)
        assert usd_path is not None, f"Prim at {prim_path} has no USD reference."
        object_name = _resolve_object_name_for_usd_path(usd_path, objects)
        assert object_name in pose_by_object, (
            f"Object name {object_name!r} not in pose_by_object (keys: {list(pose_by_object.keys())})."
        )
        pose = pose_by_object[object_name]
        pose_t = pose.to_tensor(device=env.device).unsqueeze(0)
        pose_t[:, :3] += env.scene.env_origins[env_id]
        poses_list.append(pose_t)
    pose_t_xyz_q_wxyz = torch.cat(poses_list, dim=0)
    asset.write_root_pose_to_sim(pose_t_xyz_q_wxyz, env_ids=env_ids)
    asset.write_root_velocity_to_sim(
        torch.zeros(pose_t_xyz_q_wxyz.shape[0], 6, device=env.device), env_ids=env_ids
    )

# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

import warp as wp
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.velocity import Velocity


def _deformable_nodal_state_for_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose: Pose,
    velocity: Velocity | None = None,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    nodal_state = asset.data.default_nodal_state_w.torch[env_ids].clone()
    target_pos_w = torch.tensor(pose.position_xyz, device=env.device).repeat(len(env_ids), 1)
    target_pos_w += env.scene.env_origins[env_ids]
    centroid_w = nodal_state[..., :3].mean(dim=1)
    nodal_state[..., :3] += (target_pos_w - centroid_w).unsqueeze(1)

    if velocity is not None:
        nodal_state[..., 3:] = torch.tensor(velocity.linear_xyz, device=env.device)
    else:
        nodal_state[..., 3:] = 0.0
    return nodal_state


def set_deformable_object_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose: Pose,
    velocity: Velocity | None = None,
) -> None:
    """Reset a deformable object's nodal state so its centroid is at ``pose``."""
    if env_ids is None:
        return
    asset = env.scene[asset_cfg.name]
    nodal_state = _deformable_nodal_state_for_pose(env, env_ids, asset_cfg, pose, velocity)
    asset.write_nodal_state_to_sim_index(nodal_state, env_ids=env_ids)
    asset.reset(env_ids=env_ids)


def set_deformable_object_pose_per_env(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_list: list[Pose],
) -> None:
    """Reset a deformable object's nodal state from a per-environment pose list."""
    if env_ids is None:
        return
    asset = env.scene[asset_cfg.name]
    assert env_ids.ndim == 1
    for cur_env in env_ids.tolist():
        cur_env_ids = torch.tensor([cur_env], device=env.device)
        nodal_state = _deformable_nodal_state_for_pose(env, cur_env_ids, asset_cfg, pose_list[cur_env])
        asset.write_nodal_state_to_sim_index(nodal_state, env_ids=cur_env_ids)
    asset.reset(env_ids=env_ids)


def set_object_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose: Pose,
    velocity: Velocity | None = None,
) -> None:
    if env_ids is None:
        return
    # Grab the object
    asset = env.scene[asset_cfg.name]
    num_envs = len(env_ids)
    # Convert the pose to the env frame
    pose_t_xyz_q_xyzw = pose.to_tensor(device=env.device).repeat(num_envs, 1)
    pose_t_xyz_q_xyzw[:, :3] += env.scene.env_origins[env_ids]
    # Set the pose and velocity
    asset.write_root_pose_to_sim(pose_t_xyz_q_xyzw, env_ids=env_ids)
    if velocity is not None:
        vel = velocity.to_tensor(device=env.device).unsqueeze(0).expand(num_envs, -1)
        asset.write_root_velocity_to_sim(vel, env_ids=env_ids)
    else:
        asset.write_root_velocity_to_sim(torch.zeros(num_envs, 6, device=env.device), env_ids=env_ids)


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
        pose_t_xyz_q_xyzw = pose.to_tensor(device=env.device).unsqueeze(0)
        pose_t_xyz_q_xyzw[0, :3] += env.scene.env_origins[cur_env, :]
        # Set the pose and velocity
        asset.write_root_pose_to_sim(pose_t_xyz_q_xyzw, env_ids=torch.tensor([cur_env], device=env.device))
        asset.write_root_velocity_to_sim(
            torch.zeros(1, 6, device=env.device), env_ids=torch.tensor([cur_env], device=env.device)
        )


def reset_all_articulation_joints(env: ManagerBasedEnv, env_ids: torch.Tensor):
    """Reset the articulation joints to the initial state."""
    for articulation_asset in env.scene.articulations.values():
        # obtain default and deal with the offset for env origins
        default_root_state = wp.to_torch(articulation_asset.data.default_root_state)[env_ids].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
        # set into the physics simulation
        articulation_asset.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        articulation_asset.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)
        # obtain default joint positions
        default_joint_pos = wp.to_torch(articulation_asset.data.default_joint_pos)[env_ids].clone()
        default_joint_vel = wp.to_torch(articulation_asset.data.default_joint_vel)[env_ids].clone()
        # set into the physics simulation
        articulation_asset.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)

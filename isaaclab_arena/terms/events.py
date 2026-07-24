# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

import warp as wp
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.scene_pose_writes import write_scene_root_poses_to_sim
from isaaclab_arena.utils.velocity import Velocity


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


def _write_scene_pose(env: ManagerBasedEnv, scene_name: str, pose: Pose, env_ids: torch.Tensor) -> None:
    """Write one env-local ``pose`` to a scene entity's root across ``env_ids`` (env origins added)."""
    asset = env.scene[scene_name]
    num_envs = len(env_ids)
    pose_t_xyz_q_xyzw = pose.to_tensor(device=env.device).repeat(num_envs, 1)
    pose_t_xyz_q_xyzw[:, :3] += env.scene.env_origins[env_ids]
    write_scene_root_poses_to_sim(asset, scene_name, pose_t_xyz_q_xyzw, env_ids, env.device)


def reset_placement_asset_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    scene_writes: list[tuple[str, Pose]],
) -> None:
    """Restore a placement asset to fixed env-local poses on reset.

    Each ``(scene entity name, pose)`` in ``scene_writes`` is written to every resetting env,
    letting a compound asset place several prims (e.g. a robot and its stand) together.
    """
    if env_ids is None:
        return
    for scene_name, pose in scene_writes:
        _write_scene_pose(env, scene_name, pose, env_ids)


def reset_placement_asset_pose_per_env(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    write_pose_list: list[list[tuple[str, Pose]]],
) -> None:
    """Restore a placement asset to a distinct per-env pose on reset.

    ``write_pose_list[env]`` holds that env's ``(scene entity name, pose)`` writes, so different
    environments can hold different solved layouts for the same asset (and its auxiliary prims).
    """
    if env_ids is None:
        return
    assert env_ids.ndim == 1, "env_ids must be a 1-D tensor of environment indices"
    num_scene_envs = env.scene.env_origins.shape[0]
    assert len(write_pose_list) == num_scene_envs, (
        f"per-env pose writes cover {len(write_pose_list)} envs, but the scene has {num_scene_envs}; "
        "write_pose_list is indexed by absolute env id and must span every environment."
    )
    for cur_env in env_ids.tolist():
        single_env = torch.tensor([cur_env], device=env.device)
        for scene_name, pose in write_pose_list[cur_env]:
            _write_scene_pose(env, scene_name, pose, single_env)


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

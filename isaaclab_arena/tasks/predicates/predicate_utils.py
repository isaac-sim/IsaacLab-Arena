# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING

import warp as wp
from isaaclab.assets import RigidObject
from isaaclab.utils.math import combine_frame_transforms

from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.utils.pose import Pose

if TYPE_CHECKING:
    from isaaclab_arena.relations.placement_asset import PlaceableAsset


@dataclass(frozen=True, slots=True, repr=False)
class ArenaAssetHandle:
    """Keep an Arena asset by identity when manager configurations are copied."""

    asset: PlaceableAsset

    def __deepcopy__(self, memo: dict[int, object]) -> ArenaAssetHandle:
        memo[id(self)] = self
        return self

    def __repr__(self) -> str:
        return f"{type(self).__name__}(asset={self.asset.name!r})"


def get_env(env):
    """Resolve to the unwrapped manager-based env regardless of wrapper depth."""
    seen = set()
    while hasattr(env, "unwrapped") and env.unwrapped is not env and id(env) not in seen:
        seen.add(id(env))
        env = env.unwrapped
    return env


def get_rigid_object(env, name: str) -> RigidObject:
    """Get a rigid object from the env's scene."""
    return get_env(env).scene[name]


def get_root_pos_w(env, name: str) -> torch.Tensor:
    """Get the root position of a rigid object in the world frame."""
    return wp.to_torch(get_rigid_object(env, name).data.root_pos_w)


def get_root_lin_vel_w(env, name: str) -> torch.Tensor:
    """Get the root linear velocity of a rigid object in the world frame."""
    return wp.to_torch(get_rigid_object(env, name).data.root_lin_vel_w)


def get_root_ang_vel_w(env, name: str) -> torch.Tensor:
    """Get the root angular velocity of a rigid object in the world frame."""
    return wp.to_torch(get_rigid_object(env, name).data.root_ang_vel_w)


def get_asset_pose_w(env, asset: PlaceableAsset) -> torch.Tensor:
    """Get an asset's world pose.

    Unregistered object references use their authored pose relative to the parent asset.
    """

    unwrapped_env = get_env(env)
    scene_key = asset.get_scene_key()
    if scene_key in unwrapped_env.scene.keys():
        if getattr(asset, "object_type", None) == ObjectType.BASE:
            initial_pose = asset.get_initial_pose()
            if initial_pose is None:
                initial_pose = Pose.identity()
            if isinstance(initial_pose, Pose):
                asset_pose_w = (
                    initial_pose.to_tensor(device=unwrapped_env.device).expand(unwrapped_env.num_envs, 7).clone()
                )
                asset_pose_w[:, :3] += unwrapped_env.scene.env_origins
                return asset_pose_w

        asset_pose_w = asset.get_object_pose(unwrapped_env, is_relative=False)
        assert asset_pose_w.shape[0] in (
            1,
            unwrapped_env.num_envs,
        ), f"Asset '{asset.name}' returned {asset_pose_w.shape[0]} poses for {unwrapped_env.num_envs} environments."
        if asset_pose_w.shape[0] == unwrapped_env.num_envs:
            return asset_pose_w

        asset_pose_w = asset_pose_w.expand(unwrapped_env.num_envs, 7).clone()
        asset_pose_w[:, :3] += unwrapped_env.scene.env_origins - unwrapped_env.scene.env_origins[:1]
        return asset_pose_w

    parent_asset = getattr(asset, "parent_asset", None)
    pose_relative_to_parent = getattr(asset, "initial_pose_relative_to_parent", None)
    assert (
        parent_asset is not None and pose_relative_to_parent is not None
    ), f"Asset '{asset.name}' is not registered in the scene and has no parent-relative pose."

    parent_pose_w = get_asset_pose_w(unwrapped_env, parent_asset)
    relative_pose = pose_relative_to_parent.to_tensor(device=unwrapped_env.device).expand(unwrapped_env.num_envs, 7)
    position_w, quaternion_w = combine_frame_transforms(
        parent_pose_w[:, :3],
        parent_pose_w[:, 3:],
        relative_pose[:, :3],
        relative_pose[:, 3:],
    )
    return torch.cat((position_w, quaternion_w), dim=-1)


def select(result: torch.Tensor, env_id: int | None) -> torch.Tensor:
    """Return the entry at ``env_id`` if requested, otherwise the full vector."""
    if env_id is None:
        return result
    return result[env_id]

# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pose conversion shared by live and recorded relation placement."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab_arena.relations.relations import RotateAroundSolution, get_anchor_objects
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.yaw import rotate_quat_by_yaw, yaw_from_quat_xyzw

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.relations.placement_result import PlacementResult


IDENTITY_ROTATION_XYZW = (0.0, 0.0, 0.0, 1.0)


def get_pose_from_layout(asset: PlaceableAsset, layout: PlacementResult) -> Pose:
    """Return an asset pose from a solved layout."""
    assert asset in layout.positions, f"Placement layout is missing non-anchor asset '{asset.name}'"
    rotate_marker = next((r for r in asset.get_relations() if isinstance(r, RotateAroundSolution)), None)
    base_rotation = rotate_marker.get_rotation_xyzw() if rotate_marker else IDENTITY_ROTATION_XYZW
    marker_yaw = yaw_from_quat_xyzw(base_rotation)
    total_yaw = layout.orientations.get(asset, marker_yaw)
    rotation = rotate_quat_by_yaw(base_rotation, total_yaw - marker_yaw)
    return Pose(position_xyz=layout.positions[asset], rotation_xyzw=rotation)


def get_scene_root_poses_from_layout(
    assets: list[PlaceableAsset],
    result: PlacementResult,
    anchor_assets: set[PlaceableAsset] | None = None,
) -> dict[PlaceableAsset, dict[str, Pose]]:
    """Expand one solved layout into environment-local scene-root poses grouped by asset."""
    anchor_assets = set(get_anchor_objects(assets)) if anchor_assets is None else anchor_assets
    poses_by_asset: dict[PlaceableAsset, dict[str, Pose]] = {}
    scene_keys: set[str] = set()
    for asset in assets:
        if asset in anchor_assets:
            continue
        root_poses = dict(asset.layout_pose_to_scene_writes(get_pose_from_layout(asset, result)))
        assert set(root_poses) == set(
            asset.get_scene_root_keys()
        ), f"Asset '{asset.name}' must provide every owned scene root"
        duplicate_keys = scene_keys.intersection(root_poses)
        assert not duplicate_keys, f"Duplicate relation-placement scene roots: {sorted(duplicate_keys)}"
        scene_keys.update(root_poses)
        poses_by_asset[asset] = root_poses
    return poses_by_asset


def write_scene_poses_to_sim(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    poses: dict[str, torch.Tensor],
) -> None:
    """Apply environment-local root poses and zero velocities."""
    for name, pose in poses.items():
        assert pose.shape == (len(env_ids), 7), f"Root poses for '{name}' must have shape (N, 7)"
    env_origins = env.scene.env_origins[env_ids]
    zero_velocity = torch.zeros((len(env_ids), 6), device=env.device)
    for name, T_E_O in poses.items():
        T_W_O = T_E_O.clone()
        T_W_O[:, :3] += env_origins
        scene_asset = env.scene[name]
        scene_asset.write_root_pose_to_sim(T_W_O, env_ids=env_ids)
        scene_asset.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)

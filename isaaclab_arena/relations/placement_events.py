# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

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


def get_rotation_xyzw(asset: PlaceableAsset) -> tuple[float, float, float, float]:
    """Return the RotateAroundSolution rotation for an asset, or identity if none."""
    rotate_marker = next((r for r in asset.get_relations() if isinstance(r, RotateAroundSolution)), None)
    return rotate_marker.get_rotation_xyzw() if rotate_marker else IDENTITY_ROTATION_XYZW


def get_base_rotation_per_asset(
    assets: list[PlaceableAsset],
) -> dict[PlaceableAsset, tuple[float, float, float, float]]:
    """Return the base rotation for each asset."""
    return {asset: get_rotation_xyzw(asset) for asset in assets}


def get_pose_from_layout(asset: PlaceableAsset, layout: PlacementResult) -> Pose:
    """Return an asset pose from a solved layout."""
    assert asset in layout.positions, f"Placement layout is missing non-anchor asset '{asset.name}'"
    base_rotation = get_rotation_xyzw(asset)
    marker_yaw = yaw_from_quat_xyzw(base_rotation)
    total_yaw = layout.orientations.get(asset, marker_yaw)
    rotation = rotate_quat_by_yaw(base_rotation, total_yaw - marker_yaw)
    return Pose(position_xyz=layout.positions[asset], rotation_xyzw=rotation)


def get_movable_asset_names(
    assets: list[PlaceableAsset],
    anchor_assets: set[PlaceableAsset],
) -> list[str]:
    """Return scene names for non-anchor placement assets."""
    return [asset.get_scene_key() for asset in assets if asset not in anchor_assets]


def validate_scene_poses(poses: dict[str, torch.Tensor]) -> None:
    """Require finite xyz/xyzw pose tensors of shape (N, 7) with unit quaternions."""
    for name, pose in poses.items():
        assert pose.ndim == 2 and pose.shape[1] == 7, f"Root poses for '{name}' must have shape (N, 7)"
        assert torch.isfinite(pose).all(), f"Root poses for '{name}' must be finite"
        assert torch.allclose(
            pose[:, 3:].square().sum(dim=-1), torch.ones_like(pose[:, 0]), atol=1e-4, rtol=0
        ), f"Root poses for '{name}' require unit quaternions"


def write_scene_poses_to_sim(env: ManagerBasedEnv, env_ids: torch.Tensor, poses: dict[str, torch.Tensor]) -> None:
    """Apply environment-local root poses and zero velocities for the selected environments.

    Frames E, W, and O denote the environment, simulation world, and object.

    Args:
        env: Constructed simulation environment.
        env_ids: Absolute indices of the N resetting environments, shape (N,).
        poses: Scene entity names mapped to xyz/xyzw tensors, each shaped (N, 7).
            Compound asset poses must first be expanded with layout_pose_to_scene_writes().
            Call validate_scene_poses() before writing unvalidated external poses.
    """
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
        layout_pose = get_pose_from_layout(asset, result)
        scene_writes = asset.layout_pose_to_scene_writes(layout_pose)
        root_poses = dict(scene_writes)
        assert len(root_poses) == len(scene_writes), f"Asset '{asset.name}' returned duplicate scene roots"
        assert set(root_poses) == set(
            asset.get_scene_root_keys()
        ), f"Asset '{asset.name}' must provide every owned scene root"
        duplicate_keys = scene_keys.intersection(root_poses)
        assert not duplicate_keys, f"Duplicate relation-placement scene roots: {sorted(duplicate_keys)}"
        scene_keys.update(root_poses)
        poses_by_asset[asset] = root_poses
    return poses_by_asset


def write_placement_result_to_sim(
    env: ManagerBasedEnv,
    env_id: int,
    result: PlacementResult,
    assets: list[PlaceableAsset],
    anchor_assets: set[PlaceableAsset] | None = None,
) -> None:
    """Write one environment's solved layout through each placement asset.

    Even writing zero velocity, the sim will still apply gravity and other forces from collisions,
    so collided assets will still be subject to move.

    Args:
        env: The Isaac Lab ManagerBasedEnv environment.
        env_id: The environment index.
        result: The placement result to write to the sim.
        assets: Assets represented by the placement result.
        anchor_assets: Optional precomputed set of fixed assets.
    """
    env_ids = torch.tensor([env_id], device=env.device)
    poses_by_asset = get_scene_root_poses_from_layout(assets, result, anchor_assets)
    for asset, root_poses in poses_by_asset.items():
        pose_tensors = {name: pose.to_tensor(device=env.device).unsqueeze(0) for name, pose in root_poses.items()}
        asset.write_scene_root_poses_to_sim(env, env_ids, pose_tensors)

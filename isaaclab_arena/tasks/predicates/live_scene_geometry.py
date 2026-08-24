# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Read current scene poses and build geometry used by spatial predicates."""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass

import isaaclab.sim as sim_utils
from isaaclab.cloner.cloner_utils import iter_clone_plan_matches
from isaaclab.envs import ManagerBasedEnv
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from isaaclab_arena.tasks.predicates.predicate_utils import runtime_buffer_to_torch


@dataclass(frozen=True)
class PoseFrameAabb:
    """Axis-aligned bounds expressed relative to an entity's live pose."""

    lower: torch.Tensor
    """Lower corners with shape ``(num_envs, 3)``."""

    upper: torch.Tensor
    """Upper corners with shape ``(num_envs, 3)``."""

    @property
    def center(self) -> torch.Tensor:
        """Return the center of each environment's bounds."""
        return (self.lower + self.upper) * 0.5


@dataclass(frozen=True)
class _SpawnedAssetEnvironmentGroup:
    """One representative asset prim and the environments cloned from it."""

    representative_prim: Usd.Prim
    """Spawned prim whose geometry represents this group."""

    environment_ids: tuple[int, ...]
    """Parallel environments populated from the representative prim."""


def _get_environment_id_from_prim_path(env: ManagerBasedEnv, prim_path: str) -> int | None:
    """Return the environment containing a prim path, or None for a global prim."""
    for environment_id, environment_prim_path in enumerate(env.scene.env_prim_paths):
        if prim_path == environment_prim_path or prim_path.startswith(f"{environment_prim_path}/"):
            return environment_id
    return None


def _resolve_single_representative_prim(
    stage: Usd.Stage,
    prim_path_expression: str,
    entity_name: str,
) -> Usd.Prim:
    """Resolve one concrete prim, including prims inside instanceable assets."""
    exact_prim = stage.GetPrimAtPath(prim_path_expression)
    if exact_prim.IsValid():
        return exact_prim

    matching_prims = sim_utils.find_matching_prims(prim_path_expression, stage=stage)
    assert len(matching_prims) == 1, (
        f"Scene entity '{entity_name}' path '{prim_path_expression}' resolved to {len(matching_prims)} prims; "
        "expected exactly one."
    )
    return matching_prims[0]


def _get_spawned_asset_environment_groups(
    env: ManagerBasedEnv,
    entity_name: str,
) -> list[_SpawnedAssetEnvironmentGroup]:
    """Group environments that were populated from the same asset prim.

    A rigid object set can spawn a different shape under the same scene name in
    each environment. The clone plan identifies which environments share a
    source asset, so each distinct shape needs to be read only once. Static pose
    rows use the same grouping when the frame view exposes only clone sources.
    """
    scene = env.scene
    scene_entity_cfg = getattr(scene.cfg, entity_name)
    prim_path_expression = scene_entity_cfg.prim_path

    if scene.clone_plan is not None:
        clone_matches = tuple(iter_clone_plan_matches(scene.clone_plan, prim_path_expression))
        if clone_matches:
            asset_environment_groups = []
            for _source_root, _destination_template, representative_path, environment_ids in clone_matches:
                representative_prim = _resolve_single_representative_prim(
                    scene.stage,
                    representative_path,
                    entity_name,
                )
                asset_environment_groups.append(
                    _SpawnedAssetEnvironmentGroup(
                        representative_prim=representative_prim,
                        environment_ids=environment_ids,
                    )
                )
            return asset_environment_groups

    matching_prims = sim_utils.find_matching_prims(prim_path_expression, stage=scene.stage)
    assert matching_prims, f"Scene entity '{entity_name}' has no USD prim matching '{prim_path_expression}'."
    if len(matching_prims) == 1:
        return [
            _SpawnedAssetEnvironmentGroup(
                representative_prim=matching_prims[0],
                environment_ids=tuple(range(env.num_envs)),
            )
        ]

    prims_by_environment: list[list[Usd.Prim]] = [[] for _ in range(env.num_envs)]
    for matching_prim in matching_prims:
        prim_path = matching_prim.GetPath().pathString
        environment_id = _get_environment_id_from_prim_path(env, prim_path)
        if environment_id is not None:
            prims_by_environment[environment_id].append(matching_prim)

    asset_environment_groups = []
    for environment_id, environment_prims in enumerate(prims_by_environment):
        assert len(environment_prims) == 1, (
            f"Scene entity '{entity_name}' resolved to {len(environment_prims)} prims in environment "
            f"{environment_id}; expected exactly one."
        )
        asset_environment_groups.append(
            _SpawnedAssetEnvironmentGroup(
                representative_prim=environment_prims[0],
                environment_ids=(environment_id,),
            )
        )
    return asset_environment_groups


def _find_rigid_body_live_pose_prim(representative_prim: Usd.Prim, entity_name: str) -> Usd.Prim:
    """Find the rigid-body prim whose transform matches the live root pose."""
    rigid_body_prims = sim_utils.get_all_matching_child_prims(
        representative_prim.GetPath(),
        predicate=lambda prim: prim.HasAPI(UsdPhysics.RigidBodyAPI),
        stage=representative_prim.GetStage(),
        traverse_instance_prims=False,
    )
    assert len(rigid_body_prims) == 1, (
        f"Rigid scene entity '{entity_name}' source '{representative_prim.GetPath()}' contains "
        f"{len(rigid_body_prims)} rigid bodies; expected exactly one."
    )
    return rigid_body_prims[0]


def _compute_spawned_geometry_aabb_relative_to_live_pose(
    live_pose_prim: Usd.Prim,
    entity_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute spawned geometry bounds in the frame returned by the live pose.

    A live pose contains translation and rotation but no scale. Removing scale
    and shear from the pose transform, while retaining them in each geometry
    transform, makes the resulting bounds include the scale that was spawned.
    Only regular asset geometry (USD ``purpose=default``) contributes.

    Returns:
        The lower and upper AABB corners, each with shape ``(3,)``.
    """
    transform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    live_pose_to_world_without_scale = transform_cache.GetLocalToWorldTransform(live_pose_prim).RemoveScaleShear()
    world_to_live_pose = live_pose_to_world_without_scale.GetInverse()
    bounding_box_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

    points_in_live_pose_frame: list[tuple[float, float, float]] = []
    for geometry_prim in Usd.PrimRange(live_pose_prim, Usd.TraverseInstanceProxies()):
        if not geometry_prim.IsA(UsdGeom.Gprim):
            continue
        if UsdGeom.Imageable(geometry_prim).ComputePurpose() != UsdGeom.Tokens.default_:
            continue

        local_range = bounding_box_cache.ComputeUntransformedBound(geometry_prim).ComputeAlignedRange()
        if local_range.IsEmpty():
            continue

        lower = local_range.GetMin()
        upper = local_range.GetMax()
        geometry_to_world = transform_cache.GetLocalToWorldTransform(geometry_prim)
        # Transform all eight corners because rotation can change which corner is extremal.
        for x_coordinate in (lower[0], upper[0]):
            for y_coordinate in (lower[1], upper[1]):
                for z_coordinate in (lower[2], upper[2]):
                    point_w = geometry_to_world.Transform(Gf.Vec3d(x_coordinate, y_coordinate, z_coordinate))
                    point_in_live_pose_frame = world_to_live_pose.Transform(point_w)
                    points_in_live_pose_frame.append(tuple(point_in_live_pose_frame))

    assert (
        points_in_live_pose_frame
    ), f"Scene entity '{entity_name}' has no regular asset geometry below '{live_pose_prim.GetPath()}'."
    points = np.asarray(points_in_live_pose_frame, dtype=np.float64)
    return points.min(axis=0), points.max(axis=0)


def _assert_geometry_frames_match_live_rigid_body_paths(
    env: ManagerBasedEnv,
    entity_name: str,
    expected_live_pose_paths: list[str | None],
) -> None:
    """Confirm that cached bounds and live rigid poses use the same prim frames."""
    live_pose_paths = getattr(env.scene[entity_name].root_view, "prim_paths", None)
    if live_pose_paths is None:
        return

    assert (
        len(live_pose_paths) == env.num_envs
    ), f"Rigid scene entity '{entity_name}' has {len(live_pose_paths)} live pose paths; expected {env.num_envs}."
    live_pose_paths = [str(path) for path in live_pose_paths]
    assert live_pose_paths == expected_live_pose_paths, (
        f"Rigid scene entity '{entity_name}' live pose paths do not match the geometry pose frames. "
        f"Expected {expected_live_pose_paths}, got {live_pose_paths}."
    )


def build_entity_pose_frame_aabbs(env: ManagerBasedEnv, entity_name: str) -> PoseFrameAabb:
    """Build one spawned-geometry AABB per environment.

    This reads the USD stage and can be expensive. Callers should retain the
    returned bounds and combine them with fresh live poses rather than rebuilding
    them on each simulation step.

    Args:
        env: The live manager-based environment.
        entity_name: Name of a rigid object or read-only scene reference.

    Returns:
        Bounds relative to the pose returned by ``get_entity_pose_w``.
    """
    scene = env.scene
    scene_entity_cfg = getattr(scene.cfg, entity_name)
    assert (
        entity_name in scene.rigid_objects or entity_name in scene.extras
    ), f"Scene entity '{entity_name}' must be a rigid object or read-only reference for geometric placement checks."
    is_rigid = entity_name in scene.rigid_objects

    lower_by_environment = torch.empty((env.num_envs, 3), dtype=torch.float32, device=env.device)
    upper_by_environment = torch.empty_like(lower_by_environment)
    coverage_count = torch.zeros(env.num_envs, dtype=torch.int32)
    expected_live_pose_paths: list[str | None] = [None] * env.num_envs

    asset_environment_groups = _get_spawned_asset_environment_groups(env, entity_name)
    for asset_environment_group in asset_environment_groups:
        representative_prim = asset_environment_group.representative_prim
        live_pose_prim = (
            _find_rigid_body_live_pose_prim(representative_prim, entity_name) if is_rigid else representative_prim
        )
        lower, upper = _compute_spawned_geometry_aabb_relative_to_live_pose(live_pose_prim, entity_name)
        environment_indices = torch.tensor(
            asset_environment_group.environment_ids,
            dtype=torch.long,
            device=env.device,
        )
        lower_by_environment[environment_indices] = torch.as_tensor(lower, dtype=torch.float32, device=env.device)
        upper_by_environment[environment_indices] = torch.as_tensor(upper, dtype=torch.float32, device=env.device)
        coverage_count[list(asset_environment_group.environment_ids)] += 1

        if is_rigid:
            representative_prim_path = representative_prim.GetPath().pathString
            live_pose_prim_path = live_pose_prim.GetPath().pathString
            assert live_pose_prim_path == representative_prim_path or live_pose_prim_path.startswith(
                f"{representative_prim_path}/"
            ), f"Rigid scene entity '{entity_name}' pose frame is not below its representative prim."
            live_pose_prim_suffix = live_pose_prim_path[len(representative_prim_path) :]
            for environment_id in asset_environment_group.environment_ids:
                environment_asset_path = scene_entity_cfg.prim_path.replace(
                    scene.env_regex_ns,
                    scene.env_prim_paths[environment_id],
                )
                expected_live_pose_paths[environment_id] = environment_asset_path + live_pose_prim_suffix

    assert torch.all(coverage_count == 1), (
        f"Scene entity '{entity_name}' geometry must cover every environment exactly once; "
        f"got coverage {coverage_count.tolist()}."
    )
    if is_rigid:
        _assert_geometry_frames_match_live_rigid_body_paths(env, entity_name, expected_live_pose_paths)

    return PoseFrameAabb(lower=lower_by_environment, upper=upper_by_environment)


def _join_pose(position_w: torch.Tensor, orientation_w: torch.Tensor) -> torch.Tensor:
    """Join position and orientation buffers into ``(x, y, z, qx, qy, qz, qw)`` poses."""
    return torch.cat((position_w, orientation_w), dim=-1)


def _repeat_global_reference_pose(
    env: ManagerBasedEnv,
    entity_name: str,
    prim_paths: list[str],
    position_w: torch.Tensor,
    orientation_w: torch.Tensor,
) -> torch.Tensor:
    """Repeat one global pose for every parallel environment."""
    assert len(prim_paths) == 1 and position_w.shape[0] == 1, (
        f"Global scene reference '{entity_name}' must provide exactly one pose, got "
        f"{len(prim_paths)} paths and {position_w.shape[0]} poses."
    )
    return _join_pose(
        position_w.expand(env.num_envs, 3),
        orientation_w.expand(env.num_envs, 4),
    )


def _order_reference_poses_by_environment(
    env: ManagerBasedEnv,
    entity_name: str,
    environment_id_by_pose_row: list[int | None],
    position_w: torch.Tensor,
    orientation_w: torch.Tensor,
) -> torch.Tensor:
    """Reorder one pose per environment into environment-id order."""
    assert position_w.shape[0] == env.num_envs, (
        f"Scene reference '{entity_name}' returned {position_w.shape[0]} poses for "
        f"{len(environment_id_by_pose_row)} environment paths."
    )
    pose_row_by_environment: list[int | None] = [None] * env.num_envs
    for pose_row, environment_id in enumerate(environment_id_by_pose_row):
        assert environment_id is not None, f"Scene reference '{entity_name}' has an unmapped pose path."
        assert (
            pose_row_by_environment[environment_id] is None
        ), f"Scene reference '{entity_name}' has more than one pose for environment {environment_id}."
        pose_row_by_environment[environment_id] = pose_row

    assert all(
        pose_row is not None for pose_row in pose_row_by_environment
    ), f"Scene reference '{entity_name}' does not provide exactly one pose per environment."
    ordered_pose_rows = [pose_row for pose_row in pose_row_by_environment if pose_row is not None]
    row_indices = torch.tensor(ordered_pose_rows, dtype=torch.long, device=position_w.device)
    return _join_pose(position_w.index_select(0, row_indices), orientation_w.index_select(0, row_indices))


def _use_environment_ordered_reference_poses_without_path_metadata(
    env: ManagerBasedEnv,
    entity_name: str,
    position_w: torch.Tensor,
    orientation_w: torch.Tensor,
) -> torch.Tensor:
    """Use backend poses that are already dense and ordered by environment."""
    assert position_w.shape[0] == env.num_envs, (
        f"Scene reference '{entity_name}' returned {position_w.shape[0]} poses without corresponding prim paths; "
        f"expected {env.num_envs}."
    )
    return _join_pose(position_w, orientation_w)


def _expand_clone_source_reference_poses(
    env: ManagerBasedEnv,
    entity_name: str,
    prim_paths: list[str],
    position_w: torch.Tensor,
    orientation_w: torch.Tensor,
) -> torch.Tensor:
    """Expand one live pose per clone source into one pose per environment."""
    assert (
        len(prim_paths) == position_w.shape[0]
    ), f"Scene reference '{entity_name}' returned {position_w.shape[0]} poses for {len(prim_paths)} prim paths."
    pose_row_by_path = {prim_path: pose_row for pose_row, prim_path in enumerate(prim_paths)}
    assert len(pose_row_by_path) == len(prim_paths), f"Scene reference '{entity_name}' has duplicate pose paths."

    dense_position_w = torch.empty((env.num_envs, 3), dtype=position_w.dtype, device=position_w.device)
    dense_orientation_w = torch.empty((env.num_envs, 4), dtype=orientation_w.dtype, device=orientation_w.device)
    coverage_count = torch.zeros(env.num_envs, dtype=torch.int32)
    environment_origins = env.scene.env_origins.to(device=position_w.device, dtype=position_w.dtype)

    for asset_environment_group in _get_spawned_asset_environment_groups(env, entity_name):
        representative_prim_path = asset_environment_group.representative_prim.GetPath().pathString
        assert (
            representative_prim_path in pose_row_by_path
        ), f"Scene reference '{entity_name}' has no live pose row for clone source '{representative_prim_path}'."
        pose_row = pose_row_by_path[representative_prim_path]
        source_environment_id = _get_environment_id_from_prim_path(env, representative_prim_path)
        environment_indices = torch.tensor(
            asset_environment_group.environment_ids,
            dtype=torch.long,
            device=position_w.device,
        )
        if source_environment_id is None:
            dense_position_w[environment_indices] = position_w[pose_row]
        else:
            position_relative_to_source_environment = position_w[pose_row] - environment_origins[source_environment_id]
            dense_position_w[environment_indices] = (
                position_relative_to_source_environment + environment_origins[environment_indices]
            )
        dense_orientation_w[environment_indices] = orientation_w[pose_row]
        coverage_count[list(asset_environment_group.environment_ids)] += 1

    assert torch.all(coverage_count == 1), (
        f"Scene reference '{entity_name}' poses must cover every environment exactly once; "
        f"got coverage {coverage_count.tolist()}."
    )
    return _join_pose(dense_position_w, dense_orientation_w)


def _get_reference_pose_w(env: ManagerBasedEnv, entity_name: str) -> torch.Tensor:
    """Read scene-reference poses and return one pose per environment.

    Depending on the scene and physics backend, a frame view can expose one
    global pose, one pose per environment, dense poses without path metadata,
    or one pose per clone source. This function normalizes all four layouts to
    environment-id order.
    """
    entity = env.scene[entity_name]
    position_w_buffer, orientation_w_buffer = entity.get_world_poses()
    position_w = runtime_buffer_to_torch(position_w_buffer)
    orientation_w = runtime_buffer_to_torch(orientation_w_buffer)
    assert (
        position_w.ndim == 2 and position_w.shape[1] == 3
    ), f"Scene reference '{entity_name}' returned position shape {tuple(position_w.shape)}; expected (num_poses, 3)."
    assert orientation_w.shape == (position_w.shape[0], 4), (
        f"Scene reference '{entity_name}' returned orientation shape {tuple(orientation_w.shape)}; "
        f"expected ({position_w.shape[0]}, 4)."
    )

    prim_paths = [str(path) for path in getattr(entity, "prim_paths", ())]
    if not prim_paths:
        return _use_environment_ordered_reference_poses_without_path_metadata(
            env,
            entity_name,
            position_w,
            orientation_w,
        )

    environment_id_by_pose_row = [_get_environment_id_from_prim_path(env, prim_path) for prim_path in prim_paths]
    if all(environment_id is None for environment_id in environment_id_by_pose_row):
        return _repeat_global_reference_pose(env, entity_name, prim_paths, position_w, orientation_w)

    assert all(
        environment_id is not None for environment_id in environment_id_by_pose_row
    ), f"Scene reference '{entity_name}' mixes global and environment-scoped pose paths."
    if len(prim_paths) == env.num_envs:
        return _order_reference_poses_by_environment(
            env,
            entity_name,
            environment_id_by_pose_row,
            position_w,
            orientation_w,
        )

    return _expand_clone_source_reference_poses(env, entity_name, prim_paths, position_w, orientation_w)


def get_entity_pose_w(env: ManagerBasedEnv, entity_name: str) -> torch.Tensor:
    """Return current entity poses as ``(x, y, z, qx, qy, qz, qw)``.

    Args:
        env: The live manager-based environment.
        entity_name: Name of a rigid object or read-only scene reference.

    Returns:
        World-frame poses with shape ``(num_envs, 7)``.
    """
    scene = env.scene
    entity = scene[entity_name]
    if entity_name in scene.rigid_objects:
        pose_w = runtime_buffer_to_torch(entity.data.root_pose_w)
        assert pose_w.shape == (env.num_envs, 7), (
            f"Rigid scene entity '{entity_name}' returned pose shape {tuple(pose_w.shape)}; "
            f"expected ({env.num_envs}, 7)."
        )
        return pose_w

    assert (
        entity_name in scene.extras
    ), f"Scene entity '{entity_name}' must be a rigid object or read-only reference for pose lookup."
    return _get_reference_pose_w(env, entity_name)

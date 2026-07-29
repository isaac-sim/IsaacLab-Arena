# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

import warp as wp
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.contact_sensor.contact_sensor import ContactSensor
from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_inv, quat_mul

from isaaclab_arena.tasks.predicates.object_settling import get_object_initial_rest_state
from isaaclab_arena.tasks.predicates.predicate_utils import (
    get_asset_pose_w,
    get_env,
    get_root_lin_vel_w,
    get_root_pos_w,
    select,
)

if TYPE_CHECKING:
    from isaaclab_arena.relations.placement_asset import PlaceableAsset


def object_is_above_height(
    env: ManagerBasedRLEnv,
    object_name: str,
    surface_height: float | None = None,
    use_settled_state: bool = False,
    distance: float = 1e-2,
    env_id: int | None = None,
) -> torch.Tensor:
    """Checks if an object is above a certain height.

    The reference height is either a fixed ``surface_height`` or, when ``use_settled_state`` is set, the
    object's recorded resting height (see ``objects_settled``). For envs where no settled state
    has been recorded, the result is always False.

    Returns True when ``object_name`` is at least ``distance`` m above a height reference.
    """

    assert (
        surface_height is not None
    ) != use_settled_state, "object_is_above_height requires exactly one of surface_height or use_settled_state"

    object_z = get_root_pos_w(env, object_name)[:, 2]
    if use_settled_state:
        settled_pos, has_settled = get_object_initial_rest_state(env, object_name)
        result = has_settled & (object_z > (settled_pos[:, 2] + distance))
    else:
        result = object_z > (surface_height + distance)
    return select(result, env_id)


def object_moving(
    env: ManagerBasedRLEnv,
    object_name: str,
    velocity_threshold: float = 1e-2,
    env_id: int | None = None,
) -> torch.Tensor:
    """Checks if an object is moving above a certain velocity threshold.

    Returns True when object_name's linear speed exceeds velocity_threshold (m/s).
    """

    speed = torch.linalg.vector_norm(get_root_lin_vel_w(env, object_name), dim=-1)
    result = speed > velocity_threshold
    return select(result, env_id)


def objects_in_proximity(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    target_object_cfg: SceneEntityCfg,
    max_y_separation: float,
    max_x_separation: float,
    max_z_separation: float,
) -> torch.Tensor:
    """Determine if two objects are within a certain proximity of each other.

    Returns True when the object is within a certain proximity of the target object.
    """

    # Get object entities from the scene
    object: RigidObject = env.scene[object_cfg.name]
    target_object: RigidObject = env.scene[target_object_cfg.name]

    # Get positions relative to environment origin
    object_pos = wp.to_torch(object.data.root_pos_w) - env.scene.env_origins
    target_object_pos = wp.to_torch(target_object.data.root_pos_w) - env.scene.env_origins

    # object to target object
    x_separation = torch.abs(object_pos[:, 0] - target_object_pos[:, 0])
    y_separation = torch.abs(object_pos[:, 1] - target_object_pos[:, 1])
    z_separation = torch.abs(object_pos[:, 2] - target_object_pos[:, 2])

    done = x_separation < max_x_separation
    done = torch.logical_and(done, y_separation < max_y_separation)
    done = torch.logical_and(done, z_separation < max_z_separation)

    return done


def contact_force_is_upward_support(
    force_matrix_w: torch.Tensor,
    force_threshold: float,
    support_cone_half_angle_deg: float,
) -> torch.Tensor:
    """Check whether destination contact forces support an object upward."""

    assert (
        force_matrix_w.ndim >= 2 and force_matrix_w.shape[-1] == 3
    ), f"force_matrix_w must have shape (num_envs, ..., 3), got {tuple(force_matrix_w.shape)}"
    assert force_threshold >= 0.0, f"force_threshold must be non-negative, got {force_threshold}"
    assert (
        0.0 <= support_cone_half_angle_deg < 90.0
    ), f"support_cone_half_angle_deg must be in [0, 90), got {support_cone_half_angle_deg}"

    if force_matrix_w.ndim == 2:
        destination_force_w = force_matrix_w
    else:
        contact_axes = tuple(range(1, force_matrix_w.ndim - 1))
        destination_force_w = force_matrix_w.sum(dim=contact_axes)
    force_magnitude = torch.linalg.vector_norm(destination_force_w, dim=-1)
    upward_force = destination_force_w[:, 2]
    minimum_upward_fraction = math.cos(math.radians(support_cone_half_angle_deg))

    return (
        (force_magnitude >= force_threshold)
        & (upward_force > 0.0)
        & (upward_force >= force_magnitude * minimum_upward_fraction)
    )


def object_centroid_in_destination_footprint(
    env: ManagerBasedRLEnv,
    object_asset: PlaceableAsset,
    destination_asset: PlaceableAsset,
    footprint_tolerance: float = 1e-2,
) -> torch.Tensor:
    """Check whether an object's bounding-box centroid is within a destination's world XY footprint."""

    assert footprint_tolerance >= 0.0, f"footprint_tolerance must be non-negative, got {footprint_tolerance}"

    unwrapped_env = get_env(env)
    object_bounding_box = _get_asset_bounding_box_per_env(object_asset, unwrapped_env.num_envs).to(unwrapped_env.device)
    destination_bounding_box = _get_asset_bounding_box_per_env(destination_asset, unwrapped_env.num_envs).to(
        unwrapped_env.device
    )

    object_bounding_box_pose_w = _get_bounding_box_pose_w(unwrapped_env, object_asset)
    destination_bounding_box_pose_w = _get_bounding_box_pose_w(unwrapped_env, destination_asset)
    object_centroid_w, _ = combine_frame_transforms(
        object_bounding_box_pose_w[:, :3],
        object_bounding_box_pose_w[:, 3:],
        object_bounding_box.center,
    )

    destination_corners = destination_bounding_box.get_corners_at()
    num_envs, num_corners, _ = destination_corners.shape
    destination_quaternions = (
        destination_bounding_box_pose_w[:, None, 3:].expand(num_envs, num_corners, 4).reshape(-1, 4)
    )
    destination_corners_w = quat_apply(
        destination_quaternions,
        destination_corners.reshape(-1, 3),
    ).reshape(num_envs, num_corners, 3)
    destination_corners_w += destination_bounding_box_pose_w[:, None, :3]

    minimum_xy = destination_corners_w[:, :, :2].amin(dim=1) - footprint_tolerance
    maximum_xy = destination_corners_w[:, :, :2].amax(dim=1) + footprint_tolerance
    return torch.all(
        (object_centroid_w[:, :2] >= minimum_xy) & (object_centroid_w[:, :2] <= maximum_xy),
        dim=-1,
    )


def _get_bounding_box_pose_w(env, asset: PlaceableAsset) -> torch.Tensor:
    """Get the world pose of the frame in which an asset's bounding box is expressed."""

    asset_pose_w = get_asset_pose_w(env, asset)
    pose_relative_to_parent = getattr(asset, "initial_pose_relative_to_parent", None)
    if pose_relative_to_parent is None:
        return asset_pose_w

    unwrapped_env = get_env(env)
    relative_pose = pose_relative_to_parent.to_tensor(device=unwrapped_env.device).expand(unwrapped_env.num_envs, 7)
    bounding_box_quaternion_w = quat_mul(asset_pose_w[:, 3:], quat_inv(relative_pose[:, 3:]))
    return torch.cat((asset_pose_w[:, :3], bounding_box_quaternion_w), dim=-1)


def _get_asset_bounding_box_per_env(asset: PlaceableAsset, num_envs: int):
    """Get root-relative bounds per environment, using assigned object-set variants when available."""

    if getattr(asset, "variant_indices_by_env", None) is not None:
        return asset.get_bounding_box_per_env(num_envs)

    bounding_box = asset.get_bounding_box()
    assert bounding_box.num_envs in (
        1,
        num_envs,
    ), f"Asset '{asset.name}' has {bounding_box.num_envs} bounding boxes for {num_envs} environments."
    if bounding_box.num_envs == num_envs:
        return bounding_box
    return type(bounding_box)(
        min_point=bounding_box.min_point.expand(num_envs, 3),
        max_point=bounding_box.max_point.expand(num_envs, 3),
    )


def object_on_destination(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("pick_up_object"),
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("pick_up_object_contact_sensor"),
    force_threshold: float = 1.0,
    velocity_threshold: float = 0.5,
    object_asset: PlaceableAsset | None = None,
    destination_asset: PlaceableAsset | None = None,
    support_cone_half_angle_deg: float = 45.0,
    footprint_tolerance: float = 1e-2,
) -> torch.Tensor:
    """Check whether an object is resting on and within the footprint of its destination.

    Returns True when destination contact supports the object upward, the object's
    bounding-box centroid is within the destination's world XY footprint, and its
    linear speed is below the threshold.
    """

    unwrapped_env = get_env(env)
    object_entity: RigidObject = unwrapped_env.scene[object_cfg.name]
    sensor: ContactSensor = unwrapped_env.scene[contact_sensor_cfg.name]
    assert object_asset is not None, "object_asset is required"
    assert destination_asset is not None, "destination_asset is required"

    force_matrix_w = wp.to_torch(sensor.data.force_matrix_w)
    supported_by_destination = contact_force_is_upward_support(
        force_matrix_w,
        force_threshold=force_threshold,
        support_cone_half_angle_deg=support_cone_half_angle_deg,
    )
    centroid_in_footprint = object_centroid_in_destination_footprint(
        env=unwrapped_env,
        object_asset=object_asset,
        destination_asset=destination_asset,
        footprint_tolerance=footprint_tolerance,
    )

    object_linear_speed = torch.linalg.vector_norm(wp.to_torch(object_entity.data.root_lin_vel_w), dim=-1)
    object_at_rest = object_linear_speed < velocity_threshold

    return supported_by_destination & centroid_in_footprint & object_at_rest


def objects_on_destinations(
    env: ManagerBasedRLEnv,
    object_cfg_list: list[SceneEntityCfg] = [SceneEntityCfg("pick_up_object")],
    contact_sensor_cfg_list: list[SceneEntityCfg] = [SceneEntityCfg("pick_up_object_contact_sensor")],
    force_threshold: float = 1.0,
    velocity_threshold: float = 0.5,
) -> torch.Tensor:
    """Check whether every object has destination contact and low linear speed.

    This preserves the existing multi-object behavior until indirect support between
    objects sharing a destination is defined.
    """

    assert len(object_cfg_list) == len(contact_sensor_cfg_list), (
        "object_cfg_list and contact_sensor_cfg_list must have equal length, got "
        f"{len(object_cfg_list)} objects and {len(contact_sensor_cfg_list)} sensors"
    )

    unwrapped_env = get_env(env)
    condition_met = torch.ones((unwrapped_env.num_envs), device=unwrapped_env.device, dtype=torch.bool)
    for object_cfg, contact_sensor_cfg in zip(object_cfg_list, contact_sensor_cfg_list):
        object_entity: RigidObject = unwrapped_env.scene[object_cfg.name]
        sensor: ContactSensor = unwrapped_env.scene[contact_sensor_cfg.name]
        assert sensor.data.force_matrix_w.shape[2] == 1
        assert sensor.data.force_matrix_w.shape[1] == 1

        force_matrix_norm = torch.linalg.vector_norm(wp.to_torch(sensor.data.force_matrix_w), dim=-1).reshape(-1)
        force_above_threshold = force_matrix_norm > force_threshold
        object_linear_speed = torch.linalg.vector_norm(wp.to_torch(object_entity.data.root_lin_vel_w), dim=-1)
        single_condition = force_above_threshold & (object_linear_speed < velocity_threshold)
        condition_met = torch.logical_and(condition_met, single_condition)
    return condition_met

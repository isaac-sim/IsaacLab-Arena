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
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

from isaaclab_arena.relations.bounding_box_helpers import get_bounding_box_per_env
from isaaclab_arena.tasks.predicates.object_settling import get_object_initial_rest_state
from isaaclab_arena.tasks.predicates.predicate_utils import (
    ArenaAssetHandle,
    get_env,
    get_root_lin_vel_w,
    get_root_pos_w,
    select,
)

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase


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


def object_in_container(
    env: ManagerBasedRLEnv,
    object_asset_handle: ArenaAssetHandle,
    container_asset_handle: ArenaAssetHandle,
) -> torch.Tensor:
    """Check whether an object's bounding-box centroid is within open-top container bounds.

    The asset handles preserve the original Arena assets and their cached bounds when manager
    configurations are copied.
    """

    return _object_centroid_in_container_bounds(
        env,
        object_asset=object_asset_handle.asset,
        container_asset=container_asset_handle.asset,
    )


def _object_centroid_in_container_bounds(
    env: ManagerBasedRLEnv,
    object_asset: ObjectBase,
    container_asset: ObjectBase,
) -> torch.Tensor:
    """Check whether an object's bounding-box centroid is within open-top container bounds.

    The centroid is transformed into the container bounding box's local frame. The container's
    local X and Y bounds and lower Z bound are enforced; its upper Z bound is intentionally ignored.
    """

    unwrapped_env = get_env(env)
    object_bounding_box = get_bounding_box_per_env(object_asset, unwrapped_env.num_envs).to(unwrapped_env.device)
    container_bounding_box = get_bounding_box_per_env(container_asset, unwrapped_env.num_envs).to(unwrapped_env.device)

    object_bounding_box_pose_w = object_asset.get_bounding_box_pose(unwrapped_env, is_relative=False)
    container_bounding_box_pose_w = container_asset.get_bounding_box_pose(unwrapped_env, is_relative=False)
    object_centroid_w, _ = combine_frame_transforms(
        object_bounding_box_pose_w[:, :3],
        object_bounding_box_pose_w[:, 3:],
        object_bounding_box.center,
    )
    object_centroid_container, _ = subtract_frame_transforms(
        container_bounding_box_pose_w[:, :3],
        container_bounding_box_pose_w[:, 3:],
        object_centroid_w,
    )

    inside_x = (object_centroid_container[:, 0] >= container_bounding_box.min_point[:, 0]) & (
        object_centroid_container[:, 0] <= container_bounding_box.max_point[:, 0]
    )
    inside_y = (object_centroid_container[:, 1] >= container_bounding_box.min_point[:, 1]) & (
        object_centroid_container[:, 1] <= container_bounding_box.max_point[:, 1]
    )
    above_bottom = object_centroid_container[:, 2] >= container_bounding_box.min_point[:, 2]
    return inside_x & inside_y & above_bottom


def contact_force_is_upward_support(
    force_matrix_w: torch.Tensor,
    force_threshold: float,
    support_cone_half_angle_deg: float,
) -> torch.Tensor:
    """Check whether filtered contact forces support an object upward."""

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


def object_not_moving(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("pick_up_object"),
    velocity_threshold: float = 0.5,
) -> torch.Tensor:
    """Checks if an object's linear speed is below a velocity threshold."""
    velocity_w = get_root_lin_vel_w(env, object_cfg.name)
    velocity_w_norm = torch.norm(velocity_w, dim=-1)
    return velocity_w_norm < velocity_threshold


def object_in_contact(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("pick_up_object"),
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("pick_up_object_contact_sensor"),
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Checks if an object's filtered contact force exceeds a threshold."""
    unwrapped_env = get_env(env)
    sensor: ContactSensor = unwrapped_env.scene[contact_sensor_cfg.name]

    # force_matrix_w shape is (N, B, M, 3), where N is the number of sensors, B is number of bodies in each sensor
    # and ``M`` is the number of filtered bodies.
    # We assume B = 1 and M = 1
    assert sensor.data.force_matrix_w.shape[2] == 1
    assert sensor.data.force_matrix_w.shape[1] == 1
    # NOTE(alexmillane, 2025-08-04): We expect the binary flags to have shape (N, )
    # where N is the number of envs.
    force_matrix_norm = torch.norm(wp.to_torch(sensor.data.force_matrix_w), dim=-1).reshape(-1)
    return force_matrix_norm > force_threshold


def object_supported_by_destination(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("pick_up_object_contact_sensor"),
    force_threshold: float = 1.0,
    support_cone_half_angle_deg: float = 45.0,
) -> torch.Tensor:
    """Check whether filtered destination contact supports an object upward."""

    unwrapped_env = get_env(env)
    sensor: ContactSensor = unwrapped_env.scene[contact_sensor_cfg.name]
    force_matrix_w = sensor.data.force_matrix_w
    assert force_matrix_w is not None, f"Contact sensor '{contact_sensor_cfg.name}' has no filtered force matrix."
    return contact_force_is_upward_support(
        wp.to_torch(force_matrix_w),
        force_threshold=force_threshold,
        support_cone_half_angle_deg=support_cone_half_angle_deg,
    )


def object_on_destination(
    env: ManagerBasedRLEnv,
    object_asset_handle: ArenaAssetHandle,
    destination_asset_handle: ArenaAssetHandle,
    object_cfg: SceneEntityCfg = SceneEntityCfg("pick_up_object"),
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("pick_up_object_contact_sensor"),
    force_threshold: float = 1.0,
    velocity_threshold: float = 0.5,
    support_cone_half_angle_deg: float = 45.0,
) -> torch.Tensor:
    """Check whether an object is inside and stably supported by its destination.

    Returns True when the object's bounding-box centroid is within the destination's open-top
    bounds, its filtered destination contact force points upward within the support cone, and
    its linear speed is below the velocity threshold.
    """

    inside_destination = object_in_container(env, object_asset_handle, destination_asset_handle)
    supported_by_destination = object_supported_by_destination(
        env,
        contact_sensor_cfg=contact_sensor_cfg,
        force_threshold=force_threshold,
        support_cone_half_angle_deg=support_cone_half_angle_deg,
    )
    velocity_below_threshold = object_not_moving(env, object_cfg, velocity_threshold)
    return inside_destination & supported_by_destination & velocity_below_threshold


def objects_on_destinations(
    env: ManagerBasedRLEnv,
    object_asset_handle_list: list[ArenaAssetHandle],
    destination_asset_handle_list: list[ArenaAssetHandle],
    object_cfg_list: list[SceneEntityCfg] = [SceneEntityCfg("pick_up_object")],
    contact_sensor_cfg_list: list[SceneEntityCfg] = [SceneEntityCfg("pick_up_object_contact_sensor")],
    force_threshold: float = 1.0,
    velocity_threshold: float = 0.5,
    support_cone_half_angle_deg: float = 45.0,
) -> torch.Tensor:
    """Multi-object version of `object_on_destination`.

    Returns True only when ALL objects in the list satisfy the destination condition.
    See `object_on_destination` for details on the single-object logic.
    """

    list_lengths = (
        len(object_cfg_list),
        len(contact_sensor_cfg_list),
        len(object_asset_handle_list),
        len(destination_asset_handle_list),
    )
    assert len(set(list_lengths)) == 1, (
        "Object configs, contact sensors, object assets, and destination assets must have equal lengths, got "
        f"{list_lengths}."
    )

    unwrapped_env = get_env(env)
    condition_met = torch.ones((unwrapped_env.num_envs), device=unwrapped_env.device, dtype=torch.bool)
    for object_cfg, contact_sensor_cfg, object_asset_handle, destination_asset_handle in zip(
        object_cfg_list,
        contact_sensor_cfg_list,
        object_asset_handle_list,
        destination_asset_handle_list,
    ):
        single_condition = object_on_destination(
            env=env,
            object_cfg=object_cfg,
            contact_sensor_cfg=contact_sensor_cfg,
            force_threshold=force_threshold,
            velocity_threshold=velocity_threshold,
            object_asset_handle=object_asset_handle,
            destination_asset_handle=destination_asset_handle,
            support_cone_half_angle_deg=support_cone_half_angle_deg,
        )
        condition_met = torch.logical_and(condition_met, single_condition)
    return condition_met

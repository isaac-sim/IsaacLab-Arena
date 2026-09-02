# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Stateless spatial predicates and geometric checks."""

from __future__ import annotations

import math
import torch

import warp as wp
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.contact_sensor.contact_sensor import ContactSensor
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from isaaclab_arena.tasks.predicates.object_settling import get_object_initial_rest_state
from isaaclab_arena.tasks.predicates.predicate_utils import get_env, get_root_lin_vel_w, get_root_pos_w, select
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox


def object_bounds_center_over_destination(
    T_W_O: torch.Tensor,
    object_bounds_center_O: torch.Tensor,
    T_W_D: torch.Tensor,
    destination_bounds_D: AxisAlignedBoundingBox,
) -> torch.Tensor:
    """Check whether an object's bounds center is over a destination.

    The check requires the object's bounds center to be inside the destination's
    X/Y footprint and above its lower Z bound. The upper Z bound is intentionally
    ignored so the same check works for open containers and supporting surfaces.
    This is a center-point test, not full-object containment.

    Args:
        T_W_O: Object poses mapping points from object frame ``O`` into world
            frame ``W``. Shape is ``(num_envs, 7)`` with quaternion order
            ``(x, y, z, w)``.
        object_bounds_center_O: Center of the object's bounds expressed in
            frame ``O``. Shape is ``(num_envs, 3)``.
        T_W_D: Destination poses mapping points from destination frame ``D``
            into world frame ``W``. Shape is ``(num_envs, 7)`` with quaternion
            order ``(x, y, z, w)``.
        destination_bounds_D: Destination bounds aligned with frame ``D`` and
            measured from its origin.

    Returns:
        One Boolean result per environment.
    """
    t_W_O, q_W_O = T_W_O[:, :3], T_W_O[:, 3:]
    t_W_D, q_W_D = T_W_D[:, :3], T_W_D[:, 3:]

    object_bounds_center_W = t_W_O + quat_apply(q_W_O, object_bounds_center_O)
    object_bounds_center_D = quat_apply_inverse(
        q_W_D,
        object_bounds_center_W - t_W_D,
    )
    center_inside_horizontal_bounds = (
        (object_bounds_center_D[:, :2] >= destination_bounds_D.min_point[:, :2])
        & (object_bounds_center_D[:, :2] <= destination_bounds_D.max_point[:, :2])
    ).all(dim=-1)
    center_above_destination_bottom = object_bounds_center_D[:, 2] >= destination_bounds_D.min_point[:, 2]
    return center_inside_horizontal_bounds & center_above_destination_bottom


def contact_force_is_upward_support(
    contact_force_w: torch.Tensor,
    force_threshold: float,
    support_cone_half_angle_deg: float,
) -> torch.Tensor:
    """Check whether contact forces point upward strongly enough.

    Args:
        contact_force_w: World-frame force vectors with shape
            ``(num_envs, 3)`` in newtons.
        force_threshold: Minimum force magnitude in newtons.
        support_cone_half_angle_deg: Maximum angle in degrees between the force
            vector and world ``+Z``. Zero accepts only a straight-up force.

    Returns:
        One Boolean result per environment.
    """
    assert (
        contact_force_w.ndim == 2 and contact_force_w.shape[1] == 3
    ), f"contact_force_w must have shape (num_envs, 3), got {tuple(contact_force_w.shape)}."
    assert force_threshold >= 0.0, f"force_threshold must be non-negative, got {force_threshold}."
    assert (
        0.0 <= support_cone_half_angle_deg < 90.0
    ), f"support_cone_half_angle_deg must be in [0, 90), got {support_cone_half_angle_deg}."

    force_magnitude = torch.linalg.vector_norm(contact_force_w, dim=-1)
    upward_force = contact_force_w[:, 2]
    minimum_upward_fraction = math.cos(math.radians(support_cone_half_angle_deg))
    return (
        (force_magnitude >= force_threshold)
        & (upward_force > 0.0)
        & (upward_force >= force_magnitude * minimum_upward_fraction)
    )


def object_is_moving_slowly(
    object_linear_velocity_w: torch.Tensor,
    velocity_threshold: float,
) -> torch.Tensor:
    """Check whether object linear speed is below the threshold."""
    return torch.linalg.vector_norm(object_linear_velocity_w, dim=-1) < velocity_threshold


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


def object_on_destination(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    destination_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    force_threshold: float,
    velocity_threshold: float,
    support_cone_half_angle_deg: float = 45.0,
) -> torch.Tensor:
    """Check whether an object is stably placed on its destination.

    The object's spawned-bounds center must be over the destination footprint
    and above its bottom. The destination must exert an upward support force,
    and the object's linear speed must be below the configured threshold.

    Args:
        env: The wrapped or unwrapped manager-based environment.
        object_cfg: The rigid object being placed.
        destination_cfg: The rigid object or scene entry receiving the object.
        contact_sensor_cfg: The object's contact sensor filtered to the destination.
        force_threshold: Minimum upward support force in newtons.
        velocity_threshold: Maximum object linear speed in meters per second.
        support_cone_half_angle_deg: Maximum angle from world ``+Z`` for the support force.

    Returns:
        One Boolean result per environment.
    """

    unwrapped_env = get_env(env)
    arena_world = unwrapped_env.arena_world
    object_pose_w = arena_world.get_pose_w(object_cfg.name)
    destination_pose_w = arena_world.get_pose_w(destination_cfg.name)
    object_center_over_destination = object_bounds_center_over_destination(
        T_W_O=object_pose_w,
        object_bounds_center_O=arena_world.get_local_aabb(object_cfg.name).center,
        T_W_D=destination_pose_w,
        destination_bounds_D=arena_world.get_local_aabb(destination_cfg.name),
    )

    contact_sensor: ContactSensor = unwrapped_env.scene[contact_sensor_cfg.name]
    force_matrix_w = contact_sensor.data.force_matrix_w
    assert force_matrix_w is not None, f"Contact sensor '{contact_sensor_cfg.name}' has no filtered force matrix."
    force_matrix_w = force_matrix_w.torch
    assert force_matrix_w.shape == (unwrapped_env.num_envs, 1, 1, 3), (
        f"Contact sensor '{contact_sensor_cfg.name}' must provide one sensed body and one filtered body; "
        f"got force shape {tuple(force_matrix_w.shape)}."
    )
    # The two zeros select the sensor's single sensed body and single filtered destination body.
    support_force_on_object_w = force_matrix_w[:, 0, 0, :]
    destination_provides_upward_support = contact_force_is_upward_support(
        contact_force_w=support_force_on_object_w,
        force_threshold=force_threshold,
        support_cone_half_angle_deg=support_cone_half_angle_deg,
    )

    object_linear_velocity_w = arena_world.get_linear_velocity_w(object_cfg.name)
    object_moves_slowly = object_is_moving_slowly(object_linear_velocity_w, velocity_threshold)
    return object_center_over_destination & destination_provides_upward_support & object_moves_slowly


def objects_on_destinations(
    env: ManagerBasedRLEnv,
    object_cfg_list: list[SceneEntityCfg],
    destination_cfg_list: list[SceneEntityCfg],
    contact_sensor_cfg_list: list[SceneEntityCfg],
    force_threshold: float,
    velocity_threshold: float,
    support_cone_half_angle_deg: float = 45.0,
) -> torch.Tensor:
    """Check whether every object is stably placed on its destination.

    Args:
        env: The wrapped or unwrapped manager-based environment.
        object_cfg_list: The rigid objects being placed.
        destination_cfg_list: The destination corresponding to each object.
        contact_sensor_cfg_list: The filtered contact sensor corresponding to each object.
        force_threshold: Minimum upward support force in newtons.
        velocity_threshold: Maximum object linear speed in meters per second.
        support_cone_half_angle_deg: Maximum angle from world ``+Z`` for each support force.

    Returns:
        One Boolean result per environment.
    """

    assert len(object_cfg_list) == len(destination_cfg_list) == len(contact_sensor_cfg_list), (
        "object_cfg_list, destination_cfg_list, and contact_sensor_cfg_list must have equal length; got "
        f"{len(object_cfg_list)} objects, {len(destination_cfg_list)} destinations, and "
        f"{len(contact_sensor_cfg_list)} sensors."
    )

    unwrapped_env = get_env(env)
    condition_met = torch.ones((unwrapped_env.num_envs), device=unwrapped_env.device, dtype=torch.bool)
    for object_cfg, destination_cfg, contact_sensor_cfg in zip(
        object_cfg_list,
        destination_cfg_list,
        contact_sensor_cfg_list,
    ):
        single_condition = object_on_destination(
            env=unwrapped_env,
            object_cfg=object_cfg,
            destination_cfg=destination_cfg,
            contact_sensor_cfg=contact_sensor_cfg,
            force_threshold=force_threshold,
            velocity_threshold=velocity_threshold,
            support_cone_half_angle_deg=support_cone_half_angle_deg,
        )
        condition_met = torch.logical_and(condition_met, single_condition)
    return condition_met

# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Stateless spatial predicates and geometric checks."""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.contact_sensor.contact_sensor import ContactSensor
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from isaaclab_arena.tasks.predicates.object_settling import get_object_initial_rest_state
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv


def object_bounds_center_over_destination(
    object_centroid_W: torch.Tensor,
    T_W_D: torch.Tensor,
    destination_bounds_D: AxisAlignedBoundingBox,
) -> torch.Tensor:
    """Check whether an object's bounds center is over a destination.

    The check requires the object's bounds center to be inside the destination's
    X/Y footprint and above its lower Z bound. The upper Z bound is intentionally
    ignored so the same check works for open containers and supporting surfaces.
    This is a center-point test, not full-object containment.

    Args:
        object_centroid_W: Object geometry centroid expressed in world frame
            ``W``. Shape is ``(num_envs, 3)``.
        T_W_D: Destination poses mapping points from destination frame ``D``
            into world frame ``W``. Shape is ``(num_envs, 7)`` with quaternion
            order ``(x, y, z, w)``.
        destination_bounds_D: Destination bounds aligned with frame ``D`` and
            measured from its origin.

    Returns:
        One Boolean result per environment.
    """
    t_W_D, q_W_D = T_W_D[:, :3], T_W_D[:, 3:]

    object_centroid_D = quat_apply_inverse(
        q_W_D,
        object_centroid_W - t_W_D,
    )
    center_inside_horizontal_bounds = (
        (object_centroid_D[:, :2] >= destination_bounds_D.min_point[:, :2])
        & (object_centroid_D[:, :2] <= destination_bounds_D.max_point[:, :2])
    ).all(dim=-1)
    center_above_destination_bottom = object_centroid_D[:, 2] >= destination_bounds_D.min_point[:, 2]
    return center_inside_horizontal_bounds & center_above_destination_bottom


def contact_force_is_upward_support(
    contact_force_w: torch.Tensor,
    force_threshold: float,
    support_cone_half_angle_rad: float,
) -> torch.Tensor:
    """Check whether contact forces point upward strongly enough.

    Args:
        contact_force_w: World-frame force vectors with shape
            ``(num_envs, 3)`` in newtons.
        force_threshold: Minimum force magnitude in newtons.
        support_cone_half_angle_rad: Maximum angle in radians between the force
            vector and world ``+Z``. Zero accepts only a straight-up force.

    Returns:
        One Boolean result per environment.
    """
    assert (
        contact_force_w.ndim == 2 and contact_force_w.shape[1] == 3
    ), f"contact_force_w must have shape (num_envs, 3), got {tuple(contact_force_w.shape)}."
    assert force_threshold >= 0.0, f"force_threshold must be non-negative, got {force_threshold}."
    assert (
        0.0 <= support_cone_half_angle_rad < math.pi / 2
    ), f"support_cone_half_angle_rad must be in [0, pi / 2), got {support_cone_half_angle_rad}."

    force_magnitude = torch.linalg.vector_norm(contact_force_w, dim=-1)
    upward_force = contact_force_w[:, 2]
    minimum_upward_fraction = math.cos(support_cone_half_angle_rad)
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


def _position_relative_to_target(
    env: IsaacLabArenaManagerBasedRLEnv,
    subject_name: str,
    receiver_name: str,
    target_offset_xyz: tuple[float, float, float],
) -> torch.Tensor:
    """Return the subject origin relative to a receiver-local target position."""
    arena_world = env.arena_world
    T_W_S = arena_world.get_pose_w(subject_name)
    T_W_R = arena_world.get_pose_w(receiver_name)
    position_R = quat_apply_inverse(T_W_R[:, 3:], T_W_S[:, :3] - T_W_R[:, :3])
    target_position_R = torch.as_tensor(target_offset_xyz, dtype=position_R.dtype, device=position_R.device)
    return position_R - target_position_R


def xy_in_proximity(
    env: IsaacLabArenaManagerBasedRLEnv,
    subject_name: str,
    receiver_name: str,
    target_offset_xyz: tuple[float, float, float],
    tolerance_xy: float,
) -> torch.Tensor:
    """Check subject proximity to a receiver-local target in the target XY plane."""
    p_rel = _position_relative_to_target(env, subject_name, receiver_name, target_offset_xyz)
    return torch.linalg.vector_norm(p_rel[:, :2], dim=-1) <= tolerance_xy


def depth_in_range(
    env: IsaacLabArenaManagerBasedRLEnv,
    subject_name: str,
    receiver_name: str,
    target_offset_xyz: tuple[float, float, float],
    depth_min: float,
    depth_max: float,
) -> torch.Tensor:
    """Check subject depth relative to a receiver-local target position."""
    assert depth_min <= depth_max, f"depth_min ({depth_min}) must not exceed depth_max ({depth_max})."
    p_rel = _position_relative_to_target(env, subject_name, receiver_name, target_offset_xyz)
    return (p_rel[:, 2] >= depth_min) & (p_rel[:, 2] <= depth_max)


def tilt_axis_aligned(
    env: IsaacLabArenaManagerBasedRLEnv,
    subject_name: str,
    receiver_name: str,
    max_tilt_rad: float,
    subject_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    receiver_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> torch.Tensor:
    """Check the angle between configured subject and receiver axes."""
    assert 0.0 <= max_tilt_rad <= math.pi, f"max_tilt_rad must be in [0, pi], got {max_tilt_rad}."
    arena_world = env.arena_world
    T_W_S = arena_world.get_pose_w(subject_name)
    T_W_R = arena_world.get_pose_w(receiver_name)
    subject_axis_F = torch.as_tensor(subject_axis, dtype=T_W_S.dtype, device=T_W_S.device)
    receiver_axis_F = torch.as_tensor(receiver_axis, dtype=T_W_R.dtype, device=T_W_R.device)
    assert subject_axis_F.shape == (3,) and torch.linalg.vector_norm(subject_axis_F) > 0
    assert receiver_axis_F.shape == (3,) and torch.linalg.vector_norm(receiver_axis_F) > 0
    subject_axis_F = subject_axis_F / torch.linalg.vector_norm(subject_axis_F)
    receiver_axis_F = receiver_axis_F / torch.linalg.vector_norm(receiver_axis_F)
    subject_axis_w = quat_apply(T_W_S[:, 3:], subject_axis_F.expand(env.num_envs, -1))
    receiver_axis_w = quat_apply(T_W_R[:, 3:], receiver_axis_F.expand(env.num_envs, -1))
    axis_dot = torch.sum(subject_axis_w * receiver_axis_w, dim=-1)
    return axis_dot >= math.cos(max_tilt_rad)


def velocity_below_threshold(
    env: IsaacLabArenaManagerBasedRLEnv,
    subject_name: str,
    linear_velocity_threshold: float,
    angular_velocity_threshold: float | None = None,
) -> torch.Tensor:
    """Check subject root linear speed and, optionally, angular speed."""
    arena_world = env.arena_world
    linear_velocity_w = arena_world.get_root_linear_velocity_w(subject_name)
    result = torch.linalg.vector_norm(linear_velocity_w, dim=-1) <= linear_velocity_threshold
    if angular_velocity_threshold is not None:
        angular_velocity_w = arena_world.get_root_angular_velocity_w(subject_name)
        result &= torch.linalg.vector_norm(angular_velocity_w, dim=-1) <= angular_velocity_threshold
    return result


def object_is_above_height(
    env: IsaacLabArenaManagerBasedRLEnv,
    object_name: str,
    surface_height: float | None = None,
    use_settled_state: bool = False,
    distance: float = 1e-2,
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

    object_z = env.arena_world.get_position_w(object_name)[:, 2]
    if use_settled_state:
        settled_pos, has_settled = get_object_initial_rest_state(env, object_name)
        result = has_settled & (object_z > (settled_pos[:, 2] + distance))
    else:
        result = object_z > (surface_height + distance)
    return result


def object_moving(
    env: IsaacLabArenaManagerBasedRLEnv,
    object_name: str,
    velocity_threshold: float = 1e-2,
) -> torch.Tensor:
    """Check whether an object is moving above a velocity threshold.

    Returns True when object_name's linear speed exceeds velocity_threshold (m/s).
    """

    arena_world = env.arena_world
    object_mean_linear_velocity_w = arena_world.get_mean_linear_velocity_w(object_name)
    speed = torch.linalg.vector_norm(object_mean_linear_velocity_w, dim=-1)
    return speed > velocity_threshold


def objects_in_proximity(
    env: IsaacLabArenaManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    target_object_cfg: SceneEntityCfg,
    max_y_separation: float,
    max_x_separation: float,
    max_z_separation: float,
) -> torch.Tensor:
    """Determine if two objects are within a certain proximity of each other.

    Returns True when the object is within a certain proximity of the target object.
    """

    arena_world = env.arena_world
    object_position_w = arena_world.get_position_w(object_cfg.name)
    target_object_position_w = arena_world.get_position_w(target_object_cfg.name)

    # object to target object
    x_separation = torch.abs(object_position_w[:, 0] - target_object_position_w[:, 0])
    y_separation = torch.abs(object_position_w[:, 1] - target_object_position_w[:, 1])
    z_separation = torch.abs(object_position_w[:, 2] - target_object_position_w[:, 2])

    done = x_separation < max_x_separation
    done = torch.logical_and(done, y_separation < max_y_separation)
    done = torch.logical_and(done, z_separation < max_z_separation)

    return done


def object_supported_by(
    object_vertices_pos_w: torch.Tensor,
    destination_bound: AxisAlignedBoundingBox,
    support_tolerance: float = 0.03,
    low_point_tolerance: float = 0.01,
    minimum_support_fraction: float = 0.5,
) -> torch.Tensor:
    """Check whether a large fraction of object's lowest vertices are close to the destination's top surface.

    This is a geometric-only implementation to replace contact-sensor based contact_force_is_upward_support.
    Use this for deformable objects which don't have contact sensor support yet, see
    https://github.com/isaac-sim/IsaacLab/issues/4410

    Args:
        object_vertices_pos_w: Object vertices in world frame ``W``.
            Shape is ``(num_envs, num_vertices, 3)``.
        destination_bound: Axis-aligned bounds of the destination object in ``W``.
        support_tolerance: Maximum vertical distance in meters between a object vertex
            and the destination top surface to count as supported.
        low_point_tolerance: Band above the lowest node height used to select
            bottom nodes for the support-fraction denominator.
        minimum_support_fraction: Minimum fraction of low nodes that must lie
            on the destination footprint near the top surface.

    Returns:
        One Boolean result per environment.
    """
    low_z = object_vertices_pos_w[..., 2].amin(dim=1, keepdim=True)
    low_mask = object_vertices_pos_w[..., 2] <= low_z + low_point_tolerance
    # TODO(qianl, 2026-09-15): use destination's vertices instead of AABB top surface/footprint for closeness check.
    near_top = torch.abs(object_vertices_pos_w[..., 2] - destination_bound.top_surface_z[:, None]) <= support_tolerance
    inside_footprint = (
        (object_vertices_pos_w[..., :2] >= destination_bound.min_point[:, None, :2])
        & (object_vertices_pos_w[..., :2] <= destination_bound.max_point[:, None, :2])
    ).all(dim=-1)
    supported_points = low_mask & near_top & inside_footprint
    support_fraction = supported_points.sum(dim=1) / low_mask.sum(dim=1).clamp_min(1)
    return support_fraction >= minimum_support_fraction


def object_on_destination(
    env: IsaacLabArenaManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    destination_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg | None,
    force_threshold: float,
    velocity_threshold: float,
    support_cone_half_angle_rad: float = math.pi / 4,
) -> torch.Tensor:
    """Check whether an object is stably placed on its destination.

    The object's spawned-bounds center must be over the destination footprint
    and above its bottom, and the object's linear speed must be below the configured threshold.
    Rigid objects must also have upward support force. Deformable objects must have enough
    low nodal points near the destination's top surface. Destinations must be rooted objects.

    Args:
        env: The live Arena manager-based environment.
        object_cfg: The object being placed.
        destination_cfg: The rooted object or scene entry receiving the object.
        contact_sensor_cfg: The object's contact sensor filtered to the destination. None for a deformable object.
        force_threshold: Minimum upward support force in newtons.
        velocity_threshold: Maximum object linear speed in meters per second.
        support_cone_half_angle_rad: Maximum angle in radians from world ``+Z`` for the support force.

    Returns:
        One Boolean result per environment.
    """

    arena_world = env.arena_world
    assert (
        destination_cfg.name not in env.scene.deformable_objects
    ), "object_on_destination does not support deformable destinations"

    object_center_over_destination = object_bounds_center_over_destination(
        object_centroid_W=arena_world.get_centroid_w(object_cfg.name),
        T_W_D=arena_world.get_pose_w(destination_cfg.name),
        destination_bounds_D=arena_world.get_aabb_in_local_frame(destination_cfg.name),
    )

    if object_cfg.name in env.scene.deformable_objects:
        # Use geometric support for deformable objects.
        object_vertices_w = arena_world.get_vertices_w(object_cfg.name)
        destination_vertices_w = arena_world.get_vertices_w(destination_cfg.name)
        destination_bound = AxisAlignedBoundingBox(
            min_point=destination_vertices_w.amin(dim=1),
            max_point=destination_vertices_w.amax(dim=1),
        )
        destination_provides_upward_support = object_supported_by(
            object_vertices_pos_w=object_vertices_w,
            destination_bound=destination_bound,
        )
    else:
        # Use contact sensor for rigid objects.
        assert contact_sensor_cfg is not None, "object_on_destination requires a contact sensor for rigid objects"
        contact_sensor: ContactSensor = env.scene[contact_sensor_cfg.name]
        force_matrix_w = contact_sensor.data.force_matrix_w
        assert force_matrix_w is not None, f"Contact sensor '{contact_sensor_cfg.name}' has no filtered force matrix."
        force_matrix_w = force_matrix_w.torch
        assert force_matrix_w.shape == (env.num_envs, 1, 1, 3), (
            f"Contact sensor '{contact_sensor_cfg.name}' must provide one sensed body and one filtered body; "
            f"got force shape {tuple(force_matrix_w.shape)}."
        )
        # The two zeros select the sensor's single sensed body and single filtered destination body.
        support_force_on_object_w = force_matrix_w[:, 0, 0, :]
        destination_provides_upward_support = contact_force_is_upward_support(
            contact_force_w=support_force_on_object_w,
            force_threshold=force_threshold,
            support_cone_half_angle_rad=support_cone_half_angle_rad,
        )

    object_mean_linear_velocity_w = arena_world.get_mean_linear_velocity_w(object_cfg.name)
    object_moves_slowly = object_is_moving_slowly(object_mean_linear_velocity_w, velocity_threshold)
    return object_center_over_destination & destination_provides_upward_support & object_moves_slowly

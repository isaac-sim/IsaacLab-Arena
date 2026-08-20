# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import Literal

import warp as wp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.contact_sensor.contact_sensor import ContactSensor

from isaaclab_arena.scene.object_geometry import object_bounds_w
from isaaclab_arena.scene.object_state import object_state
from isaaclab_arena.tasks.predicates.object_settling import get_object_initial_rest_state
from isaaclab_arena.tasks.predicates.predicate_utils import get_env, get_root_lin_vel_w, get_root_pos_w, select

HeightMode = Literal["centroid", "min", "max", "min_point", "max_point"]


def object_is_above_height(
    env: ManagerBasedRLEnv,
    object_name: str,
    surface_height: float | None = None,
    use_settled_state: bool = False,
    distance: float = 1e-2,
    env_id: int | None = None,
    height_mode: HeightMode = "centroid",
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

    object_z = _object_height(env, object_name, height_mode)
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

    # Get positions relative to environment origin
    object_pos = get_root_pos_w(env, object_cfg.name) - env.scene.env_origins
    target_object_pos = get_root_pos_w(env, target_object_cfg.name) - env.scene.env_origins

    # object to target object
    x_separation = torch.abs(object_pos[:, 0] - target_object_pos[:, 0])
    y_separation = torch.abs(object_pos[:, 1] - target_object_pos[:, 1])
    z_separation = torch.abs(object_pos[:, 2] - target_object_pos[:, 2])

    done = x_separation < max_x_separation
    done = torch.logical_and(done, y_separation < max_y_separation)
    done = torch.logical_and(done, z_separation < max_z_separation)

    return done


def object_is_below_height(
    env: ManagerBasedRLEnv,
    object_name: str,
    minimum_height: float,
    height_mode: HeightMode = "centroid",
) -> torch.Tensor:
    """Return whether the selected object height is below ``minimum_height``."""
    return _object_height(env, object_name, height_mode) < minimum_height


def object_supported_by(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    destination_cfg: SceneEntityCfg,
    support_tolerance: float = 0.03,
    low_point_tolerance: float = 0.01,
    minimum_support_fraction: float = 0.5,
) -> torch.Tensor:
    """Check deformable support using low nodal points and destination bounds."""
    points = object_state(env, object_cfg.name).points
    assert points is not None, f"Object {object_cfg.name!r} has no deformable point state"
    assert points.position_w is not None, f"Object {object_cfg.name!r} has no nodal positions"

    position_w = points.position_w
    low_z = position_w[..., 2].amin(dim=1, keepdim=True)
    low_mask = position_w[..., 2] <= low_z + low_point_tolerance
    destination_bounds = object_bounds_w(env, destination_cfg.name)

    inside_xy = torch.all(
        (position_w[..., :2] >= destination_bounds.min_point[:, None, :2])
        & (position_w[..., :2] <= destination_bounds.max_point[:, None, :2]),
        dim=-1,
    )
    near_top = torch.abs(position_w[..., 2] - destination_bounds.top_surface_z[:, None]) <= support_tolerance
    supported_points = low_mask & inside_xy & near_top
    support_fraction = supported_points.sum(dim=1) / low_mask.sum(dim=1).clamp_min(1)
    return support_fraction >= minimum_support_fraction


def object_on_destination(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("pick_up_object"),
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("pick_up_object_contact_sensor"),
    force_threshold: float = 1.0,
    velocity_threshold: float = 0.5,
) -> torch.Tensor:
    """Checks if an object is in contact with it's destination location via a contact sensor.

    Returns True when the object is in contact with destination above a force threshold
    and below a velocity threshold.
    """

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
    force_above_threshold = force_matrix_norm > force_threshold

    velocity_w = get_root_lin_vel_w(env, object_cfg.name)
    velocity_w_norm = torch.norm(velocity_w, dim=-1)
    velocity_below_threshold = velocity_w_norm < velocity_threshold

    condition_met = torch.logical_and(force_above_threshold, velocity_below_threshold)

    return condition_met


def _object_height(env, object_name: str, height_mode: HeightMode) -> torch.Tensor:
    """Reduce aggregate or nodal world height with the requested semantics."""
    state = object_state(env, object_name)
    assert state.aggregate.position_w is not None, f"Object {object_name!r} has no aggregate position"
    if height_mode == "centroid":
        return state.aggregate.position_w[:, 2]

    kinematics = state.points if state.points is not None else state.aggregate
    assert kinematics.position_w is not None, f"Object {object_name!r} has no positions"
    point_z = kinematics.position_w[..., 2]
    if state.points is None:
        return point_z
    if height_mode in ("min", "min_point"):
        return point_z.amin(dim=1)
    if height_mode in ("max", "max_point"):
        return point_z.amax(dim=1)
    raise ValueError(f"Unsupported height mode: {height_mode!r}")


def objects_on_destinations(
    env: ManagerBasedRLEnv,
    object_cfg_list: list[SceneEntityCfg] = [SceneEntityCfg("pick_up_object")],
    contact_sensor_cfg_list: list[SceneEntityCfg] = [SceneEntityCfg("pick_up_object_contact_sensor")],
    force_threshold: float = 1.0,
    velocity_threshold: float = 0.5,
) -> torch.Tensor:
    """Multi-object version of `object_on_destination`.

    Returns True only when ALL objects in the list satisfy the destination condition.
    See `object_on_destination` for details on the single-object logic.
    """

    assert len(object_cfg_list) == len(contact_sensor_cfg_list), (
        "object_cfg_list and contact_sensor_cfg_list must have equal length, got "
        f"{len(object_cfg_list)} objects and {len(contact_sensor_cfg_list)} sensors"
    )

    unwrapped_env = get_env(env)
    condition_met = torch.ones((unwrapped_env.num_envs), device=unwrapped_env.device, dtype=torch.bool)
    for object_cfg, contact_sensor_cfg in zip(object_cfg_list, contact_sensor_cfg_list):
        single_condition = object_on_destination(
            env=env,
            object_cfg=object_cfg,
            contact_sensor_cfg=contact_sensor_cfg,
            force_threshold=force_threshold,
            velocity_threshold=velocity_threshold,
        )
        condition_met = torch.logical_and(condition_met, single_condition)
    return condition_met

# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors.contact_sensor.contact_sensor import ContactSensor
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

from isaaclab_arena.tasks.predicates.live_scene_geometry import (
    PoseFrameAabb,
    build_entity_pose_frame_aabbs,
    get_entity_pose_w,
)
from isaaclab_arena.tasks.predicates.object_settling import get_object_initial_rest_state
from isaaclab_arena.tasks.predicates.predicate_utils import (
    get_env,
    get_root_lin_vel_w,
    get_root_pos_w,
    runtime_buffer_to_torch,
    select,
)


def object_bounds_center_over_destination(
    object_pose_w: torch.Tensor,
    object_bounds: PoseFrameAabb,
    destination_pose_w: torch.Tensor,
    destination_bounds: PoseFrameAabb,
) -> torch.Tensor:
    """Check whether an object's bounds center is over a destination.

    The check requires the object center to be inside the destination's X/Y
    footprint and above its lower Z bound. The upper Z bound is intentionally
    ignored so the same check works for open containers and supporting surfaces.
    This is a center-point test, not full-object containment.

    Args:
        object_pose_w: Object poses in world coordinates with shape
            ``(num_envs, 7)`` and quaternion order ``(x, y, z, w)``.
        object_bounds: Object bounds relative to ``object_pose_w``.
        destination_pose_w: Destination poses in world coordinates with shape
            ``(num_envs, 7)`` and quaternion order ``(x, y, z, w)``.
        destination_bounds: Destination bounds relative to
            ``destination_pose_w``.

    Returns:
        One Boolean result per environment.
    """
    object_bounds_center_w, _ = combine_frame_transforms(
        object_pose_w[:, :3],
        object_pose_w[:, 3:],
        object_bounds.center,
    )
    object_bounds_center_in_destination, _ = subtract_frame_transforms(
        destination_pose_w[:, :3],
        destination_pose_w[:, 3:],
        object_bounds_center_w,
    )
    center_inside_horizontal_bounds = (
        (object_bounds_center_in_destination[:, :2] >= destination_bounds.lower[:, :2])
        & (object_bounds_center_in_destination[:, :2] <= destination_bounds.upper[:, :2])
    ).all(dim=-1)
    center_above_destination_bottom = object_bounds_center_in_destination[:, 2] >= destination_bounds.lower[:, 2]
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


class GeometricObjectOnDestinationTerm(ManagerTermBase):
    """Check object placement using cached spawned geometry and current state.

    Construction reads and caches the object and destination bounds. Each call
    combines those bounds with current poses, filtered contact force, and object
    velocity. The object bounds center must be over the destination footprint,
    the contact force must point upward, and object speed must be below the
    configured threshold.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        object_cfg: SceneEntityCfg = cfg.params["object_cfg"]
        destination_cfg: SceneEntityCfg = cfg.params["destination_cfg"]
        force_threshold: float = cfg.params["force_threshold"]
        velocity_threshold: float = cfg.params["velocity_threshold"]
        support_cone_half_angle_deg: float = cfg.params.get("support_cone_half_angle_deg", 45.0)

        assert (
            object_cfg.name in env.scene.rigid_objects
        ), f"GeometricObjectOnDestinationTerm requires rigid object '{object_cfg.name}'."
        assert force_threshold >= 0.0, f"force_threshold must be non-negative, got {force_threshold}."
        assert velocity_threshold >= 0.0, f"velocity_threshold must be non-negative, got {velocity_threshold}."
        assert (
            0.0 <= support_cone_half_angle_deg < 90.0
        ), f"support_cone_half_angle_deg must be in [0, 90), got {support_cone_half_angle_deg}."

        self._object_name = object_cfg.name
        self._destination_name = destination_cfg.name
        self._object_bounds = build_entity_pose_frame_aabbs(env, self._object_name)
        self._destination_bounds = build_entity_pose_frame_aabbs(env, self._destination_name)

    def __call__(
        self,
        env: ManagerBasedEnv,
        object_cfg: SceneEntityCfg,
        destination_cfg: SceneEntityCfg,
        contact_sensor_cfg: SceneEntityCfg,
        force_threshold: float,
        velocity_threshold: float,
        support_cone_half_angle_deg: float = 45.0,
    ) -> torch.Tensor:
        """Return which environments currently satisfy the geometric placement check."""
        assert (
            object_cfg.name == self._object_name
        ), f"This term cached geometry for object '{self._object_name}', but was called with '{object_cfg.name}'."
        assert destination_cfg.name == self._destination_name, (
            f"This term cached geometry for destination '{self._destination_name}', "
            f"but was called with '{destination_cfg.name}'."
        )

        object_pose_w = get_entity_pose_w(env, self._object_name)
        destination_pose_w = get_entity_pose_w(env, self._destination_name)
        object_center_over_destination = object_bounds_center_over_destination(
            object_pose_w=object_pose_w,
            object_bounds=self._object_bounds,
            destination_pose_w=destination_pose_w,
            destination_bounds=self._destination_bounds,
        )

        contact_sensor: ContactSensor = env.scene[contact_sensor_cfg.name]
        force_matrix_w = contact_sensor.data.force_matrix_w
        assert force_matrix_w is not None, f"Contact sensor '{contact_sensor_cfg.name}' has no filtered force matrix."
        force_matrix_w = runtime_buffer_to_torch(force_matrix_w)
        assert force_matrix_w.shape == (env.num_envs, 1, 1, 3), (
            f"Contact sensor '{contact_sensor_cfg.name}' must provide one sensed body and one filtered body; "
            f"got force shape {tuple(force_matrix_w.shape)}."
        )
        support_force_on_object_w = force_matrix_w[:, 0, 0, :]
        destination_provides_upward_support = contact_force_is_upward_support(
            contact_force_w=support_force_on_object_w,
            force_threshold=force_threshold,
            support_cone_half_angle_deg=support_cone_half_angle_deg,
        )

        object_entity: RigidObject = env.scene[self._object_name]
        object_linear_velocity_w = runtime_buffer_to_torch(object_entity.data.root_lin_vel_w)
        object_is_moving_slowly = torch.linalg.vector_norm(object_linear_velocity_w, dim=-1) < velocity_threshold
        return object_center_over_destination & destination_provides_upward_support & object_is_moving_slowly


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

    root_state_entity_names = env.scene.rigid_objects.keys() | env.scene.articulations.keys()
    assert (
        object_cfg.name in root_state_entity_names
    ), f"objects_in_proximity requires a rigid object or articulation root, got '{object_cfg.name}'."
    assert (
        target_object_cfg.name in root_state_entity_names
    ), f"objects_in_proximity requires a rigid target or articulation root, got '{target_object_cfg.name}'."

    # Get object entities from the scene
    object_entity: RigidObject | Articulation = env.scene[object_cfg.name]
    target_object_entity: RigidObject | Articulation = env.scene[target_object_cfg.name]

    # Get positions relative to environment origin
    object_pos = runtime_buffer_to_torch(object_entity.data.root_pos_w) - env.scene.env_origins
    target_object_pos = runtime_buffer_to_torch(target_object_entity.data.root_pos_w) - env.scene.env_origins

    x_separation = torch.abs(object_pos[:, 0] - target_object_pos[:, 0])
    y_separation = torch.abs(object_pos[:, 1] - target_object_pos[:, 1])
    z_separation = torch.abs(object_pos[:, 2] - target_object_pos[:, 2])

    within_max_separation = x_separation < max_x_separation
    within_max_separation &= y_separation < max_y_separation
    within_max_separation &= z_separation < max_z_separation
    return within_max_separation


def object_on_destination(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("pick_up_object"),
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("pick_up_object_contact_sensor"),
    force_threshold: float = 1.0,
    velocity_threshold: float = 0.5,
) -> torch.Tensor:
    """Check destination-filtered contact magnitude and object linear speed.

    This compatibility predicate does not check the object's position or contact
    direction. Use ``GeometricObjectOnDestinationTerm`` when placement
    geometry is required, such as for task success.

    Returns True when filtered contact exceeds ``force_threshold`` and object
    speed is below ``velocity_threshold``.
    """

    unwrapped_env = get_env(env)
    object_entity: RigidObject = unwrapped_env.scene[object_cfg.name]
    contact_sensor: ContactSensor = unwrapped_env.scene[contact_sensor_cfg.name]

    # force_matrix_w shape is (N, B, M, 3), where N is the number of sensors, B is number of bodies in each sensor
    # and ``M`` is the number of filtered bodies.
    # We assume B = 1 and M = 1
    assert contact_sensor.data.force_matrix_w.shape[2] == 1
    assert contact_sensor.data.force_matrix_w.shape[1] == 1
    # NOTE(alexmillane, 2025-08-04): We expect the binary flags to have shape (N, )
    # where N is the number of envs.
    force_matrix_norm = torch.norm(runtime_buffer_to_torch(contact_sensor.data.force_matrix_w), dim=-1).reshape(-1)
    force_above_threshold = force_matrix_norm > force_threshold

    velocity_w = runtime_buffer_to_torch(object_entity.data.root_lin_vel_w)
    velocity_w_norm = torch.norm(velocity_w, dim=-1)
    velocity_below_threshold = velocity_w_norm < velocity_threshold

    condition_met = torch.logical_and(force_above_threshold, velocity_below_threshold)

    return condition_met


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

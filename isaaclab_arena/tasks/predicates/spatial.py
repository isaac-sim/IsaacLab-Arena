# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import numpy as np
import torch

import isaaclab.sim as sim_utils
import warp as wp
from isaaclab.assets import RigidObject
from isaaclab.cloner.cloner_utils import iter_clone_plan_matches
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors.contact_sensor.contact_sensor import ContactSensor
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from isaaclab_arena.tasks.predicates.object_settling import get_object_initial_rest_state
from isaaclab_arena.tasks.predicates.predicate_utils import get_env, get_root_lin_vel_w, get_root_pos_w, select


def _to_torch(value) -> torch.Tensor:
    """Return a Torch view of an Isaac Lab runtime buffer."""
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "torch"):
        return value.torch
    return wp.to_torch(value)


def _find_source_prims(stage: Usd.Stage, prim_path_expression: str) -> list[Usd.Prim]:
    """Resolve a clone source path, including paths inside instanceable assets."""
    exact_prim = stage.GetPrimAtPath(prim_path_expression)
    if exact_prim.IsValid():
        return [exact_prim]
    return sim_utils.find_matching_prims(prim_path_expression, stage=stage)


def _get_environment_id_from_prim_path(env: ManagerBasedEnv, prim_path: str) -> int | None:
    """Return the environment containing a prim path, or None for a global prim."""
    for environment_id, environment_prim_path in enumerate(env.scene.env_prim_paths):
        if prim_path == environment_prim_path or prim_path.startswith(f"{environment_prim_path}/"):
            return environment_id
    return None


def _get_source_prim_groups(env: ManagerBasedEnv, entity_name: str) -> list[tuple[Usd.Prim, tuple[int, ...]]]:
    """Return one representative prim and the environments that use it."""
    scene = env.scene
    entity_cfg = getattr(scene.cfg, entity_name)
    prim_path_expression = entity_cfg.prim_path
    clone_plan = scene.clone_plan

    if clone_plan is not None:
        clone_matches = tuple(iter_clone_plan_matches(clone_plan, prim_path_expression))
        if clone_matches:
            source_prim_groups = []
            for _, _, source_path, environment_ids in clone_matches:
                source_prims = _find_source_prims(scene.stage, source_path)
                assert len(source_prims) == 1, (
                    f"Scene entity '{entity_name}' source '{source_path}' resolved to {len(source_prims)} prims; "
                    "expected exactly one."
                )
                source_prim_groups.append((source_prims[0], environment_ids))
            return source_prim_groups

    matching_prims = sim_utils.find_matching_prims(prim_path_expression, stage=scene.stage)
    assert matching_prims, f"Scene entity '{entity_name}' has no USD prim matching '{prim_path_expression}'."
    if len(matching_prims) == 1:
        return [(matching_prims[0], tuple(range(env.num_envs)))]

    prims_by_environment: list[list[Usd.Prim]] = [[] for _ in range(env.num_envs)]
    for prim in matching_prims:
        prim_path = prim.GetPath().pathString
        for environment_id, environment_prim_path in enumerate(scene.env_prim_paths):
            if prim_path == environment_prim_path or prim_path.startswith(f"{environment_prim_path}/"):
                prims_by_environment[environment_id].append(prim)
                break

    source_prim_groups = []
    for environment_id, environment_prims in enumerate(prims_by_environment):
        assert len(environment_prims) == 1, (
            f"Scene entity '{entity_name}' resolved to {len(environment_prims)} prims in environment "
            f"{environment_id}; expected exactly one."
        )
        source_prim_groups.append((environment_prims[0], (environment_id,)))
    return source_prim_groups


def _find_pose_frame_prim(source_prim: Usd.Prim, is_rigid: bool, entity_name: str) -> Usd.Prim:
    """Find the USD prim whose pose matches the live entity pose."""
    if not is_rigid:
        return source_prim

    rigid_body_prims = sim_utils.get_all_matching_child_prims(
        source_prim.GetPath(),
        predicate=lambda prim: prim.HasAPI(UsdPhysics.RigidBodyAPI),
        stage=source_prim.GetStage(),
        traverse_instance_prims=False,
    )
    assert len(rigid_body_prims) == 1, (
        f"Rigid scene entity '{entity_name}' source '{source_prim.GetPath()}' contains "
        f"{len(rigid_body_prims)} rigid bodies; expected exactly one."
    )
    return rigid_body_prims[0]


def _compute_pose_frame_aabb(pose_frame_prim: Usd.Prim, entity_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Compute a live USD bound relative to the prim's translation and rotation."""
    transform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    pose_to_world = transform_cache.GetLocalToWorldTransform(pose_frame_prim).RemoveScaleShear()
    world_to_pose = pose_to_world.GetInverse()
    bounding_box_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

    pose_frame_points: list[tuple[float, float, float]] = []
    for geometry_prim in Usd.PrimRange(pose_frame_prim, Usd.TraverseInstanceProxies()):
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
        for x_coordinate in (lower[0], upper[0]):
            for y_coordinate in (lower[1], upper[1]):
                for z_coordinate in (lower[2], upper[2]):
                    point_w = geometry_to_world.Transform(Gf.Vec3d(x_coordinate, y_coordinate, z_coordinate))
                    point_pose = world_to_pose.Transform(point_w)
                    pose_frame_points.append((point_pose[0], point_pose[1], point_pose[2]))

    assert (
        pose_frame_points
    ), f"Scene entity '{entity_name}' has no default-purpose geometry below '{pose_frame_prim.GetPath()}'."
    points = np.asarray(pose_frame_points, dtype=np.float64)
    return points.min(axis=0), points.max(axis=0)


def _get_pose_frame_aabb(env: ManagerBasedEnv, entity_cfg: SceneEntityCfg) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one pose-frame AABB per parallel environment from the live USD scene."""
    entity_name = entity_cfg.name
    scene = env.scene
    scene_entity_cfg = getattr(scene.cfg, entity_name)
    assert (
        entity_name in scene.rigid_objects or entity_name in scene.extras
    ), f"Scene entity '{entity_name}' must be a rigid object or static reference for geometric placement checks."
    is_rigid = entity_name in scene.rigid_objects

    lower_by_environment = torch.empty((env.num_envs, 3), dtype=torch.float32, device=env.device)
    upper_by_environment = torch.empty_like(lower_by_environment)
    coverage_count = torch.zeros(env.num_envs, dtype=torch.int32)
    expected_pose_path_by_environment: list[str | None] = [None] * env.num_envs

    for source_prim, environment_ids in _get_source_prim_groups(env, entity_name):
        pose_frame_prim = _find_pose_frame_prim(source_prim, is_rigid=is_rigid, entity_name=entity_name)
        lower, upper = _compute_pose_frame_aabb(pose_frame_prim, entity_name=entity_name)
        environment_ids_tensor = torch.tensor(environment_ids, dtype=torch.long, device=env.device)
        lower_by_environment[environment_ids_tensor] = torch.as_tensor(lower, dtype=torch.float32, device=env.device)
        upper_by_environment[environment_ids_tensor] = torch.as_tensor(upper, dtype=torch.float32, device=env.device)
        coverage_count[list(environment_ids)] += 1

        if is_rigid:
            source_prim_path = source_prim.GetPath().pathString
            pose_frame_prim_path = pose_frame_prim.GetPath().pathString
            assert pose_frame_prim_path == source_prim_path or pose_frame_prim_path.startswith(
                f"{source_prim_path}/"
            ), f"Rigid scene entity '{entity_name}' pose frame is not below its source prim."
            pose_frame_suffix = pose_frame_prim_path[len(source_prim_path) :]
            for environment_id in environment_ids:
                environment_asset_path = scene_entity_cfg.prim_path.replace(
                    scene.env_regex_ns, scene.env_prim_paths[environment_id]
                )
                expected_pose_path_by_environment[environment_id] = environment_asset_path + pose_frame_suffix

    assert torch.all(coverage_count == 1), (
        f"Scene entity '{entity_name}' geometry must cover every environment exactly once; "
        f"got coverage {coverage_count.tolist()}."
    )

    if is_rigid:
        live_pose_paths = getattr(scene[entity_name].root_view, "prim_paths", None)
        if live_pose_paths is not None:
            assert len(live_pose_paths) == env.num_envs, (
                f"Rigid scene entity '{entity_name}' has {len(live_pose_paths)} live pose paths; "
                f"expected {env.num_envs}."
            )
            live_pose_paths = [str(path) for path in live_pose_paths]
            assert live_pose_paths == expected_pose_path_by_environment, (
                f"Rigid scene entity '{entity_name}' live pose paths do not match the geometry pose frames. "
                f"Expected {expected_pose_path_by_environment}, got {live_pose_paths}."
            )
    return lower_by_environment, upper_by_environment


def _get_static_reference_pose_w(env: ManagerBasedEnv, entity_name: str) -> torch.Tensor:
    """Read static-reference poses and arrange them in environment order."""
    entity = env.scene[entity_name]
    position_w_buffer, orientation_w_buffer = entity.get_world_poses()
    position_w = _to_torch(position_w_buffer)
    orientation_w = _to_torch(orientation_w_buffer)
    assert (
        position_w.ndim == 2 and position_w.shape[1] == 3
    ), f"Static reference '{entity_name}' returned position shape {tuple(position_w.shape)}; expected (num_poses, 3)."
    assert orientation_w.shape == (position_w.shape[0], 4), (
        f"Static reference '{entity_name}' returned orientation shape {tuple(orientation_w.shape)}; "
        f"expected ({position_w.shape[0]}, 4)."
    )

    prim_paths = [str(path) for path in getattr(entity, "prim_paths", ())]
    prim_environment_ids = [_get_environment_id_from_prim_path(env, prim_path) for prim_path in prim_paths]
    if prim_paths and all(environment_id is None for environment_id in prim_environment_ids):
        assert len(prim_paths) == 1 and position_w.shape[0] == 1, (
            f"Global static reference '{entity_name}' must provide exactly one pose, got "
            f"{len(prim_paths)} paths and {position_w.shape[0]} poses."
        )
        return torch.cat(
            (position_w.expand(env.num_envs, 3), orientation_w.expand(env.num_envs, 4)),
            dim=-1,
        )

    if prim_paths:
        assert all(
            environment_id is not None for environment_id in prim_environment_ids
        ), f"Static reference '{entity_name}' mixes global and environment-scoped pose paths."

    if len(prim_paths) == env.num_envs:
        assert position_w.shape[0] == env.num_envs, (
            f"Static reference '{entity_name}' returned {position_w.shape[0]} poses for "
            f"{len(prim_paths)} environment paths."
        )
        row_by_environment: list[int | None] = [None] * env.num_envs
        for row_index, environment_id in enumerate(prim_environment_ids):
            assert environment_id is not None, f"Static reference '{entity_name}' has an unmapped pose path."
            assert (
                row_by_environment[environment_id] is None
            ), f"Static reference '{entity_name}' has more than one pose for environment {environment_id}."
            row_by_environment[environment_id] = row_index
        assert all(
            row_index is not None for row_index in row_by_environment
        ), f"Static reference '{entity_name}' does not provide exactly one pose per environment."
        row_indices = torch.tensor(row_by_environment, dtype=torch.long, device=position_w.device)
        return torch.cat((position_w.index_select(0, row_indices), orientation_w.index_select(0, row_indices)), dim=-1)

    if not prim_paths:
        assert position_w.shape[0] == env.num_envs, (
            f"Static reference '{entity_name}' returned {position_w.shape[0]} poses without corresponding prim paths; "
            f"expected {env.num_envs}."
        )
        return torch.cat((position_w, orientation_w), dim=-1)

    assert (
        len(prim_paths) == position_w.shape[0]
    ), f"Static reference '{entity_name}' returned {position_w.shape[0]} poses for {len(prim_paths)} prim paths."
    pose_row_by_path = {prim_path: row_index for row_index, prim_path in enumerate(prim_paths)}
    assert len(pose_row_by_path) == len(prim_paths), f"Static reference '{entity_name}' has duplicate pose paths."

    dense_position_w = torch.empty((env.num_envs, 3), dtype=position_w.dtype, device=position_w.device)
    dense_orientation_w = torch.empty((env.num_envs, 4), dtype=orientation_w.dtype, device=orientation_w.device)
    coverage_count = torch.zeros(env.num_envs, dtype=torch.int32)
    environment_origins = env.scene.env_origins.to(device=position_w.device, dtype=position_w.dtype)
    for source_prim, environment_ids in _get_source_prim_groups(env, entity_name):
        source_prim_path = source_prim.GetPath().pathString
        assert (
            source_prim_path in pose_row_by_path
        ), f"Static reference '{entity_name}' has no live pose row for clone source '{source_prim_path}'."
        pose_row = pose_row_by_path[source_prim_path]
        source_environment_id = _get_environment_id_from_prim_path(env, source_prim_path)
        environment_ids_tensor = torch.tensor(environment_ids, dtype=torch.long, device=position_w.device)
        if source_environment_id is None:
            dense_position_w[environment_ids_tensor] = position_w[pose_row]
        else:
            source_position_env = position_w[pose_row] - environment_origins[source_environment_id]
            dense_position_w[environment_ids_tensor] = source_position_env + environment_origins[environment_ids_tensor]
        dense_orientation_w[environment_ids_tensor] = orientation_w[pose_row]
        coverage_count[list(environment_ids)] += 1

    assert torch.all(coverage_count == 1), (
        f"Static reference '{entity_name}' poses must cover every environment exactly once; "
        f"got coverage {coverage_count.tolist()}."
    )
    return torch.cat((dense_position_w, dense_orientation_w), dim=-1)


def _get_pose_w(env: ManagerBasedEnv, entity_cfg: SceneEntityCfg) -> torch.Tensor:
    """Read a rigid pose or static-reference pose in world coordinates."""
    entity_name = entity_cfg.name
    scene = env.scene
    entity = scene[entity_name]
    if entity_name in scene.rigid_objects:
        pose_w = _to_torch(entity.data.root_pose_w)
        assert pose_w.shape == (env.num_envs, 7), (
            f"Rigid scene entity '{entity_name}' returned pose shape {tuple(pose_w.shape)}; "
            f"expected ({env.num_envs}, 7)."
        )
        return pose_w

    assert (
        entity_name in scene.extras
    ), f"Scene entity '{entity_name}' must be a rigid object or static reference for pose lookup."
    return _get_static_reference_pose_w(env, entity_name)


def object_centroid_in_open_top_bounds(
    object_pose_w: torch.Tensor,
    object_aabb_lower: torch.Tensor,
    object_aabb_upper: torch.Tensor,
    destination_pose_w: torch.Tensor,
    destination_aabb_lower: torch.Tensor,
    destination_aabb_upper: torch.Tensor,
) -> torch.Tensor:
    """Check whether an object's AABB centroid is within open-top destination bounds."""
    object_centroid_pose = (object_aabb_lower + object_aabb_upper) * 0.5
    object_centroid_w, _ = combine_frame_transforms(
        object_pose_w[:, :3],
        object_pose_w[:, 3:],
        object_centroid_pose,
    )
    object_centroid_destination, _ = subtract_frame_transforms(
        destination_pose_w[:, :3],
        destination_pose_w[:, 3:],
        object_centroid_w,
    )
    inside_x = (object_centroid_destination[:, 0] >= destination_aabb_lower[:, 0]) & (
        object_centroid_destination[:, 0] <= destination_aabb_upper[:, 0]
    )
    inside_y = (object_centroid_destination[:, 1] >= destination_aabb_lower[:, 1]) & (
        object_centroid_destination[:, 1] <= destination_aabb_upper[:, 1]
    )
    above_bottom = object_centroid_destination[:, 2] >= destination_aabb_lower[:, 2]
    return inside_x & inside_y & above_bottom


def contact_force_is_upward_support(
    contact_force_w: torch.Tensor,
    force_threshold: float,
    support_cone_half_angle_deg: float,
) -> torch.Tensor:
    """Check whether a world-frame contact force supports an object upward."""
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


class ObjectOnDestinationTerm(ManagerTermBase):
    """Check geometric placement, upward support, and low motion using cached live-scene bounds."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        object_cfg: SceneEntityCfg = cfg.params["object_cfg"]
        destination_cfg: SceneEntityCfg = cfg.params["destination_cfg"]
        assert (
            object_cfg.name in env.scene.rigid_objects
        ), f"ObjectOnDestinationTerm requires rigid object '{object_cfg.name}'."
        self._object_aabb_lower, self._object_aabb_upper = _get_pose_frame_aabb(env, object_cfg)
        self._destination_aabb_lower, self._destination_aabb_upper = _get_pose_frame_aabb(env, destination_cfg)

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
        """Return whether the object is inside, supported by, and still on its destination."""
        object_pose_w = _get_pose_w(env, object_cfg)
        destination_pose_w = _get_pose_w(env, destination_cfg)
        inside_destination = object_centroid_in_open_top_bounds(
            object_pose_w=object_pose_w,
            object_aabb_lower=self._object_aabb_lower,
            object_aabb_upper=self._object_aabb_upper,
            destination_pose_w=destination_pose_w,
            destination_aabb_lower=self._destination_aabb_lower,
            destination_aabb_upper=self._destination_aabb_upper,
        )

        sensor: ContactSensor = env.scene[contact_sensor_cfg.name]
        force_matrix_w = sensor.data.force_matrix_w
        assert force_matrix_w is not None, f"Contact sensor '{contact_sensor_cfg.name}' has no filtered force matrix."
        force_matrix_w = _to_torch(force_matrix_w)
        assert force_matrix_w.shape == (env.num_envs, 1, 1, 3), (
            f"Contact sensor '{contact_sensor_cfg.name}' must provide one sensed body and one filtered body; "
            f"got force shape {tuple(force_matrix_w.shape)}."
        )
        supported_by_destination = contact_force_is_upward_support(
            contact_force_w=force_matrix_w[:, 0, 0, :],
            force_threshold=force_threshold,
            support_cone_half_angle_deg=support_cone_half_angle_deg,
        )

        object: RigidObject = env.scene[object_cfg.name]
        velocity_w = _to_torch(object.data.root_lin_vel_w)
        velocity_below_threshold = torch.linalg.vector_norm(velocity_w, dim=-1) < velocity_threshold
        return inside_destination & supported_by_destination & velocity_below_threshold


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
    object: RigidObject = unwrapped_env.scene[object_cfg.name]
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

    velocity_w = wp.to_torch(object.data.root_lin_vel_w)
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

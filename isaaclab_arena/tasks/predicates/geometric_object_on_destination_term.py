# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Manager-owned geometric placement term with cached spawned bounds.

Isaac Lab constructs the term once for a live environment, so it can reuse the
object and destination bounds. Stateless geometric checks remain in
``spatial.py``.
"""

from __future__ import annotations

import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors.contact_sensor.contact_sensor import ContactSensor

from isaaclab_arena.tasks.predicates.live_scene_geometry import LiveSceneEntityGeometry
from isaaclab_arena.tasks.predicates.predicate_utils import runtime_buffer_to_torch
from isaaclab_arena.tasks.predicates.spatial import (
    contact_force_is_upward_support,
    object_bounds_center_over_destination,
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
        destination_pose_cfg: SceneEntityCfg = cfg.params["destination_pose_cfg"]
        destination_prim_path: str = cfg.params["destination_prim_path"]
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
        self._destination_pose_entity_name = destination_pose_cfg.name
        self._destination_prim_path = destination_prim_path
        self._object_geometry = LiveSceneEntityGeometry(env, object_cfg)
        self._destination_geometry = LiveSceneEntityGeometry(
            env,
            destination_pose_cfg,
            geometry_prim_path=destination_prim_path,
        )

    def __call__(
        self,
        env: ManagerBasedEnv,
        object_cfg: SceneEntityCfg,
        destination_pose_cfg: SceneEntityCfg,
        destination_prim_path: str,
        contact_sensor_cfg: SceneEntityCfg,
        force_threshold: float,
        velocity_threshold: float,
        support_cone_half_angle_deg: float = 45.0,
    ) -> torch.Tensor:
        """Return which environments currently satisfy the geometric placement check."""
        assert (
            object_cfg.name == self._object_name
        ), f"This term cached geometry for object '{self._object_name}', but was called with '{object_cfg.name}'."
        assert destination_pose_cfg.name == self._destination_pose_entity_name, (
            f"This term cached geometry using destination pose entity '{self._destination_pose_entity_name}', "
            f"but was called with '{destination_pose_cfg.name}'."
        )
        assert destination_prim_path == self._destination_prim_path, (
            f"This term cached destination geometry at '{self._destination_prim_path}', "
            f"but was called with '{destination_prim_path}'."
        )

        object_pose_w = self._object_geometry.get_pose_w()
        destination_pose_w = self._destination_geometry.get_pose_w()
        object_center_over_destination = object_bounds_center_over_destination(
            object_pose_w=object_pose_w,
            object_bounds=self._object_geometry.bounds_in_live_pose_frame,
            destination_pose_w=destination_pose_w,
            destination_bounds=self._destination_geometry.bounds_in_live_pose_frame,
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

# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Manager-owned object-on-destination term with cached spawned bounds.

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

from isaaclab_arena.tasks.predicates.live_scene_geometry import (
    AssetBaseCfgPoseReader,
    compute_spawned_geometry_bounds_in_entity_frame,
)
from isaaclab_arena.tasks.predicates.spatial import (
    contact_force_is_upward_support,
    object_bounds_center_over_destination,
    object_is_moving_slowly,
)


# TODO(cvolk, 2026-08-24): [arena-world-migration] Replace term-owned geometry caching and direct live-state reads
# with env.arena_world queries, then make this a plain predicate function.
class ObjectOnDestinationTerm(ManagerTermBase):
    """Check object placement using cached spawned geometry and current state.

    Construction caches the object bounds center and destination bounds in
    their respective entity frames. Each call combines them with current poses,
    filtered contact force, and object velocity. The object bounds center must
    be over the destination footprint, the contact force must point upward, and
    object speed must be below the configured threshold.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        object_cfg: SceneEntityCfg = cfg.params["object_cfg"]
        destination_cfg: SceneEntityCfg = cfg.params["destination_cfg"]

        assert (
            object_cfg.name in env.scene.rigid_objects
        ), f"ObjectOnDestinationTerm requires rigid object '{object_cfg.name}'."

        self._object_name = object_cfg.name
        self._destination_name = destination_cfg.name
        self._object_rigid_object: RigidObject = env.scene[self._object_name]
        object_bounds_O = compute_spawned_geometry_bounds_in_entity_frame(env, object_cfg)
        self._object_bounds_center_O = object_bounds_O.center
        self._destination_bounds_D = compute_spawned_geometry_bounds_in_entity_frame(env, destination_cfg)
        self._destination_rigid_object: RigidObject | None = None
        self._destination_asset_base_pose_reader: AssetBaseCfgPoseReader | None = None
        if self._destination_name in env.scene.rigid_objects:
            # RigidObjectCfg destinations expose their pose through rigid-body state.
            self._destination_rigid_object = env.scene[self._destination_name]
        elif self._destination_name in env.scene.extras:
            # AssetBaseCfg destinations need a FrameView created after environment cloning.
            self._destination_asset_base_pose_reader = AssetBaseCfgPoseReader(env, self._destination_name)
        else:
            unsupported_destination_message = (
                f"Destination '{self._destination_name}' must be a RigidObjectCfg or AssetBaseCfg scene entry."
            )
            assert False, unsupported_destination_message

    def close(self) -> None:
        """Release the frame view owned by this term. Safe to call more than once."""
        if self._destination_asset_base_pose_reader is not None:
            self._destination_asset_base_pose_reader.close()
            self._destination_asset_base_pose_reader = None

    def __del__(self):
        """Release the frame view when the termination manager drops this term."""
        # TODO(amillane, 2026-08-31): Move this upstream post v0.3.0 -- ManagerTermBase should
        # grow a teardown hook that each manager calls for its own terms, so resource-owning
        # terms no longer need their own destructor.
        self.close()

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

        T_W_O = self._object_rigid_object.data.root_pose_w.torch
        T_W_D = self._get_destination_pose_in_world_frame()
        object_center_over_destination = object_bounds_center_over_destination(
            T_W_O=T_W_O,
            object_bounds_center_O=self._object_bounds_center_O,
            T_W_D=T_W_D,
            destination_bounds_D=self._destination_bounds_D,
        )

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
            support_cone_half_angle_deg=support_cone_half_angle_deg,
        )

        object_linear_velocity_w = self._object_rigid_object.data.root_lin_vel_w.torch
        object_moves_slowly = object_is_moving_slowly(object_linear_velocity_w, velocity_threshold)
        return object_center_over_destination & destination_provides_upward_support & object_moves_slowly

    def _get_destination_pose_in_world_frame(self) -> torch.Tensor:
        """Read the pose from rigid-body state or an AssetBaseCfg scene entry."""
        if self._destination_rigid_object is not None:
            return self._destination_rigid_object.data.root_pose_w.torch

        assert self._destination_asset_base_pose_reader is not None
        return self._destination_asset_base_pose_reader.get_pose_in_world_frame()

# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Manager adapter for the geometric object-on-destination predicate."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors.contact_sensor.contact_sensor import ContactSensor

from isaaclab_arena.tasks.predicates.spatial import (
    contact_force_is_upward_support,
    object_bounds_center_over_destination,
    object_is_moving_slowly,
)


class ObjectOnDestinationTerm(ManagerTermBase):
    """Check object placement using ArenaWorld geometry and current state."""

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
        arena_world = env.arena_world
        T_W_O = arena_world.get_pose_w(object_cfg.name)
        T_W_D = arena_world.get_pose_w(destination_cfg.name)
        object_center_over_destination = object_bounds_center_over_destination(
            T_W_O=T_W_O,
            object_bounds_center_O=arena_world.get_local_aabb(object_cfg.name).center,
            T_W_D=T_W_D,
            destination_bounds_D=arena_world.get_local_aabb(destination_cfg.name),
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

        object_linear_velocity_w = arena_world.get_linear_velocity_w(object_cfg.name)
        object_moves_slowly = object_is_moving_slowly(object_linear_velocity_w, velocity_threshold)
        return object_center_over_destination & destination_provides_upward_support & object_moves_slowly

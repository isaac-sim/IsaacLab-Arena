# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

import warp as wp
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

from isaaclab_arena.affordances.affordance_base import AffordanceBase


class Movable(AffordanceBase):
    """Interface for objects that can be pushed or moved across the floor.

    Movable objects are characterized by the ability to be displaced from their
    initial position (e.g. carts on wheels, trolleys, wheeled furniture).
    Displacement is measured in the XY-plane (horizontal movement only).
    """

    def __init__(self, displacement_threshold: float = 0.1, **kwargs):
        """Initialize a movable object.

        Args:
            displacement_threshold: Default threshold (meters) for determining
                whether the object has been moved. Measured as Euclidean
                distance in the XY-plane.
            **kwargs: Additional arguments passed to AffordanceBase.
        """
        super().__init__(**kwargs)
        self.displacement_threshold = displacement_threshold

    def get_displacement(self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
        """Get horizontal (XY) displacement from the initial position.

        Args:
            env: The environment instance.
            asset_cfg: Asset configuration. If None, uses the object's name.

        Returns:
            Euclidean distance in the XY-plane from the initial position.
            Shape: [num_envs].
        """
        if asset_cfg is None:
            asset_cfg = SceneEntityCfg(self.name)
        object_entity = env.scene[asset_cfg.name]
        current_pos = wp.to_torch(object_entity.data.root_pos_w)[:, :2]
        initial_pos = wp.to_torch(object_entity.data.default_root_state)[:, :2]
        env_origins_xy = env.scene.env_origins[:, :2]
        return torch.norm(current_pos - (initial_pos + env_origins_xy), dim=-1)

    def is_moved(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg | None = None,
        displacement_threshold: float | None = None,
    ) -> torch.Tensor:
        """Check whether the object has been displaced beyond a threshold.

        Args:
            env: The environment instance.
            asset_cfg: Asset configuration. If None, uses the object's name.
            displacement_threshold: Threshold in meters. If None, uses the
                object's default displacement_threshold.

        Returns:
            Boolean tensor indicating whether the object has been moved.
            Shape: [num_envs].
        """
        if displacement_threshold is None:
            displacement_threshold = self.displacement_threshold
        displacement = self.get_displacement(env, asset_cfg)
        return displacement > displacement_threshold

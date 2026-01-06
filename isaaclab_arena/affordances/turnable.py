# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math

import torch

from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

from isaaclab_arena.affordances.affordance_base import AffordanceBase
from isaaclab_arena.utils.joint_utils import \
    (get_unnormalized_joint_position, normalize_value, set_unnormalized_joint_position)


class Turnable(AffordanceBase):
    """
    Interface for turnable objects with discrete levels.

    Examples: controller knob/dial/rotary switch to adjust volume/speed/power/temperature etc.

    Knob Level Diagram:

                  min_level_angle                              max_level_angle
                         ↓                                             ↓
        ─────────────────●─────────────────────────────────────────────●
        │   Dead Zone    │                Active Range                 │
        ├────────────────┼─────────────────────────────────────────────┤
                         │                                             │
Level -1 (INIT)       Level 0      Level 1    ...    Level (X-2)    Level (X-1)
                         │          │                   │              │
                         ├──────────┼───────────────────┼──────────────┤
        0°               θ_min      θ₁                  θ₂            θ_max
Joint Lower Limit                                                 Joint Upper Limit

    Where:
        - Dead Zone: 0° to min_level_angle (Level -1, mostly for safety or aesthetic purposes)
        - Active Range: min_level_angle to max_level_angle (Levels 0 to X-1)
        - X = total number of discrete levels (num_levels)
        - Level -1: Dead zone (joint angle < min_level_angle)
        - Level 0: First active level at min_level_angle
        - Level X-1: Last active level at max_level_angle
        - target_level = integer in [-1, X-1]
        - min_level_angle = joint angle (degrees) at level 0 (start of active range)
        - max_level_angle = joint angle (degrees) at level X-1 (end of active range)
        - Intermediate levels are linearly interpolated between min and max angles
    """

    def __init__(self, turnable_joint_name: str, min_level_angle: float, max_level_angle: float, num_levels: int, **kwargs):
        """
        Initialize a turnable object.

        Args:
            turnable_joint_name: Name of the revolute joint that can be turned
            min_level_angle: Joint angle (degrees) at level 0 (first active level)
            max_level_angle: Joint angle (degrees) at level (num_levels-1) (last active level)
            num_levels: Total number of discrete active levels (must be >= 1)
            **kwargs: Additional arguments passed to AffordanceBase
        """
        super().__init__(**kwargs)
        self.turnable_joint_name = turnable_joint_name

        assert min_level_angle >= 0.0, "min_level_angle must be non-negative"
        assert min_level_angle < max_level_angle, "min_level_angle must be less than max_level_angle"
        self.min_level_angle = min_level_angle * math.pi / 180.0
        self.max_level_angle = max_level_angle * math.pi / 180.0

        self.num_levels = num_levels
        assert self.num_levels >= 1, "num_levels must be at least 1"

    def get_turning_level(self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
        """Get the current turning level of the object (in range [-1, num_levels-1])."""
        if asset_cfg is None:
            asset_cfg = SceneEntityCfg(self.name)
        asset_cfg = self._add_joint_name_to_scene_entity_cfg(asset_cfg)
        theta = get_unnormalized_joint_position(env, asset_cfg)

        # If in dead zone, return level -1
        in_dead_zone = theta < self.min_level_angle

        # Clamp to active range for level calculation
        theta_clamped = torch.clamp(theta, self.min_level_angle, self.max_level_angle)
        normalized_position = normalize_value(theta_clamped, self.min_level_angle, self.max_level_angle)

        # Rounding ensures it switches levels at the halfway mark between level angles.
        # Level 0 is at min_level_angle, Level (num_levels-1) is at max_level_angle
        level = torch.round(normalized_position * (self.num_levels - 1))

        # Set level to -1 for dead zone positions
        level = torch.where(in_dead_zone, torch.full_like(level, -1.0), level)

        return level

    def turn_to_level(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg | None = None,
        target_level: int = -1,
    ):
        """Set the turning level of the object to the given level (in all the environments).

        Args:
            env: The environment
            env_ids: Environment IDs to apply the change to
            asset_cfg: Scene entity configuration
            target_level: Target level in range [-1, num_levels-1]
                   -1 = dead zone (sets joint to 0°)
                   0 = first active level (sets joint to min_level_angle)
                   num_levels-1 = last active level (sets joint to max_level_angle)
        """
        assert target_level >= -1 and target_level < self.num_levels, f"target_level must be between -1 and {self.num_levels-1}"
        if asset_cfg is None:
            asset_cfg = SceneEntityCfg(self.name)
        asset_cfg = self._add_joint_name_to_scene_entity_cfg(asset_cfg)

        if target_level == -1:
            # Dead zone: set to 0°
            theta = 0.0
        else:
            # Active range: map level [0, num_levels-1] to angle [min_level_angle, max_level_angle]
            step_size = (self.max_level_angle - self.min_level_angle) / (self.num_levels - 1)
            theta = self.min_level_angle + step_size * target_level

        set_unnormalized_joint_position(env, asset_cfg, theta, env_ids)

    def is_at_level(self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg | None = None, target_level: int = -1) -> torch.Tensor:
        """Check if the object is at the given level (in all the environments)."""
        if asset_cfg is None:
            asset_cfg = SceneEntityCfg(self.name)
        asset_cfg = self._add_joint_name_to_scene_entity_cfg(asset_cfg)
        current_level = self.get_turning_level(env, asset_cfg)
        return torch.abs(current_level - target_level) <= 1e-6

    def _add_joint_name_to_scene_entity_cfg(self, asset_cfg: SceneEntityCfg) -> SceneEntityCfg:
        asset_cfg.joint_names = [self.turnable_joint_name]
        return asset_cfg

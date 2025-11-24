# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

from isaaclab_arena.affordances.affordance_base import AffordanceBase
from isaaclab_arena.utils.joint_utils import get_normalized_joint_position, set_normalized_joint_position


class Pressable(AffordanceBase):
    """Interface for pressable objects."""

    def __init__(self, pressable_joint_name: str, pressedness_threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.pressable_joint_name = pressable_joint_name
        self.pressedness_threshold = pressedness_threshold

    def is_pressed(
        self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg | None = None, pressedness_threshold: float | None = None
    ) -> torch.Tensor:
        """Returns a boolean tensor of whether the object is pressed."""
        if asset_cfg is None:
            asset_cfg = SceneEntityCfg(self.name)
        # We allow for overriding the object-level threshold by passing an argument to this
        # function explicitly. Otherwise we use the object-level threshold.
        if pressedness_threshold is None:
            pressedness_threshold = self.pressedness_threshold
        asset_cfg = self._add_joint_name_to_scene_entity_cfg(asset_cfg)
        return get_normalized_joint_position(env, asset_cfg) > pressedness_threshold

    def press(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg | None = None,
        pressed_percentage: float = 1.0,
    ):
        """Press the object (in all the environments)."""
        if asset_cfg is None:
            asset_cfg = SceneEntityCfg(self.name)
        asset_cfg = self._add_joint_name_to_scene_entity_cfg(asset_cfg)
        set_normalized_joint_position(env, asset_cfg, pressed_percentage, env_ids)

    def unpress(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg | None = None,
        unpressed_percentage: float = 1.0,
    ):
        """Unpress the object (in all the environments)."""
        pressed_percentage = 1.0 - unpressed_percentage
        self.press(env, env_ids, asset_cfg, pressed_percentage)

    def _add_joint_name_to_scene_entity_cfg(self, asset_cfg: SceneEntityCfg) -> SceneEntityCfg:
        asset_cfg.joint_names = [self.pressable_joint_name]
        return asset_cfg

# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Affordance for objects with a joint that is pressed, such as buttons."""

import torch

from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

from isaaclab_arena.affordances.affordance_base import AffordanceBase
from isaaclab_arena.utils.joint_utils import get_normalized_joint_position, set_normalized_joint_position


class Pressable(AffordanceBase):
    """Interface for pressable objects."""

    def __init__(self, pressable_joint_name: str, pressedness_threshold: float = 0.5, **kwargs):
        """Initialize the pressable affordance.

        Args:
            pressable_joint_name: Name of the joint that is pressed.
            pressedness_threshold: Pressedness, as a fraction of the joint range, above which the object is pressed.
            **kwargs: Keyword arguments forwarded to the next class in the method resolution order.
        """
        super().__init__(**kwargs)
        self.pressable_joint_name = pressable_joint_name
        self.pressedness_threshold = pressedness_threshold

    def is_pressed(
        self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg | None = None, pressedness_threshold: float | None = None
    ) -> torch.Tensor:
        """Returns a boolean tensor of whether the object is pressed.

        Args:
            env: The environment.
            asset_cfg: Scene entity of the object; the entity named after this object when omitted.
            pressedness_threshold: Pressedness above which the object is pressed; the object's
                `pressedness_threshold` when omitted.
        """
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
        """Press the object (in all the environments).

        Args:
            env: The environment.
            env_ids: Environment indices to apply the change to; all environments when None.
            asset_cfg: Scene entity of the object; the entity named after this object when omitted.
            pressed_percentage: Target joint position as a fraction of the joint range.
        """
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
        """Unpress the object (in all the environments).

        Args:
            env: The environment.
            env_ids: Environment indices to apply the change to; all environments when None.
            asset_cfg: Scene entity of the object; the entity named after this object when omitted.
            unpressed_percentage: Fraction of the joint range the joint is released by, measured from the fully
                pressed position.
        """
        pressed_percentage = 1.0 - unpressed_percentage
        self.press(env, env_ids, asset_cfg, pressed_percentage)

    def _add_joint_name_to_scene_entity_cfg(self, asset_cfg: SceneEntityCfg) -> SceneEntityCfg:
        asset_cfg.joint_names = [self.pressable_joint_name]
        return asset_cfg

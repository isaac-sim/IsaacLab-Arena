# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Affordance for objects with a revolute joint that opens and closes, such as doors."""

import torch

from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

from isaaclab_arena.affordances.affordance_base import AffordanceBase
from isaaclab_arena.utils.joint_utils import get_normalized_joint_position, set_normalized_joint_position


class Openable(AffordanceBase):
    """Interface for openable objects."""

    def __init__(self, openable_joint_name: str, openable_threshold: float = 0.5, **kwargs):
        """Initialize the openable affordance.

        Args:
            openable_joint_name: Name of the revolute joint that opens and closes the object.
            openable_threshold: Openness, as a fraction of the joint range, that separates open from closed.
            **kwargs: Keyword arguments forwarded to the next class in the method resolution order.
        """
        super().__init__(**kwargs)
        # TODO(alexmillane, 2025.08.26): We probably want to be able to define the polarity of the joint.
        self.openable_joint_name = openable_joint_name
        # For a bistate object, we use a single threshold
        # is_open: openness > threshold
        # is_closed: openness <= threshold
        self.openable_threshold = openable_threshold

    def get_openness(self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
        """Returns the percentage open that the object is.

        Args:
                env: The environment.
                asset_cfg: Scene entity of the object; derived from the object's name when omitted.

            Returns:
                Openness as a fraction of the joint range, with shape (num_envs,).
        """
        if asset_cfg is None:
            asset_cfg = SceneEntityCfg(self.name)
        asset_cfg = self._add_joint_name_to_scene_entity_cfg(asset_cfg)
        return get_normalized_joint_position(env, asset_cfg)

    def is_open(
        self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg | None = None, threshold: float | None = None
    ) -> torch.Tensor:
        """Returns a boolean tensor of whether the object is open.

        For a bistate object, this checks if openness > threshold.

        Args:
            env: The environment.
            asset_cfg: Scene entity of the object; derived from the object's name when omitted.
            threshold: Openness above which the object is open; the object's `openable_threshold` when omitted.
        """
        if threshold is not None:
            used_threshold = threshold
        else:
            used_threshold = self.openable_threshold
        openness = self.get_openness(env, asset_cfg)
        return openness > used_threshold

    def is_closed(
        self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg | None = None, threshold: float | None = None
    ) -> torch.Tensor:
        """Returns a boolean tensor of whether the object is closed.

        For a bistate object, this checks if openness <= threshold.
        This is the logical inverse of is_open().

        Args:
            env: The environment.
            asset_cfg: Scene entity of the object; derived from the object's name when omitted.
            threshold: Openness at or below which the object is closed; the object's
                `openable_threshold` when omitted.
        """
        if threshold is not None:
            used_threshold = threshold
        else:
            used_threshold = self.openable_threshold
        openness = self.get_openness(env, asset_cfg)
        return openness <= used_threshold

    def rotate_revolute_joint(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg | None = None,
        percentage: float = 0.0,
    ):
        """Rotate the revolute joint of the object to the given percentage (in all the environments).

        Args:
            env: The environment.
            env_ids: Environment indices to apply the change to; all environments when None.
            asset_cfg: Scene entity of the object; derived from the object's name when omitted.
            percentage: Target joint position as a fraction of the joint range, in [0, 1].
        """
        assert percentage >= 0.0 and percentage <= 1.0, "Percentage must be between 0.0 and 1.0"
        if asset_cfg is None:
            asset_cfg = SceneEntityCfg(self.name)
        asset_cfg = self._add_joint_name_to_scene_entity_cfg(asset_cfg)
        set_normalized_joint_position(env, asset_cfg, percentage, env_ids)

    # keep below for backwards compatibility
    def open(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg | None = None,
        percentage: float = 1.0,
    ):
        """Open the object by rotating its revolute joint to the given percentage.

        Args:
            env: The environment.
            env_ids: Environment indices to apply the change to; all environments when None.
            asset_cfg: Scene entity of the object; derived from the object's name when omitted.
            percentage: Target joint position as a fraction of the joint range, in [0, 1].
        """
        self.rotate_revolute_joint(env, env_ids, asset_cfg, percentage)

    # keep below for backwards compatibility
    def close(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg | None = None,
        percentage: float = 0.0,
    ):
        """Close the object by rotating its revolute joint to the given percentage.

        Args:
            env: The environment.
            env_ids: Environment indices to apply the change to; all environments when None.
            asset_cfg: Scene entity of the object; derived from the object's name when omitted.
            percentage: Target joint position as a fraction of the joint range, in [0, 1].
        """
        self.rotate_revolute_joint(env, env_ids, asset_cfg, percentage)

    def _add_joint_name_to_scene_entity_cfg(self, asset_cfg: SceneEntityCfg) -> SceneEntityCfg:
        asset_cfg.joint_names = [self.openable_joint_name]
        return asset_cfg

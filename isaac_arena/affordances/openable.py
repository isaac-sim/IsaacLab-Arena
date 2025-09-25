# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

from isaac_arena.affordances.affordance_base import AffordanceBase


def normalize_value(value: torch.Tensor, min_value: float, max_value: float):
    return (value - min_value) / (max_value - min_value)


def unnormalize_value(value: float, min_value: float, max_value: float):
    return min_value + (max_value - min_value) * value


def get_normalized_joint_position(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg):
    articulation = env.scene.articulations[asset_cfg.name]
    assert len(asset_cfg.joint_names) == 1, "Only one joint name is supported for now."
    joint_index = articulation.data.joint_names.index(asset_cfg.joint_names[0])
    joint_position = articulation.data.joint_pos[:, joint_index]
    joint_position_limits = articulation.data.joint_pos_limits[0, joint_index, :]
    joint_min, joint_max = joint_position_limits[0], joint_position_limits[1]
    normalized_position = normalize_value(joint_position, joint_min, joint_max)
    if joint_min < 0.0:
        normalized_position = 1 - normalized_position
    return normalized_position


def set_normalized_joint_position(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, target_joint_position: float, env_ids: torch.Tensor | None = None
):
    articulation = env.scene.articulations[asset_cfg.name]
    assert len(asset_cfg.joint_names) == 1, "Only one joint name is supported for now."
    joint_index = articulation.data.joint_names.index(asset_cfg.joint_names[0])
    joint_position_limits = articulation.data.joint_pos_limits[0, joint_index, :]
    joint_min, joint_max = joint_position_limits[0], joint_position_limits[1]
    if joint_min < 0.0:
        target_joint_position = 1 - target_joint_position
    target_joint_position_unnormlized = unnormalize_value(target_joint_position, joint_min, joint_max)
    articulation.write_joint_position_to_sim(
        torch.tensor([[target_joint_position_unnormlized]]).to(env.device),
        torch.tensor([joint_index]).to(env.device),
        env_ids=env_ids.to(env.device) if env_ids is not None else None,
    )


class Openable(AffordanceBase):
    """Interface for openable objects."""

    def __init__(self, openable_joint_name: str, openable_open_threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        # TODO(alexmillane, 2025.08.26): We probably want to be able to define the polarity of the joint.
        self.openable_joint_name = openable_joint_name
        self.openable_open_threshold = openable_open_threshold

    def is_open(
        self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg | None = None, threshold: float | None = None
    ) -> torch.Tensor:
        """Returns a boolean tensor of whether the object is open."""
        if asset_cfg is None:
            asset_cfg = SceneEntityCfg(self.name)
        # We allow for overriding the object-level threshold by passing an argument to this
        # function explicitly. Otherwise we use the object-level threshold.
        if threshold is not None:
            openable_open_threshold = threshold
        else:
            openable_open_threshold = self.openable_open_threshold
        asset_cfg = self._add_joint_name_to_scene_entity_cfg(asset_cfg)
        return get_normalized_joint_position(env, asset_cfg) > openable_open_threshold

    def open(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg | None = None,
        percentage: float = 1.0,
    ):
        """Open the object (in all the environments)."""
        if asset_cfg is None:
            asset_cfg = SceneEntityCfg(self.name)
        asset_cfg = self._add_joint_name_to_scene_entity_cfg(asset_cfg)
        set_normalized_joint_position(env, asset_cfg, percentage, env_ids)

    def close(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg | None = None,
        percentage: float = 0.0,
    ):
        """Close the object (in all the environments)."""
        if asset_cfg is None:
            asset_cfg = SceneEntityCfg(self.name)
        asset_cfg = self._add_joint_name_to_scene_entity_cfg(asset_cfg)
        set_normalized_joint_position(env, asset_cfg, percentage, env_ids)

    def _add_joint_name_to_scene_entity_cfg(self, asset_cfg: SceneEntityCfg) -> SceneEntityCfg:
        asset_cfg.joint_names = [self.openable_joint_name]
        return asset_cfg

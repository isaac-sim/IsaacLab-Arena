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

from typing import Any, Union

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnvCfg
from isaaclab_arena.utils.configclass import make_configclass

AssetCfg = Union[AssetBaseCfg, RigidObjectCfg, ArticulationCfg, ContactSensorCfg]


class Scene:

    def __init__(self, assets: list[Asset] | None = None):
        self.assets: dict[str, Asset] = {}
        # We add these here so a user can override them if they want.
        self.observation_cfg = None
        self.events_cfg = None
        self.termination_cfg = None
        if assets is not None:
            self.add_assets(assets)

    def add_asset(self, asset: Asset):
        assert asset.name is not None, "Asset with the same name already exists"
        self.assets[asset.name] = asset

    def add_assets(self, assets: list[Asset]):
        for asset in assets:
            self.add_asset(asset)

    def get_scene_cfg(self) -> Any:
        """Returns a configclass containing all the scene elements."""
        # Combine the configs into a configclass.
        fields: list[tuple[str, type, AssetCfg]] = []
        for asset in self.assets.values():
            for asset_cfg_name, asset_cfg in asset.get_cfgs().items():
                fields.append((asset_cfg_name, type(asset_cfg), asset_cfg))
        NewConfigClass = make_configclass("SceneCfg", fields)
        new_config_class = NewConfigClass()
        return new_config_class

    def get_observation_cfg(self) -> Any:
        return self.observation_cfg

    def get_events_cfg(self) -> Any:
        return self.events_cfg

    def get_termination_cfg(self) -> Any:
        return self.termination_cfg

    def modify_env_cfg(self, env_cfg: IsaacLabArenaManagerBasedRLEnvCfg) -> IsaacLabArenaManagerBasedRLEnvCfg:
        return env_cfg

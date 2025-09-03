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

from abc import abstractmethod
from typing import Any

from isaaclab.envs import ManagerBasedRLMimicEnv

from isaac_arena.assets.asset import Asset
from isaac_arena.geometry.pose import Pose


class EmbodimentBase(Asset):

    name: str | None = None
    tags: list[str] = ["embodiment"]

    def __init__(self):
        self.scene_config: Any | None = None
        self.camera_config: Any | None = None
        self.action_config: Any | None = None
        self.observation_config: Any | None = None
        self.event_config: Any | None = None
        self.mimic_env: Any | None = None
        self.xr: Any | None = None

    def get_scene_cfg(self) -> Any:
        return self.scene_config

    def get_action_cfg(self) -> Any:
        return self.action_config

    def get_observation_cfg(self) -> Any:
        return self.observation_config

    def get_event_cfg(self) -> Any:
        return self.event_config

    def get_mimic_env(self) -> ManagerBasedRLMimicEnv:
        return self.mimic_env

    def get_xr_cfg(self) -> Any:
        return self.xr

    def get_camera_cfg(self) -> Any:
        return self.camera_config

    @abstractmethod
    def get_retargeters_cfg(self, retargeter_name: str) -> Any:
        raise NotImplementedError("Function not implemented yet.")

    def set_robot_initial_pose(self, pose: Pose):
        if self.scene_config is None or not hasattr(self.scene_config, "robot"):
            raise RuntimeError("scene_config must be populated with a `robot` before calling `set_robot_initial_pose`.")
        self.scene_config.robot.init_state.pos = pose.position_xyz
        self.scene_config.robot.init_state.rot = pose.rotation_wxyz

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
from isaac_arena.utils.cameras import make_camera_observation_cfg
from isaac_arena.utils.configclass import combine_configclass_instances


class EmbodimentBase(Asset):

    name: str | None = None
    tags: list[str] = ["embodiment"]

    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None):
        self.enable_cameras = enable_cameras
        self.initial_pose = initial_pose
        # These should be filled by the subclass
        self.scene_config: Any | None = None
        self.camera_config: Any | None = None
        self.action_config: Any | None = None
        self.observation_config: Any | None = None
        self.event_config: Any | None = None
        self.mimic_env: Any | None = None
        self.xr: Any | None = None

    def set_initial_pose(self, pose: Pose) -> None:
        self.initial_pose = pose

    def get_scene_cfg(self) -> Any:
        if self.enable_cameras:
            if self.camera_config is not None:
                return combine_configclass_instances(
                    "SceneCfg",
                    self.scene_config,
                    self.camera_config,
                )
        if self.initial_pose is not None:
            self.scene_config = self._update_scene_cfg_with_robot_initial_pose(self.scene_config, self.initial_pose)
        return self.scene_config

    def get_action_cfg(self) -> Any:
        return self.action_config

    def get_observation_cfg(self) -> Any:
        if self.enable_cameras:
            if self.camera_config is not None:
                camera_observation_config = make_camera_observation_cfg(self.camera_config)
                return combine_configclass_instances(
                    "ObservationCfg",
                    self.observation_config,
                    camera_observation_config,
                )
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

    def _update_scene_cfg_with_robot_initial_pose(self, scene_config: Any, pose: Pose) -> Any:
        if scene_config is None or not hasattr(scene_config, "robot"):
            raise RuntimeError("scene_config must be populated with a `robot` before calling `set_robot_initial_pose`.")
        scene_config.robot.init_state.pos = pose.position_xyz
        scene_config.robot.init_state.rot = pose.rotation_wxyz
        return scene_config

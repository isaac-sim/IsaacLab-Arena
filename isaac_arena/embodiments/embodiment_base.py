# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

from abc import ABC
from typing import Any

from isaac_arena.geometry.pose import Pose


class EmbodimentBase(ABC):

    def __init__(self):
        self.scene_config: Any | None = None
        self.action_config: Any | None = None
        self.observation_config: Any | None = None
        self.event_config: Any | None = None

    def get_scene_cfg(self) -> Any:
        return self.scene_config

    def get_action_cfg(self) -> Any:
        return self.action_config

    def get_observation_cfg(self) -> Any:
        return self.observation_config

    def get_event_cfg(self) -> Any:
        return self.event_config

    def set_robot_initial_pose(self, pose: Pose):
        if self.scene_config is None or not hasattr(self.scene_config, "robot"):
            raise RuntimeError("scene_config must be populated with a `robot` before calling `set_robot_initial_pose`.")
        self.scene_config.robot.init_state.pos = pose.position_xyz
        self.scene_config.robot.init_state.rot = pose.rotation_wxyz

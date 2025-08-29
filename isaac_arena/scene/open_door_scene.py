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

from dataclasses import MISSING
from typing import Any

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.utils import configclass

from isaac_arena.assets.asset import Asset
from isaac_arena.geometry.pose import Pose
from isaac_arena.scene.scene import SceneBase


@configclass
class OpenDoorSceneCfg:

    # The scene of the environment where the task is performed
    background_scene: AssetBaseCfg = MISSING

    # The object to pick up
    interactable_object: ArticulationCfg = MISSING


class OpenDoorScene(SceneBase):

    def __init__(self, background_scene: Asset, interactable_object: Asset):
        super().__init__()
        # Save the background and the pick up object
        self.background_scene = background_scene
        self.interactable_object = interactable_object
        # Set the pose of the pick up object and the robot
        # self.interactable_object.set_initial_pose(self.background_scene.object_pose)
        self.robot_initial_pose = background_scene.get_robot_initial_pose()

    def get_scene_cfg(self) -> OpenDoorSceneCfg:
        # Set the initial pose of the pick up object
        # NOTE(alexmillane, 2025-08-27): We do this here, at the last moment as the env is built,
        # to give the user the maximum time to be able to set the initial pose of the pick up object
        # directly from outside.
        if not self.interactable_object.is_initial_pose_set():
            self.interactable_object.set_initial_pose(self.background_scene.object_pose)
        else:
            print("Pickup object initial pose set explicitly, not using background scene object pose")
        return OpenDoorSceneCfg(
            background_scene=self.background_scene.get_background_cfg(),
            interactable_object=self.interactable_object.get_object_cfg(),
        )

    def get_robot_initial_pose(self) -> Pose:
        return self.robot_initial_pose

    def get_observation_cfg(self) -> Any:
        pass

    def get_events_cfg(self) -> Any:
        pass

    def get_termination_cfg(self) -> Any:
        pass


# TODO(alexmillane, 2025.08.28): Implement reset event.
# @configclass
# class OpenDoorEventCfg:
#     """Configuration for Pick and Place."""

#     reset_pick_up_object_pose: EventTerm = MISSING

#     def __init__(self, object_pose: Pose):
#         self.reset_pick_up_object_pose = EventTerm(
#             func=franka_stack_events.randomize_object_pose,
#             mode="reset",
#             # NOTE: We use a randomize term but set the pose range to the same value to achieve constant pose for now.
#             params={
#                 "pose_range": {
#                     "x": (object_pose.position_xyz[0], object_pose.position_xyz[0]),
#                     "y": (object_pose.position_xyz[1], object_pose.position_xyz[1]),
#                     "z": (object_pose.position_xyz[2], object_pose.position_xyz[2]),
#                 },
#                 "asset_cfgs": [SceneEntityCfg("interactable_object")],
#             },
#         )

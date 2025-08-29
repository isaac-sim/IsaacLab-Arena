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

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

from isaac_arena.assets.asset import Asset
from isaac_arena.geometry.pose import Pose
from isaac_arena.scene.scene import SceneBase


@configclass
class PickAndPlaceSceneCfg:

    # The scene of the environment where the task is performed
    background_scene: AssetBaseCfg = MISSING

    # The object to pick up
    pick_up_object: RigidObjectCfg = MISSING

    # The contact sensor on the pick up object
    pick_up_object_contact_sensor: ContactSensorCfg = MISSING

    # The object to place the object on/into
    destination_object: RigidObjectCfg = MISSING


class PickAndPlaceScene(SceneBase):

    def __init__(self, background_scene: Asset, pick_up_object: Asset):
        super().__init__()
        # Save the background and the pick up object
        self.background_scene = background_scene
        self.pick_up_object = pick_up_object
        # Set the pose of the pick up object and the robot
        self.robot_initial_pose = background_scene.get_robot_initial_pose()

    def get_scene_cfg(self) -> PickAndPlaceSceneCfg:
        # Set the initial pose of the pick up object
        # NOTE(alexmillane, 2025-08-27): We do this here, at the last moment as the env is built,
        # to give the user the maximum time to be able to set the initial pose of the pick up object
        # directly from outside.
        if not self.pick_up_object.is_initial_pose_set():
            self.pick_up_object.set_initial_pose(self.background_scene.object_pose)
        else:
            print("Pickup object initial pose set explicitly, not using background scene object pose")
        return PickAndPlaceSceneCfg(
            background_scene=self.background_scene.get_background_cfg(),
            pick_up_object=self.pick_up_object.get_object_cfg(),
            pick_up_object_contact_sensor=self.pick_up_object.get_contact_sensor_cfg(
                contact_against_prim_paths=[self.background_scene.get_destination_cfg().prim_path],
            ),
            destination_object=self.background_scene.get_destination_cfg(),
        )

    def get_robot_initial_pose(self) -> Pose:
        return self.robot_initial_pose

    def get_observation_cfg(self) -> Any:
        pass

    def get_events_cfg(self) -> Any:
        return PickAndPlaceEventCfg(object_pose=self.background_scene.object_pose)

    def get_termination_cfg(self) -> Any:
        return PickAndPlaceTerminationCfg(minimum_height=self.background_scene.object_min_z)


@configclass
class PickAndPlaceEventCfg:
    """Configuration for Pick and Place."""

    reset_pick_up_object_pose: EventTerm = MISSING

    def __init__(self, object_pose: Pose):
        self.reset_pick_up_object_pose = EventTerm(
            func=franka_stack_events.randomize_object_pose,
            mode="reset",
            # NOTE: We use a randomize term but set the pose range to the same value to achieve constant pose for now.
            params={
                "pose_range": {
                    "x": (object_pose.position_xyz[0], object_pose.position_xyz[0]),
                    "y": (object_pose.position_xyz[1], object_pose.position_xyz[1]),
                    "z": (object_pose.position_xyz[2], object_pose.position_xyz[2]),
                },
                "asset_cfgs": [SceneEntityCfg("pick_up_object")],
            },
        )


@configclass
class PickAndPlaceTerminationCfg:
    """Configuration for Pick and Place."""

    object_dropped: TerminationTermCfg = MISSING

    def __init__(self, minimum_height: float = 0.5):
        self.object_dropped = TerminationTermCfg(
            func=mdp_isaac_lab.root_height_below_minimum,
            params={"minimum_height": minimum_height, "asset_cfg": SceneEntityCfg("pick_up_object")},
        )

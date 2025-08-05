# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

from dataclasses import MISSING
from typing import Any

from isaac_arena.assets.asset import Asset
from isaac_arena.geometry.pose import Pose
from isaac_arena.scene.scene import SceneBase
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events


@configclass
class PickAndPlaceSceneCfg:

    # The scene of the environment where the task is performed
    background_scene: AssetBaseCfg = MISSING

    # The object to pick up
    pick_up_object: RigidObjectCfg = MISSING

    # The object to place the object on/into
    destination_object: RigidObjectCfg = MISSING


class PickAndPlaceScene(SceneBase):

    def __init__(self, background_scene: Asset, pick_up_object: Asset):
        super().__init__()
        # Save the background and the pick up object
        self.background_scene = background_scene
        self.pick_up_object = pick_up_object
        # Set the pose of the pick up object and the robot
        self.pick_up_object.set_initial_pose(self.background_scene.object_pose)
        self.robot_initial_pose = background_scene.get_robot_initial_pose()

    def get_scene_cfg(self) -> PickAndPlaceSceneCfg:
        return PickAndPlaceSceneCfg(
            background_scene=self.background_scene.get_background_cfg(),
            pick_up_object=self.pick_up_object.get_object_cfg(),
            destination_object=self.background_scene.get_destination_cfg(),
        )

    def get_robot_initial_pose(self) -> Pose:
        return self.robot_initial_pose

    def get_observation_cfg(self) -> Any:
        pass

    def get_events_cfg(self) -> Any:
        return PickAndPlaceEventCfg(object_pose=self.background_scene.object_pose)


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

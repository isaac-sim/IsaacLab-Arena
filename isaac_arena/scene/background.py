# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

import isaaclab.sim as sim_utils
from isaac_arena.geometry.pose import Pose
from isaac_arena.scene.asset import Asset
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg


class Background(Asset):
    """
    Encapsulates the background scene config for a environment.
    """

    def __init__(self, background_scene_cfg: AssetBaseCfg, name: str, tags: list[str], robot_initial_pose: Pose):
        super().__init__(name, tags)
        self.background_scene_cfg = background_scene_cfg
        self.robot_initial_pose = robot_initial_pose

    def get_background_cfg(self) -> AssetBaseCfg:
        """Return the configured background scene asset."""
        return self.background_scene_cfg

    def get_robot_initial_pose(self) -> Pose:
        """Return the configured robot initial pose."""
        return self.robot_initial_pose


class PickAndPlaceBackground(Background):
    """
    Encapsulates the background scene config for a environment.
    """

    def __init__(
        self,
        background_scene_cfg: AssetBaseCfg,
        destination_object_cfg: RigidObjectCfg,
        object_location_cfg: RigidObjectCfg.InitialStateCfg,
        robot_initial_pose: Pose,
        name: str,
    ):
        super().__init__(background_scene_cfg, name, ["background", "pick_and_place"], robot_initial_pose)
        self.destination_object_cfg = destination_object_cfg
        self.object_location_cfg = object_location_cfg

    def get_destination_cfg(self) -> RigidObjectCfg:
        """Return the configured destination-object asset."""
        return self.destination_object_cfg

    def get_object_location_cfg(self) -> RigidObjectCfg.InitialStateCfg:
        """Return the configured pick-up object location."""
        return self.object_location_cfg


class KitchenPickAndPlaceBackground(PickAndPlaceBackground):
    """
    Encapsulates the background scene and destination-object config for a kitchen pick-and-place environment.
    """

    def __init__(self):
        # Background scene (static kitchen environment)
        super().__init__(
            background_scene_cfg=AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Kitchen",
                init_state=AssetBaseCfg.InitialStateCfg(pos=[0.772, 3.39, -0.895], rot=[0.70711, 0, 0, -0.70711]),
                spawn=UsdFileCfg(
                    usd_path=(
                        "omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/isaac_arena/kitchen_scene_teleop_v3.usd"
                    )
                ),
            ),
            # NOTE(alexmillane, 2025.07.28): We used to use the bottom of drawer with mugs as the destination object.
            # I have just changed it to the cabinet. But this is not tested as working yet.
            destination_object_cfg=RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Kitchen/Cabinet_B_02",
            ),
            object_location_cfg=RigidObjectCfg.InitialStateCfg(pos=[0.35, 0.0, 0.094], rot=[0.0, 0.0, 0.0, 1.0]),
            robot_initial_pose=Pose.identity(),
            name="kitchen_pick_and_place",
        )


class PackingTablePickAndPlaceBackground(PickAndPlaceBackground):
    """
    Encapsulates the background scene and destination-object config for a packing table pick-and-place environment.
    """

    def __init__(self):
        super().__init__(
            background_scene_cfg=AssetBaseCfg(
                prim_path="/World/envs/env_.*/PackingTable",
                init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
                spawn=UsdFileCfg(
                    usd_path="omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/mindmap/packing_table_arena.usd",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                ),
            ),
            destination_object_cfg=RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/PackingTable/container_h20",
            ),
            object_location_cfg=RigidObjectCfg.InitialStateCfg(pos=[-0.35, 0.40, 1.0413], rot=[1.0, 0.0, 0.0, 0.0]),
            robot_initial_pose=Pose(
                position_xyz=(0.0, 0.0, 1.0),
                rotation_wxyz=(0.7071068, 0, 0, 0.7071068),
            ),
            name="packing_table_pick_and_place",
        )

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

from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

from isaac_arena.assets.asset import Asset
from isaac_arena.assets.register import register_asset
from isaac_arena.geometry.pose import Pose


class Background(Asset):
    """
    Encapsulates the background scene config for a environment.
    """

    # Defined in Asset, restated here for clariry
    # name: str | None = None
    # tags: list[str] | None = None
    background_scene_cfg: AssetBaseCfg | None = None

    def __init__(self, robot_initial_pose: Pose):
        super().__init__()
        self.robot_initial_pose = robot_initial_pose

    def get_robot_initial_pose(self) -> Pose:
        """Return the configured robot initial pose."""
        return self.robot_initial_pose

    def get_cfgs(self) -> dict[str, Any]:
        assert self.name is not None, "Background name is not set"
        return {
            self.name: self.background_scene_cfg,
        }


class PickAndPlaceBackground(Background):
    """
    Encapsulates the background scene config for a environment.
    """

    destination_object_cfg: RigidObjectCfg | None = None
    object_pose: Pose | None = None

    def __init__(self, robot_initial_pose: Pose):
        super().__init__(robot_initial_pose)

    def get_destination_cfg(self) -> RigidObjectCfg:
        """Return the configured destination-object asset."""
        return self.destination_object_cfg


@register_asset
class KitchenPickAndPlaceBackground(PickAndPlaceBackground):
    """
    Encapsulates the background scene and destination-object config for a kitchen pick-and-place environment.
    """

    name = "kitchen_pick_and_place"
    tags = ["background", "pick_and_place"]
    default_robot_initial_pose = Pose.identity()
    background_scene_cfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Kitchen",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.772, 3.39, -0.895], rot=[0.70711, 0, 0, -0.70711]),
        spawn=UsdFileCfg(
            usd_path="omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/isaac_arena/kitchen_scene_teleop_v3.usd"
        ),
    )
    # NOTE(alexmillane, 2025.07.31): Eventually we'll likely want to make these dynamic
    # such that a single background can be used for multiple tasks.
    destination_object_cfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Kitchen/Cabinet_B_02",
    )
    object_pose = Pose(
        position_xyz=(0.45, -0.05, 0.094),
        rotation_wxyz=(0.0, 0.0, 0.0, 1.0),
    )
    object_min_z = -0.2

    def __init__(self, robot_initial_pose: Pose = default_robot_initial_pose):
        super().__init__(robot_initial_pose)


@register_asset
class PackingTablePickAndPlaceBackground(PickAndPlaceBackground):
    """
    Encapsulates the background scene and destination-object config for a packing table pick-and-place environment.
    """

    name = "packing_table_pick_and_place"
    tags = ["background", "pick_and_place"]
    default_robot_initial_pose = Pose.identity()
    background_scene_cfg = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.72193, -0.04727, -0.92512], rot=[0.70711, 0.0, 0.0, -0.70711]),
        spawn=UsdFileCfg(
            usd_path="omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/mindmap/packing_table_arena.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )
    # NOTE(alexmillane, 2025.07.31): Eventually we'll likely want to make these dynamic
    # such that a single background can be used for multiple tasks.
    destination_object_cfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PackingTable/container_h20",
    )
    object_pose = Pose(
        position_xyz=(0.32623, -0.00586, 0.08186),
        rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    object_min_z = -0.2

    def __init__(self, robot_initial_pose: Pose = default_robot_initial_pose):
        super().__init__(robot_initial_pose)


@register_asset
class GalileoPickAndPlaceBackground(PickAndPlaceBackground):
    """
    Encapsulates the background scene and destination-object config for a galileo pick-and-place environment.
    """

    name = "galileo_pick_and_place"
    tags = ["background", "pick_and_place"]
    default_robot_initial_pose = Pose.identity()
    background_scene_cfg = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Galileo",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[4.420, 1.408, -0.795], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path="omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/isaac_arena/galileo_simplified.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )
    # NOTE(alexmillane, 2025.07.31): Eventually we'll likely want to make these dynamic
    # such that a single background can be used for multiple tasks.
    destination_object_cfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Galileo/container_h20",
    )
    object_pose = Pose(
        position_xyz=(0.0, 0.0, 0.0),
        rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    object_min_z = -0.2

    def __init__(self, robot_initial_pose: Pose = default_robot_initial_pose):
        super().__init__(robot_initial_pose)

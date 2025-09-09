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

from isaac_arena.assets.background import Background
from isaac_arena.assets.register import register_asset
from isaac_arena.geometry.pose import Pose


@register_asset
class KitchenBackground(Background):
    """
    Encapsulates the background scene and destination-object config for a kitchen pick-and-place environment.
    """

    name = "kitchen"
    tags = ["background"]
    default_robot_initial_pose = Pose.identity()
    usd_path = "omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/isaac_arena/kitchen_scene_teleop_v3.usd"
    initial_pose = Pose(position_xyz=(0.772, 3.39, -0.895), rotation_wxyz=(0.70711, 0, 0, -0.70711))
    object_min_z = -0.2

    def __init__(self, robot_initial_pose: Pose = default_robot_initial_pose):
        super().__init__(robot_initial_pose)


@register_asset
class PackingTableBackground(Background):
    """
    Encapsulates the background scene and destination-object config for a packing table pick-and-place environment.
    """

    name = "packing_table"
    tags = ["background"]
    default_robot_initial_pose = Pose.identity()
    usd_path = "omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/mindmap/packing_table_arena.usd"
    initial_pose = Pose(position_xyz=(0.72193, -0.04727, -0.92512), rotation_wxyz=(0.70711, 0.0, 0.0, -0.70711))
    object_min_z = -0.2

    def __init__(self, robot_initial_pose: Pose = default_robot_initial_pose):
        super().__init__(robot_initial_pose)


@register_asset
class GalileoBackground(Background):
    """
    Encapsulates the background scene and destination-object config for a galileo pick-and-place environment.
    """

    name = "galileo"
    tags = ["background"]
    default_robot_initial_pose = Pose.identity()
    usd_path = "omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/isaac_arena/galileo_simplified.usd"
    initial_pose = Pose(position_xyz=(4.420, 1.408, -0.795), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
    object_min_z = -0.2

    def __init__(self, robot_initial_pose: Pose = default_robot_initial_pose):
        super().__init__(robot_initial_pose)

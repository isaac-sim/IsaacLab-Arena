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

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab_arena.assets.background import Background
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose


class LibraryBackground(Background):
    """
    Base class for objects in the library which are defined in this file.
    These objects have class attributes (rather than instance attributes).
    """

    name: str
    tags: list[str]
    usd_path: str
    initial_pose: Pose
    object_min_z: float

    def __init__(self, **kwargs):
        super().__init__(
            name=self.name,
            tags=self.tags,
            usd_path=self.usd_path,
            initial_pose=self.initial_pose,
            object_min_z=self.object_min_z,
            **kwargs,
        )


# TODO(peterd, 2025.11.05): Update all OV drive paths to use {ISAACLAB_NUCLEUS_DIR}
# alias prior to public release once assets are synced to S3
@register_asset
class KitchenBackground(LibraryBackground):
    """
    Encapsulates the background scene for the kitchen.
    """

    name = "kitchen"
    tags = ["background"]
    usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Arena/assets/background_library/kitchen_background/kitchen_background.usd"
    initial_pose = Pose(position_xyz=(0.772, 3.39, -0.895), rotation_wxyz=(0.70711, 0, 0, -0.70711))
    object_min_z = -0.2

    def __init__(self):
        super().__init__()


@register_asset
class KitchenWithOpenDrawerBackground(LibraryBackground):
    """
    Encapsulates the background scene for the kitchen with an open drawer.
    """

    name = "kitchen_with_open_drawer"
    tags = ["background"]
    usd_path = (
        f"{ISAACLAB_NUCLEUS_DIR}/Arena/assets/background_library/kitchen_scene_teleop_v3/kitchen_scene_teleop_v3.usd"
    )
    initial_pose = Pose(position_xyz=(0.772, 3.39, -0.895), rotation_wxyz=(0.70711, 0, 0, -0.70711))
    object_min_z = -0.2

    def __init__(self):
        super().__init__()


@register_asset
class PackingTableBackground(LibraryBackground):
    """
    Encapsulates the background scene for the packing table.
    """

    name = "packing_table"
    tags = ["background"]
    usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Arena/assets/background_library/packing_table/packing_table.usd"
    initial_pose = Pose(position_xyz=(0.72193, -0.04727, -0.92512), rotation_wxyz=(0.70711, 0.0, 0.0, -0.70711))
    object_min_z = -0.2

    def __init__(self):
        super().__init__()


@register_asset
class GalileoBackground(LibraryBackground):
    """
    Encapsulates the background scene for the galileo room.
    """

    name = "galileo"
    tags = ["background"]
    usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Arena/assets/background_library/galileo_simplified/galileo_simplified.usd"
    initial_pose = Pose(position_xyz=(4.420, 1.408, -0.795), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
    object_min_z = -0.2

    def __init__(self):
        super().__init__()


@register_asset
class GalileoLocomanipBackground(LibraryBackground):
    """
    Encapsulates the background scene for the galileo room for locomanip.
    """

    name = "galileo_locomanip"
    tags = ["background"]
    default_robot_initial_pose = Pose.identity()
    usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Arena/assets/background_library/galileo_locomanip/galileo_locomanip.usd"
    initial_pose = Pose(position_xyz=(4.420, 1.408, -0.795), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
    object_min_z = -0.2

    def __init__(self):
        super().__init__()

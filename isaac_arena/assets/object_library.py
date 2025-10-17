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

from isaac_arena.affordances.openable import Openable
from isaac_arena.affordances.pressable import Pressable
from isaac_arena.assets.object import Object
from isaac_arena.assets.object_base import ObjectType
from isaac_arena.assets.register import register_asset
from isaac_arena.utils.pose import Pose


class LibraryObject(Object):
    """
    Base class for objects in the library which are defined in this file.
    These objects have class attributes (rather than instance attributes).
    """

    name: str
    tags: list[str]
    usd_path: str
    object_type: ObjectType = ObjectType.RIGID
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None, **kwargs):
        super().__init__(
            name=self.name,
            prim_path=prim_path,
            tags=self.tags,
            usd_path=self.usd_path,
            object_type=self.object_type,
            scale=self.scale,
            initial_pose=initial_pose,
            **kwargs,
        )


@register_asset
class CrackerBox(LibraryObject):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    name = "cracker_box"
    tags = ["object"]
    usd_path = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd"

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class MustardBottle(LibraryObject):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    name = "mustard_bottle"
    tags = ["object"]
    usd_path = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd"

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class SugarBox(LibraryObject):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    name = "sugar_box"
    tags = ["object"]
    usd_path = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd"

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class TomatoSoupCan(LibraryObject):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    name = "tomato_soup_can"
    tags = ["object"]
    usd_path = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd"

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class LightWheelKettle21(LibraryObject):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    name = "lightwheel_kettle_21"
    tags = ["object"]
    usd_path = "omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/isaac_arena/g1_locomanip_assets/asset_release/object_library/Kettle021/Kettle021.usd"

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class LightWheelPot51(LibraryObject):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    name = "lightwheel_pot_51"
    tags = ["object"]
    usd_path = "omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/isaac_arena/g1_locomanip_assets/asset_release/object_library/Pot051/Pot051.usd"

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class SketchFabSprayCan3(LibraryObject):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    name = "sketchfab_spray_can_3"
    tags = ["object"]
    prim_path = ("{ENV_REGEX_NS}/target_sketchfab_spray_can_3",)
    usd_path = "omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/isaac_arena/g1_locomanip_assets/asset_release/object_library/spray_bottle_3/spray_bottle_3.usd"

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class PowerDrill(LibraryObject):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    name = "power_drill"
    tags = ["object"]
    usd_path = "omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/isaac_arena/g1_locomanip_assets/asset_release/object_library/power_drill_physics/power_drill_physics.usd"

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class Microwave(LibraryObject, Openable):
    """A microwave oven."""

    # Only required when using Lightwheel SDK
    from lightwheel_sdk.loader import object_loader

    name = "microwave"
    tags = ["object", "openable"]
    file_path, object_name, metadata = object_loader.acquire_by_registry(
        registry_type="fixtures", registry_name=["microwave"], file_type="USD"
    )
    usd_path = file_path
    object_type = ObjectType.ARTICULATION

    # Openable affordance parameters
    openable_joint_name = "microjoint"
    openable_open_threshold = 0.5

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(
            prim_path=prim_path,
            initial_pose=initial_pose,
            openable_joint_name=self.openable_joint_name,
            openable_open_threshold=self.openable_open_threshold,
        )


@register_asset
class CoffeeMachine(LibraryObject, Pressable):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    # Only required when using Lightwheel SDK
    from lightwheel_sdk.loader import object_loader

    name = "coffee_machine"
    tags = ["object", "pressable"]
    file_path, object_name, metadata = object_loader.acquire_by_registry(
        registry_type="fixtures", registry_name=["coffee_machine"], file_type="USD"
    )
    usd_path = file_path
    object_type = ObjectType.ARTICULATION

    # Openable affordance parameters
    pressable_joint_name = "CoffeeMachine108_Button002_joint"
    pressedness_threshold = 0.5

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(
            prim_path=prim_path,
            initial_pose=initial_pose,
            pressable_joint_name=self.pressable_joint_name,
            pressedness_threshold=self.pressedness_threshold,
        )


@register_asset
class OfficeTable(LibraryObject):
    """
    A basic office table.
    """

    name = "office_table"
    tags = ["object"]
    usd_path = "omniverse://isaac-dev.ov.nvidia.com/Isaac/IsaacLab/Mimic/nut_pour_task/nut_pour_assets/table.usd"
    default_prim_path = "{ENV_REGEX_NS}/office_table"
    scale = (1.0, 1.0, 0.7)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class BlueSortingBin(LibraryObject):
    """
    A blue plastic sorting bin.
    """

    name = "blue_sorting_bin"
    tags = ["object"]
    usd_path = "omniverse://isaac-dev.ov.nvidia.com/Isaac/IsaacLab/Mimic/exhaust_pipe_task/exhaust_pipe_assets/blue_sorting_bin.usd"
    default_prim_path = "{ENV_REGEX_NS}/blue_sorting_bin"
    scale = (4.0, 2.0, 1.0)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class BlueExhaustPipe(LibraryObject):
    """
    A blue exhaust pipe.
    """

    name = "blue_exhaust_pipe"
    tags = ["object"]
    usd_path = "omniverse://isaac-dev.ov.nvidia.com/Isaac/IsaacLab/Mimic/exhaust_pipe_task/exhaust_pipe_assets/blue_exhaust_pipe.usd"
    default_prim_path = "{ENV_REGEX_NS}/blue_exhaust_pipe"
    scale = (0.55, 0.55, 1.4)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class BrownBox(LibraryObject):
    """
    A brown box.
    """

    name = "brown_box"
    tags = ["object"]
    usd_path = "omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/isaac_arena/g1_locomanip_assets/asset_release/object_library/brown_box/brown_box.usd"
    default_prim_path = "{ENV_REGEX_NS}/brown_box"
    scale = (1.0, 1.0, 1.0)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)

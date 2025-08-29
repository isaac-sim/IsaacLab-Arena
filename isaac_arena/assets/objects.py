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
from isaaclab.assets import RigidObjectCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

from isaac_arena.assets.affordances import Openable
from isaac_arena.assets.asset import Asset
from isaac_arena.assets.register_asset import registerasset
from isaac_arena.geometry.pose import Pose


class Object(Asset):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    # Defined in Asset, restated here for clariry
    # name: str | None = None
    # tags: list[str] | None = None
    usd_path: str | None = None
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __init__(
        self,
        prim_path: str,
        initial_pose: Pose | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.prim_path = prim_path
        self.initial_pose = initial_pose

    def set_prim_path(self, prim_path: str) -> None:
        self.prim_path = prim_path

    def set_initial_pose(self, pose: Pose) -> None:
        self.initial_pose = pose

    def is_initial_pose_set(self) -> bool:
        return self.initial_pose is not None

    def get_object_cfg(self) -> RigidObjectCfg:
        """Return the configured pick-up object asset."""
        return self._generate_cfg()

    def get_contact_sensor_cfg(self, contact_against_prim_paths: list[str] | None = None) -> ContactSensorCfg:
        if contact_against_prim_paths is None:
            contact_against_prim_paths = []
        return ContactSensorCfg(
            prim_path=self.prim_path,
            filter_prim_paths_expr=contact_against_prim_paths,
        )

    def _generate_cfg(self) -> RigidObjectCfg:
        object_cfg = RigidObjectCfg(
            prim_path=self.prim_path,
            spawn=UsdFileCfg(
                usd_path=self.usd_path,
                scale=self.scale,
                activate_contact_sensors=True,
            ),
        )
        # Optionally specify initial pose
        if self.initial_pose is not None:
            object_cfg.init_state.pos = self.initial_pose.position_xyz
            object_cfg.init_state.rot = self.initial_pose.rotation_wxyz
        return object_cfg


@registerasset
class CrackerBox(Object):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    name = "cracker_box"
    tags = ["object"]
    usd_path = "omniverse://isaac-dev.ov.nvidia.com/NVIDIA/Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd"
    default_prim_path = "{ENV_REGEX_NS}/target_cracker_box"

    def __init__(self, prim_path: str = default_prim_path, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@registerasset
class MustardBottle(Object):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    name = "mustard_bottle"
    tags = ["object"]
    usd_path = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd"
    default_prim_path = "{ENV_REGEX_NS}/target_mustard_bottle"

    def __init__(self, prim_path: str = default_prim_path, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@registerasset
class SugarBox(Object):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    name = "sugar_box"
    tags = ["object"]
    prim_path = ("{ENV_REGEX_NS}/target_sugar_box",)
    usd_path = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd"
    default_prim_path = "{ENV_REGEX_NS}/target_sugar_box"

    def __init__(self, prim_path: str = default_prim_path, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@registerasset
class TomatoSoupCan(Object):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    name = "tomato_soup_can"
    tags = ["object"]
    prim_path = ("{ENV_REGEX_NS}/target_tomato_soup_can",)
    usd_path = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd"
    default_prim_path = "{ENV_REGEX_NS}/target_tomato_soup_can"

    def __init__(self, prim_path: str = default_prim_path, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@registerasset
class LightWheelKettle21(Object):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    name = "lightwheel_kettle_21"
    tags = ["object"]
    prim_path = ("{ENV_REGEX_NS}/target_lightwheel_kettle_21",)
    usd_path = "omniverse://isaac-dev.ov.nvidia.com/Projects/isaac_arena/sample_assets/lightwheel/kettle/Kettle021/Kettle021.usd"
    default_prim_path = "{ENV_REGEX_NS}/target_lightwheel_kettle_21"

    def __init__(self, prim_path: str = default_prim_path, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@registerasset
class SketchFabSprayCan3(Object):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    name = "sketchfab_spray_can_3"
    tags = ["object"]
    prim_path = ("{ENV_REGEX_NS}/target_sketchfab_spray_can_3",)
    usd_path = "omniverse://isaac-dev.ov.nvidia.com/Projects/isaac_arena/sample_assets/sketchfab/spray_bottle/spray_bottle_3/spray_bottle_3.usd"
    default_prim_path = "{ENV_REGEX_NS}/target_sketchfab_spray_can_3"

    def __init__(self, prim_path: str = default_prim_path, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


# TODO(alexmillane, 2025.08.28): Cleanup. Push this override here into the object base class.
@registerasset
class Microwave(Object, Openable):
    """A microwave oven."""

    name = "microwave"
    tags = ["object", "openable"]
    usd_path = "omniverse://isaac-dev.ov.nvidia.com/Projects/isaac_arena/interactable_objects/microwave.usd"
    default_prim_path = "{ENV_REGEX_NS}/target_microwave"

    # Openable affordance parameters
    openable_joint_name = "microjoint"
    openable_open_threshold = 0.5

    def __init__(self, prim_path: str = default_prim_path, initial_pose: Pose | None = None):
        super().__init__(
            prim_path=prim_path,
            initial_pose=initial_pose,
            openable_joint_name=self.openable_joint_name,
            openable_open_threshold=self.openable_open_threshold,
        )

    def get_object_cfg(self) -> ArticulationCfg:
        # TODO(alexmillane, 2025.08.28): This is a hack. Fix.
        # We're overriding the get_object_cfg method in the object base class.
        # We need to move this to the object base class, and detect the correct type of
        # cfg to return.
        # The problem is that all the other objects return a RigidObjectCfg,
        # but this one returns an ArticulationCfg. So we're abusing things here.
        object_cfg = ArticulationCfg(
            prim_path=self.prim_path,
            spawn=UsdFileCfg(
                usd_path=self.usd_path,
                scale=self.scale,
                activate_contact_sensors=True,
            ),
            actuators={},
        )
        # Optionally specify initial pose
        if self.initial_pose is not None:
            object_cfg.init_state.pos = self.initial_pose.position_xyz
            object_cfg.init_state.rot = self.initial_pose.rotation_wxyz
        return object_cfg

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

from isaac_arena.assets.asset import Asset
from isaac_arena.assets.register_asset import registerasset
from isaac_arena.geometry.pose import Pose
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg


class Object(Asset):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    usd_path: str | None = None
    scale: tuple[float, float, float] | None = None

    def __init__(
        self,
        prim_path: str,
        initial_pose: Pose | None = None,
    ):
        super().__init__()
        self.prim_path = prim_path
        self.initial_pose = initial_pose

    def set_prim_path(self, prim_path: str) -> None:
        self.prim_path = prim_path

    def set_initial_pose(self, pose: Pose) -> None:
        self.initial_pose = pose

    def get_object_cfg(self) -> RigidObjectCfg:
        """Return the configured pick-up object asset."""
        return self._generate_cfg()

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
    scale = (1.0, 1.0, 1.0)
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
    scale = (1.0, 1.0, 1.0)
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
    scale = (1.0, 1.0, 1.0)
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
    scale = (1.0, 1.0, 1.0)
    default_prim_path = "{ENV_REGEX_NS}/target_tomato_soup_can"

    def __init__(self, prim_path: str = default_prim_path, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)

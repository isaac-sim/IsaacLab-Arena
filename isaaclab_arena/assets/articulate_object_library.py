# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


import os
from typing import TYPE_CHECKING, Any

import isaaclab.sim as sim_utils

if TYPE_CHECKING:
    from isaaclab_arena.assets.hdr_image import HDRImage
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from isaaclab_arena.affordances.openable import Openable
from isaaclab_arena.affordances.placeable import Placeable
from isaaclab_arena.affordances.pressable import Pressable
from isaaclab_arena.affordances.turnable import Turnable
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.assets.object_utils import (
    EMPTY_ARTICULATION_INIT_STATE_CFG,
    RIGID_BODY_PROPS_HIGH_PRECISION,
    RIGID_BODY_PROPS_MEDIUM_PRECISION,
)
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose

# TODO(xinjieyao, 2026.01.07): Remove staging bucket and use production bucket for release.
ISAACLAB_STAGING_NUCLEUS_DIR = (
    "https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab"
)


class LibraryObject(Object):
    """
    Base class for objects in the library which are defined in this file.
    These objects have class attributes (rather than instance attributes).
    """

    name: str
    tags: list[str]
    usd_path: str | None = None
    object_type: ObjectType = ObjectType.RIGID
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    spawn_cfg_addon: dict[str, Any] = {}
    asset_cfg_addon: dict[str, Any] = {}

    def __init__(
        self,
        instance_name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        scale: tuple[float, float, float] | None = None,
        **kwargs,
    ):
        name = instance_name if instance_name is not None else self.name
        scale = scale if scale is not None else self.scale
        super().__init__(
            name=name,
            prim_path=prim_path,
            tags=self.tags,
            usd_path=self.usd_path,
            object_type=self.object_type,
            scale=scale,
            initial_pose=initial_pose,
            spawn_cfg_addon=self.spawn_cfg_addon,
            asset_cfg_addon=self.asset_cfg_addon,
            **kwargs,
        )


@register_asset
class Microwave(LibraryObject, Openable):
    """A microwave oven."""

    # Only required when using Lightwheel SDK
    from lightwheel_sdk.loader import object_loader

    name = "microwave"
    tags = ["object", "openable"]
    file_path, object_name, metadata = object_loader.acquire_by_registry(
        registry_type="fixtures", file_name="Microwave039", file_type="USD"
    )
    usd_path = file_path
    scale = (0.596, 0.596, 0.596)
    object_type = ObjectType.ARTICULATION

    # Openable affordance parameters
    openable_joint_name = "microjoint"
    openable_threshold = 0.5  # Bistate threshold (open > threshold, closed <= threshold)

    def __init__(
        self, instance_name: str | None = None, prim_path: str | None = None, initial_pose: Pose | None = None
    ):
        super().__init__(
            instance_name=instance_name,
            prim_path=prim_path,
            initial_pose=initial_pose,
            openable_joint_name=self.openable_joint_name,
            openable_threshold=self.openable_threshold,
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
        registry_type="fixtures", file_name="CoffeeMachine108", file_type="USD"
    )
    usd_path = file_path
    scale = (0.583, 0.583, 0.583)
    object_type = ObjectType.ARTICULATION

    # Openable affordance parameters
    pressable_joint_name = "CoffeeMachine108_Button002_joint"
    pressedness_threshold = 0.5

    def __init__(
        self, instance_name: str | None = None, prim_path: str | None = None, initial_pose: Pose | None = None
    ):
        super().__init__(
            instance_name=instance_name,
            prim_path=prim_path,
            initial_pose=initial_pose,
            pressable_joint_name=self.pressable_joint_name,
            pressedness_threshold=self.pressedness_threshold,
        )


@register_asset
class StandMixer(LibraryObject, Turnable):
    """
    Stand mixer with a knob that can be turned to different levels.
    """

    name = "stand_mixer"
    tags = ["object", "turnable"]

    # TODO(xinjieyao, 2026.01.07): Trigger sync to production bucket for release.
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/lightwheel_StandMixer013/StandMixer013.usd"
    object_type = ObjectType.ARTICULATION

    # knob turnable affordance parameters
    turnable_joint_name = "knob_speed_joint"
    min_level_angle_deg = 40.0
    max_level_angle_deg = 280.0
    num_levels = 7

    def __init__(
        self, instance_name: str | None = None, prim_path: str | None = None, initial_pose: Pose | None = None
    ):
        super().__init__(
            instance_name=instance_name,
            prim_path=prim_path,
            initial_pose=initial_pose,
            turnable_joint_name=self.turnable_joint_name,
            min_level_angle_deg=self.min_level_angle_deg,
            max_level_angle_deg=self.max_level_angle_deg,
            num_levels=self.num_levels,
        )


@register_asset
class Peg(LibraryObject):
    """
    A peg.
    """

    name = "peg"
    tags = ["object"]
    usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Factory/factory_peg_8mm.usd"
    object_type = ObjectType.ARTICULATION
    scale = (3.0, 3.0, 3.0)
    spawn_cfg_addon = {
        "rigid_props": RIGID_BODY_PROPS_HIGH_PRECISION,
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.019),
        "collision_props": sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    }
    asset_cfg_addon = {
        "init_state": EMPTY_ARTICULATION_INIT_STATE_CFG,
    }


@register_asset
class Hole(LibraryObject):
    """
    A hole.
    """

    name = "hole"
    tags = ["object"]
    usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Factory/factory_hole_8mm.usd"
    object_type = ObjectType.ARTICULATION
    scale = (3.0, 3.0, 3.0)
    spawn_cfg_addon = {
        "rigid_props": RIGID_BODY_PROPS_HIGH_PRECISION,
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.05),
        "collision_props": sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    }
    asset_cfg_addon = {
        "init_state": EMPTY_ARTICULATION_INIT_STATE_CFG,
    }


@register_asset
class SmallGear(LibraryObject):
    """
    A small gear.
    """

    name = "small_gear"
    tags = ["object"]
    usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Factory/factory_gear_small.usd"
    object_type = ObjectType.ARTICULATION
    scale = (2.0, 2.0, 2.0)
    spawn_cfg_addon = {
        "rigid_props": RIGID_BODY_PROPS_MEDIUM_PRECISION,
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.019),
        "collision_props": sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    }
    asset_cfg_addon = {
        "init_state": EMPTY_ARTICULATION_INIT_STATE_CFG,
    }


@register_asset
class LargeGear(LibraryObject):
    """
    A large gear.
    """

    name = "large_gear"
    tags = ["object"]
    usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Factory/factory_gear_large.usd"
    object_type = ObjectType.ARTICULATION
    scale = (2.0, 2.0, 2.0)
    spawn_cfg_addon = {
        "rigid_props": RIGID_BODY_PROPS_MEDIUM_PRECISION,
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.019),
        "collision_props": sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    }
    asset_cfg_addon = {
        "init_state": EMPTY_ARTICULATION_INIT_STATE_CFG,
    }


@register_asset
class GearBase(LibraryObject):
    """
    Gear base.
    """

    name = "gear_base"
    tags = ["object"]
    usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Factory/factory_gear_base.usd"
    object_type = ObjectType.ARTICULATION
    scale = (2.0, 2.0, 2.0)
    spawn_cfg_addon = {
        "rigid_props": RIGID_BODY_PROPS_MEDIUM_PRECISION,
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.05),
        "collision_props": sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    }
    asset_cfg_addon = {
        "init_state": EMPTY_ARTICULATION_INIT_STATE_CFG,
    }


@register_asset
class MediumGear(LibraryObject):
    """
    A medium gear.
    """

    name = "medium_gear"
    tags = ["object"]
    usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Factory/factory_gear_medium.usd"
    object_type = ObjectType.ARTICULATION
    scale = (2.0, 2.0, 2.0)
    spawn_cfg_addon = {
        "rigid_props": RIGID_BODY_PROPS_MEDIUM_PRECISION,
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.019),
        "collision_props": sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    }
    asset_cfg_addon = {
        "init_state": EMPTY_ARTICULATION_INIT_STATE_CFG,
    }


# --- Lightwheel ARTICULATION objects ---


@register_asset
class Microwave011(LibraryObject, Openable):
    name = "microwave_011"
    tags = ["object", "lightwheel", "openable"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Microwave011/Microwave011.usd")
    scale = (0.615, 0.615, 0.615)
    object_type = ObjectType.ARTICULATION
    openable_joint_name = "microjoint"
    openable_threshold = 0.5

    def __init__(self, instance_name=None, prim_path=None, initial_pose=None):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose,
                         openable_joint_name=self.openable_joint_name, openable_threshold=self.openable_threshold)


@register_asset
class Microwave031(LibraryObject, Openable):
    name = "microwave_031"
    tags = ["object", "lightwheel", "openable"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Microwave031/Microwave031.usd")
    scale = (0.642, 0.642, 0.642)
    object_type = ObjectType.ARTICULATION
    openable_joint_name = "microjoint"
    openable_threshold = 0.5

    def __init__(self, instance_name=None, prim_path=None, initial_pose=None):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose,
                         openable_joint_name=self.openable_joint_name, openable_threshold=self.openable_threshold)


@register_asset
class Microwave032(LibraryObject, Openable):
    name = "microwave_032"
    tags = ["object", "lightwheel", "openable"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Microwave032/Microwave032.usd")
    scale = (0.535, 0.535, 0.535)
    object_type = ObjectType.ARTICULATION
    openable_joint_name = "microjoint"
    openable_threshold = 0.5

    def __init__(self, instance_name=None, prim_path=None, initial_pose=None):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose,
                         openable_joint_name=self.openable_joint_name, openable_threshold=self.openable_threshold)


@register_asset
class Microwave034(LibraryObject, Openable):
    name = "microwave_034"
    tags = ["object", "lightwheel", "openable"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Microwave034/Microwave034.usd")
    scale = (0.51, 0.51, 0.51)
    object_type = ObjectType.ARTICULATION
    openable_joint_name = "microjoint"
    openable_threshold = 0.5

    def __init__(self, instance_name=None, prim_path=None, initial_pose=None):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose,
                         openable_joint_name=self.openable_joint_name, openable_threshold=self.openable_threshold)


@register_asset
class Microwave036(LibraryObject, Openable):
    name = "microwave_036"
    tags = ["object", "lightwheel", "openable"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Microwave036/Microwave036.usd")
    scale = (0.553, 0.553, 0.553)
    object_type = ObjectType.ARTICULATION
    openable_joint_name = "microjoint"
    openable_threshold = 0.5

    def __init__(self, instance_name=None, prim_path=None, initial_pose=None):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose,
                         openable_joint_name=self.openable_joint_name, openable_threshold=self.openable_threshold)


@register_asset
class Microwave037(LibraryObject, Openable):
    name = "microwave_037"
    tags = ["object", "lightwheel", "openable"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Microwave037/Microwave037.usd")
    scale = (0.553, 0.553, 0.553)
    object_type = ObjectType.ARTICULATION
    openable_joint_name = "microjoint"
    openable_threshold = 0.5

    def __init__(self, instance_name=None, prim_path=None, initial_pose=None):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose,
                         openable_joint_name=self.openable_joint_name, openable_threshold=self.openable_threshold)


@register_asset
class Microwave041(LibraryObject, Openable):
    name = "microwave_041"
    tags = ["object", "lightwheel", "openable"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Microwave041/Microwave041.usd")
    scale = (0.571, 0.571, 0.571)
    object_type = ObjectType.ARTICULATION
    openable_joint_name = "microjoint"
    openable_threshold = 0.5

    def __init__(self, instance_name=None, prim_path=None, initial_pose=None):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose,
                         openable_joint_name=self.openable_joint_name, openable_threshold=self.openable_threshold)


@register_asset
class Microwave045(LibraryObject, Openable):
    name = "microwave_045"
    tags = ["object", "lightwheel", "openable"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Microwave045/Microwave045.usd")
    scale = (0.716, 0.716, 0.716)
    object_type = ObjectType.ARTICULATION
    openable_joint_name = "microjoint"
    openable_threshold = 0.5

    def __init__(self, instance_name=None, prim_path=None, initial_pose=None):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose,
                         openable_joint_name=self.openable_joint_name, openable_threshold=self.openable_threshold)


@register_asset
class Microwave089(LibraryObject, Openable):
    name = "microwave_089"
    tags = ["object", "lightwheel", "openable"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Microwave089/Microwave089.usd")
    scale = (0.6, 0.6, 0.6)
    object_type = ObjectType.ARTICULATION
    openable_joint_name = "microjoint"
    openable_threshold = 0.5

    def __init__(self, instance_name=None, prim_path=None, initial_pose=None):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose,
                         openable_joint_name=self.openable_joint_name, openable_threshold=self.openable_threshold)


@register_asset
class MokaPot001(LibraryObject, Openable):
    name = "moka_pot_001"
    tags = ["object", "lightwheel", "openable"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/MokaPot001/MokaPot001.usd")
    object_type = ObjectType.ARTICULATION
    openable_joint_name = "MokaPot001_Lid_joint"
    openable_threshold = 0.5

    def __init__(self, instance_name=None, prim_path=None, initial_pose=None):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose,
                         openable_joint_name=self.openable_joint_name, openable_threshold=self.openable_threshold)


@register_asset
class SaladDressing001(LibraryObject, Openable):
    name = "salad_dressing_001"
    tags = ["object", "lightwheel", "openable"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/SaladDressing001/SaladDressing001.usd")
    object_type = ObjectType.ARTICULATION
    openable_joint_name = "SaladDressing001_Lid_joint"
    openable_threshold = 0.5

    def __init__(self, instance_name=None, prim_path=None, initial_pose=None):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose,
                         openable_joint_name=self.openable_joint_name, openable_threshold=self.openable_threshold)


@register_asset
class Toaster039(LibraryObject, Pressable):
    name = "toaster_039"
    tags = ["object", "lightwheel", "pressable"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Toaster039/Toaster039.usd")
    object_type = ObjectType.ARTICULATION
    pressable_joint_name = "lever_joint"
    pressedness_threshold = 0.5

    def __init__(self, instance_name=None, prim_path=None, initial_pose=None):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose,
                         pressable_joint_name=self.pressable_joint_name, pressedness_threshold=self.pressedness_threshold)


@register_asset
class Toaster052(LibraryObject, Pressable):
    name = "toaster_052"
    tags = ["object", "lightwheel", "pressable"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Toaster052/Toaster052.usd")
    object_type = ObjectType.ARTICULATION
    pressable_joint_name = "lever_joint"
    pressedness_threshold = 0.5

    def __init__(self, instance_name=None, prim_path=None, initial_pose=None):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose,
                         pressable_joint_name=self.pressable_joint_name, pressedness_threshold=self.pressedness_threshold)

# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from typing import Any

import isaaclab.sim as sim_utils
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
    "https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/"
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
        self, instance_name: str | None = None, prim_path: str | None = None, initial_pose: Pose | None = None, **kwargs
    ):
        name = instance_name if instance_name is not None else self.name
        super().__init__(
            name=name,
            prim_path=prim_path,
            tags=self.tags,
            usd_path=self.usd_path,
            object_type=self.object_type,
            scale=self.scale,
            initial_pose=initial_pose,
            spawn_cfg_addon=self.spawn_cfg_addon,
            asset_cfg_addon=self.asset_cfg_addon,
            **kwargs,
        )


@register_asset
class CrackerBox(LibraryObject):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    name = "cracker_box"
    tags = ["object"]
    usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd"

    def __init__(
        self, instance_name: str | None = None, prim_path: str | None = None, initial_pose: Pose | None = None
    ):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class MustardBottle(LibraryObject):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    name = "mustard_bottle"
    tags = ["object"]
    usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd"

    def __init__(
        self, instance_name: str | None = None, prim_path: str | None = None, initial_pose: Pose | None = None
    ):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class SugarBox(LibraryObject):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    name = "sugar_box"
    tags = ["object"]
    usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd"

    def __init__(
        self, instance_name: str | None = None, prim_path: str | None = None, initial_pose: Pose | None = None
    ):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class TomatoSoupCan(LibraryObject):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    name = "tomato_soup_can"
    tags = ["object"]
    usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd"

    def __init__(
        self, instance_name: str | None = None, prim_path: str | None = None, initial_pose: Pose | None = None
    ):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class PowerDrill(LibraryObject):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    name = "power_drill"
    tags = ["object"]
    usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Arena/assets/object_library/power_drill_physics/power_drill_physics.usd"

    def __init__(
        self, instance_name: str | None = None, prim_path: str | None = None, initial_pose: Pose | None = None
    ):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose)


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
        registry_type="fixtures", registry_name=["coffee_machine"], file_type="USD"
    )
    usd_path = file_path
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
class OfficeTable(LibraryObject):
    """
    A basic office table.
    """

    name = "office_table"
    tags = ["object"]
    usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Mimic/nut_pour_task/nut_pour_assets/table.usd"
    default_prim_path = "{ENV_REGEX_NS}/office_table"
    scale = (1.0, 1.0, 0.7)

    def __init__(
        self, instance_name: str | None = None, prim_path: str | None = None, initial_pose: Pose | None = None
    ):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class BlueSortingBin(LibraryObject):
    """
    A blue plastic sorting bin.
    """

    name = "blue_sorting_bin"
    tags = ["object"]
    usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Mimic/exhaust_pipe_task/exhaust_pipe_assets/blue_sorting_bin.usd"
    default_prim_path = "{ENV_REGEX_NS}/blue_sorting_bin"
    scale = (4.0, 2.0, 1.0)

    def __init__(
        self, instance_name: str | None = None, prim_path: str | None = None, initial_pose: Pose | None = None
    ):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class BlueExhaustPipe(LibraryObject):
    """
    A blue exhaust pipe.
    """

    name = "blue_exhaust_pipe"
    tags = ["object"]
    usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Mimic/exhaust_pipe_task/exhaust_pipe_assets/blue_exhaust_pipe.usd"
    default_prim_path = "{ENV_REGEX_NS}/blue_exhaust_pipe"
    scale = (0.55, 0.55, 1.4)

    def __init__(
        self, instance_name: str | None = None, prim_path: str | None = None, initial_pose: Pose | None = None
    ):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class BrownBox(LibraryObject):
    """
    A brown box.
    """

    name = "brown_box"
    tags = ["object"]
    usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Arena/assets/object_library/brown_box/brown_box.usd"
    default_prim_path = "{ENV_REGEX_NS}/brown_box"
    scale = (1.0, 1.0, 1.0)

    def __init__(
        self, instance_name: str | None = None, prim_path: str | None = None, initial_pose: Pose | None = None
    ):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class Mug(LibraryObject, Placeable):
    """
    A mug.
    """

    name = "mug"
    tags = ["object"]
    usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Objects/Mug/mug.usd"
    object_type = ObjectType.RIGID
    default_prim_path = "{ENV_REGEX_NS}/Mug"
    scale = (1.0, 1.0, 1.0)

    # Placeable affordance parameters
    upright_axis_name = "z"
    orientation_threshold = 0.5

    def __init__(
        self, instance_name: str | None = None, prim_path: str | None = None, initial_pose: Pose | None = None
    ):
        super().__init__(
            instance_name=instance_name,
            prim_path=prim_path,
            initial_pose=initial_pose,
            upright_axis_name=self.upright_axis_name,
            orientation_threshold=self.orientation_threshold,
        )
        RIGID_BODY_PROPS = sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )
        self.object_cfg.spawn.rigid_props = RIGID_BODY_PROPS
        self.object_cfg.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=0.25)


@register_asset
class GroundPlane(LibraryObject):
    """
    A ground plane.
    """

    name = "ground_plane"
    tags = ["ground_plane"]
    # Setting a global prim path for the ground plane. Will not get repeated for each environment.
    default_prim_path = "/World/GroundPlane"
    object_type = ObjectType.SPAWNER
    default_spawner_cfg = GroundPlaneCfg()

    def __init__(
        self,
        instance_name: str | None = None,
        prim_path: str | None = default_prim_path,
        initial_pose: Pose | None = None,
        spawner_cfg: sim_utils.GroundPlaneCfg = default_spawner_cfg,
    ):
        self.spawner_cfg = spawner_cfg
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class Light(LibraryObject):
    """
    A dome light.
    """

    name = "light"
    tags = ["light"]
    # Setting a global prim path for the dome light. Will not get repeated for each environment.
    default_prim_path = "/World/Light"
    object_type = ObjectType.SPAWNER
    default_spawner_cfg = sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0)

    def __init__(
        self,
        instance_name: str | None = None,
        prim_path: str | None = default_prim_path,
        initial_pose: Pose | None = None,
        spawner_cfg: sim_utils.LightCfg = default_spawner_cfg,
    ):
        self.spawner_cfg = spawner_cfg
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class DexCube(LibraryObject):
    """
    A cube.
    """

    name = "dex_cube"
    tags = ["object"]
    usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"
    scale = (0.8, 0.8, 0.8)
    object_type = ObjectType.RIGID

    def __init__(
        self, instance_name: str | None = None, prim_path: str | None = None, initial_pose: Pose | None = None
    ):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose)


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


@register_asset
class SweetPotato(LibraryObject):
    """
    SweetPotato
    """

    # Only required when using Lightwheel SDK
    from lightwheel_sdk.loader import object_loader

    name = "sweet_potato"
    tags = ["object", "vegetable", "graspable"]
    file_path, object_name, metadata = object_loader.acquire_by_registry(
        registry_type="objects", file_name="SweetPotato005", file_type="USD"
    )
    usd_path = file_path
    object_type = ObjectType.RIGID
    scale = (1.3, 1.3, 1.3)

    def __init__(
        self, instance_name: str | None = None, prim_path: str | None = None, initial_pose: Pose | None = None
    ):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose)

# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from typing import TYPE_CHECKING, Any

import isaaclab.sim as sim_utils

if TYPE_CHECKING:
    from isaaclab_arena.assets.hdr_image import HDRImage
from isaaclab.assets import RigidObjectCfg
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
class CrackerBox(LibraryObject):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    name = "cracker_box"
    tags = ["object"]
    usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd"

    def __init__(
        self,
        instance_name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        scale: tuple[float, float, float] | None = None,
    ):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose, scale=scale)


@register_asset
class MustardBottle(LibraryObject):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    name = "mustard_bottle"
    tags = ["object"]
    usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd"

    def __init__(
        self,
        instance_name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        scale: tuple[float, float, float] | None = None,
    ):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose, scale=scale)


@register_asset
class SugarBox(LibraryObject):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    name = "sugar_box"
    tags = ["object"]
    usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd"

    def __init__(
        self,
        instance_name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        scale: tuple[float, float, float] | None = None,
    ):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose, scale=scale)


@register_asset
class TomatoSoupCan(LibraryObject):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    name = "tomato_soup_can"
    tags = ["object"]
    usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd"

    def __init__(
        self,
        instance_name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        scale: tuple[float, float, float] | None = None,
    ):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose, scale=scale)


@register_asset
class PowerDrill(LibraryObject):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    name = "power_drill"
    tags = ["object"]
    usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Arena/assets/object_library/power_drill_physics/power_drill_physics.usd"

    def __init__(
        self,
        instance_name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        scale: tuple[float, float, float] | None = None,
    ):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose, scale=scale)


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
        registry_type="fixtures", file_name="CoffeeMachine108", file_type="USD"
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
    scale = (1.0, 1.0, 0.7)

    def __init__(
        self,
        instance_name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        scale: tuple[float, float, float] | None = None,
    ):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose, scale=scale)


@register_asset
class BlueSortingBin(LibraryObject):
    """
    A blue plastic sorting bin.
    """

    name = "blue_sorting_bin"
    tags = ["object"]
    usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Mimic/exhaust_pipe_task/exhaust_pipe_assets/blue_sorting_bin.usd"
    scale = (4.0, 2.0, 1.0)

    def __init__(
        self,
        instance_name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        scale: tuple[float, float, float] | None = None,
    ):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose, scale=scale)


@register_asset
class BlueExhaustPipe(LibraryObject):
    """
    A blue exhaust pipe.
    """

    name = "blue_exhaust_pipe"
    tags = ["object"]
    usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Mimic/exhaust_pipe_task/exhaust_pipe_assets/blue_exhaust_pipe.usd"
    scale = (0.55, 0.55, 1.4)

    def __init__(
        self,
        instance_name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        scale: tuple[float, float, float] | None = None,
    ):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose, scale=scale)


@register_asset
class BrownBox(LibraryObject):
    """
    A brown box.
    """

    name = "brown_box"
    tags = ["object"]
    usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Arena/assets/object_library/brown_box/brown_box.usd"
    scale = (1.0, 1.0, 1.0)

    def __init__(
        self,
        instance_name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        scale: tuple[float, float, float] | None = None,
    ):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose, scale=scale)


@register_asset
class Mug(LibraryObject, Placeable):
    """
    A mug.
    """

    name = "mug"
    tags = ["object"]
    usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Objects/Mug/mug.usd"
    object_type = ObjectType.RIGID
    scale = (1.0, 1.0, 1.0)

    # Placeable affordance parameters
    upright_axis_name = "z"
    orientation_threshold = 0.5

    def __init__(
        self,
        instance_name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        scale: tuple[float, float, float] | None = None,
    ):
        super().__init__(
            instance_name=instance_name,
            prim_path=prim_path,
            initial_pose=initial_pose,
            scale=scale,
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
class Sphere(LibraryObject):
    """
    A sphere with rigid body physics (dynamic by default).
    """

    name = "sphere"
    tags = ["object"]
    object_type = ObjectType.SPAWNER
    scale = (1.0, 1.0, 1.0)
    default_spawner_cfg = sim_utils.SphereCfg(
        radius=0.1,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.25),
    )

    def __init__(
        self,
        instance_name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        scale: tuple[float, float, float] | None = None,
        spawner_cfg: sim_utils.SphereCfg = default_spawner_cfg,
    ):
        self.spawner_cfg = spawner_cfg
        super().__init__(
            instance_name=instance_name,
            prim_path=prim_path,
            initial_pose=initial_pose,
            scale=scale,
        )

    def _generate_spawner_cfg(self) -> RigidObjectCfg:
        object_cfg = RigidObjectCfg(
            prim_path=self.prim_path,
            spawn=self.spawner_cfg,
        )
        object_cfg = self._add_initial_pose_to_cfg(object_cfg)
        if self.initial_velocity is not None:
            object_cfg.init_state.lin_vel = self.initial_velocity.linear_xyz
            object_cfg.init_state.ang_vel = self.initial_velocity.angular_xyz
        return object_cfg

    def _requires_reset_pose_event(self) -> bool:
        return self.get_initial_pose() is not None and self.reset_pose


@register_asset
class DomeLight(LibraryObject):
    """A dome light, optionally textured with an HDR image environment map.

    The light can be used plain (uniform color) or combined with an HDR image
    texture via :meth:`add_hdr`::

        hdr = hdr_registry.get_hdr_by_name("home_office_robolab")()
        dome_light = asset_registry.get_asset_by_name("light")()
        dome_light.add_hdr(hdr)

    You can also pass the HDRImage instance directly to the constructor::

        hdr = hdr_registry.get_hdr_by_name("home_office_robolab")()
        dome_light = asset_registry.get_asset_by_name("light")(hdr=hdr)
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
        spawner_cfg: sim_utils.DomeLightCfg = default_spawner_cfg,
        hdr: "HDRImage | None" = None,  # noqa: F821
    ):
        self.spawner_cfg = spawner_cfg
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose)
        if hdr is not None:
            self.add_hdr(hdr)

    def add_hdr(self, hdr: "HDRImage") -> None:  # noqa: F821
        """Attach an HDR environment map texture to this dome light.

        Args:
            hdr: An :class:`HDRImage` instance.
        """
        from isaaclab_arena.assets.hdr_image import HDRImage

        assert isinstance(hdr, HDRImage), f"Expected an HDRImage instance, got {type(hdr)}"
        # Rebuild the spawner cfg with the HDR texture while preserving
        # user-specified intensity and other settings.
        self.spawner_cfg = sim_utils.DomeLightCfg(
            texture_file=hdr.texture_file,
            texture_format=hdr.texture_format,  # type: ignore[arg-type]
            intensity=self.spawner_cfg.intensity,
            color=self.spawner_cfg.color,
            visible_in_primary_ray=True,
        )
        # Re-initialize the object cfg so the scene picks up the change.
        self.object_cfg = self._init_object_cfg()


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
        self,
        instance_name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        scale: tuple[float, float, float] | None = None,
    ):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose, scale=scale)


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
class Broccoli(LibraryObject):
    """
    Brocolli
    """

    # Only required when using Lightwheel SDK
    from lightwheel_sdk.loader import object_loader

    name = "broccoli"
    tags = ["object", "vegetable", "graspable"]
    file_path, object_name, metadata = object_loader.acquire_by_registry(
        registry_type="objects", registry_name=["broccoli"], file_type="USD"
    )
    usd_path = file_path
    object_type = ObjectType.RIGID

    def __init__(
        self,
        instance_name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        scale: tuple[float, float, float] | None = None,
    ):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose, scale=scale)


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
    scale = (1.5, 1.5, 1.5)

    def __init__(
        self,
        instance_name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        scale: tuple[float, float, float] | None = None,
    ):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose, scale=scale)


@register_asset
class Jug(LibraryObject):
    """
    Jug
    """

    # Only required when using Lightwheel SDK
    from lightwheel_sdk.loader import object_loader

    name = "jug"
    tags = ["object", "graspable"]
    file_path, object_name, metadata = object_loader.acquire_by_registry(
        registry_type="objects", file_name="Jug005", file_type="USD"
    )
    usd_path = file_path
    object_type = ObjectType.RIGID
    scale = (2.0, 2.0, 2.0)

    def __init__(
        self,
        instance_name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        scale: tuple[float, float, float] | None = None,
    ):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose, scale=scale)


@register_asset
class BeerBottle(LibraryObject):
    """
    Beer Bottle
    """

    # Only required when using Lightwheel SDK
    from lightwheel_sdk.loader import object_loader

    name = "beer_bottle"
    tags = ["object", "graspable"]
    file_path, object_name, metadata = object_loader.acquire_by_registry(
        registry_type="objects", file_name="beer016", file_type="USD"
    )
    usd_path = file_path
    object_type = ObjectType.RIGID
    scale = (1.2, 1.2, 1.2)

    def __init__(
        self,
        instance_name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        scale: tuple[float, float, float] | None = None,
    ):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose, scale=scale)


@register_asset
class RedCube(LibraryObject):
    """
    A red cube.
    """

    name = "red_cube"
    tags = ["object"]

    # TODO(lanceli, 2026.02.04): There is a known bug where rigid body attributes can only bind to the root layer.
    # As a workaround, the original assets from ISAAC_NUCLEUS_DIR have been adjusted and uploaded to ISAAC_NUCLEUS_STAGING_DIR.
    # Once this bug is resolved, the original assets can be used instead.

    # usd_path =f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd" # not support, rigid body attribute need to be bind to root xform.
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/isaac_blocks/red_block_root_rigid.usd"

    object_type = ObjectType.RIGID
    default_prim_path = "{ENV_REGEX_NS}/RedCube"
    scale = (0.02, 0.02, 0.02)

    def __init__(
        self,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        scale: tuple[float, float, float] | None = None,
    ):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose, scale=scale)


@register_asset
class GreenCube(LibraryObject):
    """
    A green cube.
    """

    name = "green_cube"
    tags = ["object"]

    # TODO(lanceli, 2026.02.04): There is a known bug where rigid body attributes can only bind to the root layer.
    # As a workaround, the original assets from ISAAC_NUCLEUS_DIR have been adjusted and uploaded to ISAAC_NUCLEUS_STAGING_DIR.
    # Once this bug is resolved, the original assets can be used instead.

    # usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/green_block.usd" # not support, rigid body attribute need to be bind to root xform.
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/isaac_blocks/green_block_root_rigid.usd"
    object_type = ObjectType.RIGID
    default_prim_path = "{ENV_REGEX_NS}/GreenCube"
    scale = (0.02, 0.02, 0.02)

    def __init__(
        self,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        scale: tuple[float, float, float] | None = None,
    ):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose, scale=scale)


@register_asset
class RedContainer(LibraryObject):
    """
    A red container.
    """

    name = "red_container"
    tags = ["object"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/isaac_container/container_h20_red.usd"
    object_type = ObjectType.RIGID
    default_prim_path = "{ENV_REGEX_NS}/red_container"
    scale = (0.5, 0.5, 0.5)

    def __init__(
        self,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        scale: tuple[float, float, float] | None = None,
    ):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose, scale=scale)


@register_asset
class GreenContainer(LibraryObject):
    """
    A green container.
    """

    name = "green_container"
    tags = ["object"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/isaac_container/container_h20_green.usd"
    object_type = ObjectType.RIGID
    default_prim_path = "{ENV_REGEX_NS}/green_container"
    scale = (0.5, 0.5, 0.5)

    def __init__(
        self,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        scale: tuple[float, float, float] | None = None,
    ):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose, scale=scale)


@register_asset
class BlueBlockBasicRobolab(LibraryObject):
    name = "blue_block_basic_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/basic/blue_block.usd"
    )


@register_asset
class GreenBlockBasicRobolab(LibraryObject):
    name = "green_block_basic_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/basic/green_block.usd"
    )


@register_asset
class RedBlockBasicRobolab(LibraryObject):
    name = "red_block_basic_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/basic/red_block.usd"
    )


@register_asset
class YellowBlockBasicRobolab(LibraryObject):
    name = "yellow_block_basic_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/basic/yellow_block.usd"
    )


@register_asset
class Avocado01FruitsVeggiesRobolab(LibraryObject):
    name = "avocado01_fruits_veggies_robolab"
    tags = ["object", "graspable", "food", "fruit", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/fruits_veggies/avocado01.usd"


@register_asset
class Lemon01FruitsVeggiesRobolab(LibraryObject):
    name = "lemon_01_fruits_veggies_robolab"
    tags = ["object", "graspable", "food", "fruit", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/fruits_veggies/lemon_01.usd"


@register_asset
class Lemon02FruitsVeggiesRobolab(LibraryObject):
    name = "lemon_02_fruits_veggies_robolab"
    tags = ["object", "graspable", "food", "fruit", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/fruits_veggies/lemon_02.usd"


@register_asset
class Lime01FruitsVeggiesRobolab(LibraryObject):
    name = "lime01_fruits_veggies_robolab"
    tags = ["object", "graspable", "food", "fruit", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/fruits_veggies/lime01.usd"


@register_asset
class Lychee01FruitsVeggiesRobolab(LibraryObject):
    name = "lychee01_fruits_veggies_robolab"
    tags = ["object", "graspable", "food", "fruit", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/fruits_veggies/lychee01.usd"


@register_asset
class Orange01FruitsVeggiesRobolab(LibraryObject):
    name = "orange_01_fruits_veggies_robolab"
    tags = ["object", "graspable", "food", "fruit", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/fruits_veggies/orange_01.usd"


@register_asset
class Orange02FruitsVeggiesRobolab(LibraryObject):
    name = "orange_02_fruits_veggies_robolab"
    tags = ["object", "graspable", "food", "fruit", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/fruits_veggies/orange_02.usd"


@register_asset
class Pomegranate01FruitsVeggiesRobolab(LibraryObject):
    name = "pomegranate01_fruits_veggies_robolab"
    tags = ["object", "graspable", "food", "fruit", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/fruits_veggies/pomegranate01.usd"


@register_asset
class RedOnionFruitsVeggiesRobolab(LibraryObject):
    name = "red_onion_fruits_veggies_robolab"
    tags = ["object", "graspable", "food", "fruit", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/fruits_veggies/red_onion.usd"


@register_asset
class HammerHandalRobolab(LibraryObject):
    name = "hammer_handal_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/handal/hammer.usd"
    )


@register_asset
class Hammer1HandalRobolab(LibraryObject):
    name = "hammer_1_handal_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/handal/hammer_1.usd"
    )


@register_asset
class Hammer2HandalRobolab(LibraryObject):
    name = "hammer_2_handal_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/handal/hammer_2.usd"
    )


@register_asset
class Hammer3HandalRobolab(LibraryObject):
    name = "hammer_3_handal_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/handal/hammer_3.usd"
    )


@register_asset
class Hammer4HandalRobolab(LibraryObject):
    name = "hammer_4_handal_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/handal/hammer_4.usd"
    )


@register_asset
class Hammer5HandalRobolab(LibraryObject):
    name = "hammer_5_handal_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/handal/hammer_5.usd"
    )


@register_asset
class Hammer6HandalRobolab(LibraryObject):
    name = "hammer_6_handal_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/handal/hammer_6.usd"
    )


@register_asset
class Hammer7HandalRobolab(LibraryObject):
    name = "hammer_7_handal_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/handal/hammer_7.usd"
    )


@register_asset
class Hammer8HandalRobolab(LibraryObject):
    name = "hammer_8_handal_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/handal/hammer_8.usd"
    )


@register_asset
class LadleHandalRobolab(LibraryObject):
    name = "ladle_handal_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/handal/ladle.usd"


@register_asset
class MeasuringCupsHandalRobolab(LibraryObject):
    name = "measuring_cups_handal_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/handal/measuring_cups.usd"


@register_asset
class MeasuringCups1HandalRobolab(LibraryObject):
    name = "measuring_cups_1_handal_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/handal/measuring_cups_1.usd"


@register_asset
class MeasuringSpoonHandalRobolab(LibraryObject):
    name = "measuring_spoon_handal_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/handal/measuring_spoon.usd"


@register_asset
class SaladTongsHandalRobolab(LibraryObject):
    name = "salad_tongs_handal_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/handal/salad_tongs.usd"
    )


@register_asset
class ServingSpoonHandalRobolab(LibraryObject):
    name = "serving_spoon_handal_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/handal/serving_spoon.usd"


@register_asset
class ServingSpoonsHandalRobolab(LibraryObject):
    name = "serving_spoons_handal_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/handal/serving_spoons.usd"


@register_asset
class SpoonHandalRobolab(LibraryObject):
    name = "spoon_handal_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/handal/spoon.usd"


@register_asset
class Spoon1HandalRobolab(LibraryObject):
    name = "spoon_1_handal_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/handal/spoon_1.usd"
    )


@register_asset
class Spoon2HandalRobolab(LibraryObject):
    name = "spoon_2_handal_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/handal/spoon_2.usd"
    )


@register_asset
class AlphabetSoupCanHopeRobolab(LibraryObject):
    name = "alphabet_soup_can_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/alphabet_soup_can.usd"


@register_asset
class BbqSauceBottleHopeRobolab(LibraryObject):
    name = "bbq_sauce_bottle_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/bbq_sauce_bottle.usd"
    scale = (0.9, 0.9, 1.4)


@register_asset
class ButterHopeRobolab(LibraryObject):
    name = "butter_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/butter.usd"
    scale = (2.2, 2.0, 1.8)


@register_asset
class CannedMushroomsHopeRobolab(LibraryObject):
    name = "canned_mushrooms_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/canned_mushrooms.usd"


@register_asset
class CannedPeachesHopeRobolab(LibraryObject):
    name = "canned_peaches_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/canned_peaches.usd"
    )


@register_asset
class CannedTunaHopeRobolab(LibraryObject):
    name = "canned_tuna_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/canned_tuna.usd"
    )


@register_asset
class ChocolatePuddingMixHopeRobolab(LibraryObject):
    name = "chocolate_pudding_mix_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/chocolate_pudding_mix.usd"


@register_asset
class CornCanHopeRobolab(LibraryObject):
    name = "corn_can_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/corn_can.usd"
    )


@register_asset
class CreamCheeseHopeRobolab(LibraryObject):
    name = "cream_cheese_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/cream_cheese.usd"
    )


@register_asset
class GranolaBarsHopeRobolab(LibraryObject):
    name = "granola_bars_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/granola_bars.usd"
    )


@register_asset
class GreenBeansCanHopeRobolab(LibraryObject):
    name = "green_beans_can_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/green_beans_can.usd"


@register_asset
class KetchupBottleHopeRobolab(LibraryObject):
    name = "ketchup_bottle_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/ketchup_bottle.usd"
    )
    scale = (0.8, 0.8, 1.2)


@register_asset
class MacaroniAndCheeseHopeRobolab(LibraryObject):
    name = "macaroni_and_cheese_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/macaroni_and_cheese.usd"


@register_asset
class MayonnaiseBottleHopeRobolab(LibraryObject):
    name = "mayonnaise_bottle_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/mayonnaise_bottle.usd"
    scale = (0.9, 0.9, 1.2)


@register_asset
class MilkCartonHopeRobolab(LibraryObject):
    name = "milk_carton_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/milk_carton.usd"
    )


@register_asset
class MustardBottleHopeRobolab(LibraryObject):
    name = "mustard_bottle_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/mustard_bottle.usd"
    )


@register_asset
class OatmealRaisinCookiesHopeRobolab(LibraryObject):
    name = "oatmeal_raisin_cookies_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/oatmeal_raisin_cookies.usd"


@register_asset
class OrangeJuiceCartonHopeRobolab(LibraryObject):
    name = "orange_juice_carton_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/orange_juice_carton.usd"


@register_asset
class ParmesanCheeseCanisterHopeRobolab(LibraryObject):
    name = "parmesan_cheese_canister_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/parmesan_cheese_canister.usd"


@register_asset
class PeasAndCarrotsHopeRobolab(LibraryObject):
    name = "peas_and_carrots_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/peas_and_carrots.usd"


@register_asset
class PineappleSlicesCanHopeRobolab(LibraryObject):
    name = "pineapple_slices_can_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/pineapple_slices_can.usd"


@register_asset
class PittedCherriesHopeRobolab(LibraryObject):
    name = "pitted_cherries_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/pitted_cherries.usd"


@register_asset
class PopcornBoxHopeRobolab(LibraryObject):
    name = "popcorn_box_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/popcorn_box.usd"
    )


@register_asset
class RaisinBoxHopeRobolab(LibraryObject):
    name = "raisin_box_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/raisin_box.usd"
    )


@register_asset
class RanchDressingHopeRobolab(LibraryObject):
    name = "ranch_dressing_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/ranch_dressing.usd"
    )
    scale = (0.8, 0.8, 1.2)


@register_asset
class SpaghettiHopeRobolab(LibraryObject):
    name = "spaghetti_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/spaghetti.usd"
    )


@register_asset
class TomatoSauceCanHopeRobolab(LibraryObject):
    name = "tomato_sauce_can_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/tomato_sauce_can.usd"


@register_asset
class YogurtCupHopeRobolab(LibraryObject):
    name = "yogurt_cup_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/yogurt_cup.usd"
    )


@register_asset
class BbqSauceBottleHot3DRobolab(LibraryObject):
    name = "bbq_sauce_bottle_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/bbq_sauce_bottle.usd"


@register_asset
class BirdhouseHot3DRobolab(LibraryObject):
    name = "birdhouse_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/birdhouse.usd"
    )


@register_asset
class CeramicMugHot3DRobolab(LibraryObject):
    name = "ceramic_mug_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/ceramic_mug.usd"
    )


@register_asset
class ClayPlatesHot3DRobolab(LibraryObject):
    name = "clay_plates_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/clay_plates.usd"
    )


@register_asset
class CoffeePotHot3DRobolab(LibraryObject):
    name = "coffee_pot_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/coffee_pot.usd"
    )


@register_asset
class ComputerMouseHot3DRobolab(LibraryObject):
    name = "computer_mouse_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/computer_mouse.usd"


@register_asset
class DumbbellHot3DRobolab(LibraryObject):
    name = "dumbbell_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/dumbbell.usd"
    )


@register_asset
class FoamRollerHot3DRobolab(LibraryObject):
    name = "foam_roller_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/foam_roller.usd"
    )


@register_asset
class FrozenVegetableBlockHot3DRobolab(LibraryObject):
    name = "frozen_vegetable_block_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/frozen_vegetable_block.usd"


@register_asset
class FrozenWafflesHot3DRobolab(LibraryObject):
    name = "frozen_waffles_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/frozen_waffles.usd"


@register_asset
class GlassesHot3DRobolab(LibraryObject):
    name = "glasses_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/glasses.usd"
    )


@register_asset
class KeyboardHot3DRobolab(LibraryObject):
    name = "keyboard_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/keyboard.usd"
    )


@register_asset
class LizardFigurineHot3DRobolab(LibraryObject):
    name = "lizard_figurine_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/lizard_figurine.usd"


@register_asset
class MarkerHot3DRobolab(LibraryObject):
    name = "marker_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/marker.usd"


@register_asset
class MegaphoneHot3DRobolab(LibraryObject):
    name = "megaphone_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/megaphone.usd"
    )


@register_asset
class MilkCartonHot3DRobolab(LibraryObject):
    name = "milk_carton_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/milk_carton.usd"
    )


@register_asset
class MugHot3DRobolab(LibraryObject):
    name = "mug_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/mug.usd"


@register_asset
class MustardBottleHot3DRobolab(LibraryObject):
    name = "mustard_bottle_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/mustard_bottle.usd"


@register_asset
class OrangeJuiceCartonHot3DRobolab(LibraryObject):
    name = "orange_juice_carton_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/orange_juice_carton.usd"


@register_asset
class ParmesanCheeseCanisterHot3DRobolab(LibraryObject):
    name = "parmesan_cheese_canister_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/parmesan_cheese_canister.usd"


@register_asset
class PitcherHot3DRobolab(LibraryObject):
    name = "pitcher_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/pitcher.usd"
    )


@register_asset
class PotatoMasherHot3DRobolab(LibraryObject):
    name = "potato_masher_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/potato_masher.usd"
    )


@register_asset
class RemoteControlHot3DRobolab(LibraryObject):
    name = "remote_control_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/remote_control.usd"


@register_asset
class RubiksCubeHot3DRobolab(LibraryObject):
    name = "rubiks_cube_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/rubiks_cube.usd"
    )


@register_asset
class SaladDressingBottleHot3DRobolab(LibraryObject):
    name = "salad_dressing_bottle_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/salad_dressing_bottle.usd"


@register_asset
class SmartphoneHot3DRobolab(LibraryObject):
    name = "smartphone_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/smartphone.usd"
    )


@register_asset
class SoupCanHot3DRobolab(LibraryObject):
    name = "soup_can_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/soup_can.usd"
    )


@register_asset
class SpatulaHot3DRobolab(LibraryObject):
    name = "spatula_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/spatula.usd"
    )


@register_asset
class StorageBoxHot3DRobolab(LibraryObject):
    name = "storage_box_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/storage_box.usd"
    )


@register_asset
class TomatoSauceCanHot3DRobolab(LibraryObject):
    name = "tomato_sauce_can_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/tomato_sauce_can.usd"


@register_asset
class WoodenBowlHot3DRobolab(LibraryObject):
    name = "wooden_bowl_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/wooden_bowl.usd"
    )


@register_asset
class WoodenSpoonsHot3DRobolab(LibraryObject):
    name = "wooden_spoons_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/wooden_spoons.usd"
    )


@register_asset
class Apple01ObjaverseRobolab(LibraryObject):
    name = "apple_01_objaverse_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/objaverse/apple_01.usd"
    )


@register_asset
class Apple02ObjaverseRobolab(LibraryObject):
    name = "apple_02_objaverse_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/objaverse/apple_02.usd"
    )


@register_asset
class Bagel00ObjaverseRobolab(LibraryObject):
    name = "bagel_00_objaverse_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/objaverse/bagel_00.usd"
    )


@register_asset
class Bagel06ObjaverseRobolab(LibraryObject):
    name = "bagel_06_objaverse_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/objaverse/bagel_06.usd"
    )


@register_asset
class Bagel07ObjaverseRobolab(LibraryObject):
    name = "bagel_07_objaverse_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/objaverse/bagel_07.usd"
    )


@register_asset
class Baguette02ObjaverseRobolab(LibraryObject):
    name = "baguette_02_objaverse_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/objaverse/baguette_02.usd"


@register_asset
class GregorysCoffeeCupObjaverseRobolab(LibraryObject):
    name = "gregorys_coffee_cup_objaverse_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/objaverse/gregorys_coffee_cup.usd"


@register_asset
class LunchbagObjaverseRobolab(LibraryObject):
    name = "lunchbag_objaverse_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/objaverse/lunchbag.usd"
    )


@register_asset
class RedBellPepperObjaverseRobolab(LibraryObject):
    name = "red_bell_pepper_objaverse_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/objaverse/red_bell_pepper.usd"


@register_asset
class SnickersBarObjaverseRobolab(LibraryObject):
    name = "snickers_bar_objaverse_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/objaverse/snickers_bar.usd"


@register_asset
class BananaYcbRobolab(LibraryObject):
    name = "banana_ycb_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/ycb/banana.usd"


@register_asset
class BowlYcbRobolab(LibraryObject):
    name = "bowl_ycb_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/ycb/bowl.usd"


@register_asset
class BrickYcbRobolab(LibraryObject):
    name = "brick_ycb_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/ycb/brick.usd"


@register_asset
class CheezItYcbRobolab(LibraryObject):
    name = "cheez_it_ycb_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/ycb/cheez_it.usd"


@register_asset
class ChocolatePuddingYcbRobolab(LibraryObject):
    name = "chocolate_pudding_ycb_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/ycb/chocolate_pudding.usd"


@register_asset
class ClampYcbRobolab(LibraryObject):
    name = "clamp_ycb_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/ycb/clamp.usd"


@register_asset
class CoffeeCanYcbRobolab(LibraryObject):
    name = "coffee_can_ycb_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/ycb/coffee_can.usd"
    )


@register_asset
class CordlessDrillYcbRobolab(LibraryObject):
    name = "cordless_drill_ycb_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/ycb/cordless_drill.usd"
    )


@register_asset
class DryEraseMarkerYcbRobolab(LibraryObject):
    name = "dry_erase_marker_ycb_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/ycb/dry_erase_marker.usd"


@register_asset
class JelloYcbRobolab(LibraryObject):
    name = "jello_ycb_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/ycb/jello.usd"


@register_asset
class MugYcbRobolab(LibraryObject):
    name = "mug_ycb_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/ycb/mug.usd"


@register_asset
class MustardYcbRobolab(LibraryObject):
    name = "mustard_ycb_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/ycb/mustard.usd"


@register_asset
class PitcherYcbRobolab(LibraryObject):
    name = "pitcher_ycb_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/ycb/pitcher.usd"


@register_asset
class ScissorsYcbRobolab(LibraryObject):
    name = "scissors_ycb_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/ycb/scissors.usd"


@register_asset
class SoftScrubYcbRobolab(LibraryObject):
    name = "soft_scrub_ycb_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/ycb/soft_scrub.usd"
    )


@register_asset
class SpamCanYcbRobolab(LibraryObject):
    name = "spam_can_ycb_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/ycb/spam_can.usd"


@register_asset
class SpringClampYcbRobolab(LibraryObject):
    name = "spring_clamp_ycb_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/ycb/spring_clamp.usd"
    )


@register_asset
class SugarBoxYcbRobolab(LibraryObject):
    name = "sugar_box_ycb_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/ycb/sugar_box.usd"
    )


@register_asset
class TomatoSoupCanYcbRobolab(LibraryObject):
    name = "tomato_soup_can_ycb_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/ycb/tomato_soup_can.usd"
    )


@register_asset
class TunaCanYcbRobolab(LibraryObject):
    name = "tuna_can_ycb_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/ycb/tuna_can.usd"


@register_asset
class WoodBlockYcbRobolab(LibraryObject):
    name = "wood_block_ycb_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/ycb/wood_block.usd"
    )


@register_asset
class TableMapleRobolab(LibraryObject):
    name = "table_maple_robolab"
    tags = ["background", "fixture", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/fixtures/table_maple.usd"

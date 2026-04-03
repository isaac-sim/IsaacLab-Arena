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


# @register_asset
# class OfficeTable(LibraryObject):
#     """
#     A basic office table.
#     """

#     name = "office_table"
#     tags = ["object"]
#     usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Mimic/nut_pour_task/nut_pour_assets/table.usd"
#     scale = (1.0, 1.0, 0.7)

#     def __init__(
#         self,
#         instance_name: str | None = None,
#         prim_path: str | None = None,
#         initial_pose: Pose | None = None,
#         scale: tuple[float, float, float] | None = None,
#     ):
#         super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose, scale=scale)


@register_asset
class BlueSortingBin(LibraryObject):
    """
    A blue plastic sorting bin.
    """

    name = "blue_sorting_bin"
    tags = ["object"]
    usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Mimic/exhaust_pipe_task/exhaust_pipe_assets/blue_sorting_bin.usd"
    scale = (1.747, 0.873, 0.437)

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





# ======================================================================
# RoboLab objects (auto-generated) — ALL objects with rich metadata
# Total: 311 objects from 8 datasets
# ======================================================================


@register_asset
class BlueBlockBasicRobolab(LibraryObject):
    """This is a blue colored block."""
    name = "blue_block_basic_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/basic/blue_block.usd"
    )
    dims = (0.0470, 0.0470, 0.0470)
    # Physics material: dynamic_friction=1.0, static_friction=0.800000011920929, restitution=0.0


@register_asset
class GreenBlockBasicRobolab(LibraryObject):
    """This is a green colored block."""
    name = "green_block_basic_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/basic/green_block.usd"
    )
    dims = (0.0470, 0.0470, 0.0470)
    # Physics material: dynamic_friction=1.0, static_friction=0.800000011920929, restitution=0.0


@register_asset
class RedBlockBasicRobolab(LibraryObject):
    """This is a red colored block."""
    name = "red_block_basic_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/basic/red_block.usd"
    )
    dims = (0.0470, 0.0470, 0.0470)
    # Physics material: dynamic_friction=1.0, static_friction=0.800000011920929, restitution=0.0


@register_asset
class YellowBlockBasicRobolab(LibraryObject):
    """This is a yellow colored block."""
    name = "yellow_block_basic_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/basic/yellow_block.usd"
    )
    dims = (0.0470, 0.0470, 0.0470)
    # Physics material: dynamic_friction=1.0, static_friction=0.800000011920929, restitution=0.0


@register_asset
class Avocado01FruitsVeggiesRobolab(LibraryObject):
    """This avocado has a dark green, bumpy skin with a pear-like shape. Its interior reveals a creamy, light green flesh surrounding a large, smooth brown seed."""
    name = "avocado01_fruits_veggies_robolab"
    tags = ["object", "graspable", "food", "fruit", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/fruits_veggies/avocado01.usd"
    )
    dims = (0.0611, 0.0607, 0.0920)
    # Physics material: dynamic_friction=5.0, static_friction=5.0, restitution=0.10000000149011612


@register_asset
class Lemon01FruitsVeggiesRobolab(LibraryObject):
    """This lemon is oval-shaped with a bright yellow, slightly textured rind. It has a fresh, citrus scent and a small protruding stem at one end."""
    name = "lemon_01_fruits_veggies_robolab"
    tags = ["object", "graspable", "food", "fruit", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/fruits_veggies/lemon_01.usd"
    )
    dims = (0.0762, 0.0496, 0.0507)
    # Physics material: dynamic_friction=5.0, static_friction=5.0, restitution=0.10000000149011612


@register_asset
class Lemon02FruitsVeggiesRobolab(LibraryObject):
    """This lemon is oval-shaped with a textured, bright yellow rind. It has a slightly pointed end and a fresh, citrus aroma, characteristic of its juicy and tangy interior."""
    name = "lemon_02_fruits_veggies_robolab"
    tags = ["object", "graspable", "food", "fruit", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/fruits_veggies/lemon_02.usd"
    )
    dims = (0.0604, 0.0397, 0.0395)
    # Physics material: dynamic_friction=5.0, static_friction=5.0, restitution=0.10000000149011612


@register_asset
class Lime01FruitsVeggiesRobolab(LibraryObject):
    """This lime is small and round with a vibrant green, slightly bumpy skin. It has an opaque appearance and a fresh, citrusy aroma, typical of organic limes."""
    name = "lime01_fruits_veggies_robolab"
    tags = ["object", "graspable", "food", "fruit", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/fruits_veggies/lime01.usd"
    )
    dims = (0.0755, 0.0613, 0.0600)
    # Physics material: dynamic_friction=5.0, static_friction=5.0, restitution=0.10000000149011612


@register_asset
class Lychee01FruitsVeggiesRobolab(LibraryObject):
    """This lychee has a rough, textured pinkish-red skin with small, bumpy protrusions. Inside, it reveals a translucent white, juicy flesh surrounding a glossy brown seed."""
    name = "lychee01_fruits_veggies_robolab"
    tags = ["object", "graspable", "food", "fruit", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/fruits_veggies/lychee01.usd"
    )
    dims = (0.0404, 0.0415, 0.0502)
    # Physics material: dynamic_friction=5.0, static_friction=5.0, restitution=0.10000000149011612


@register_asset
class Orange01FruitsVeggiesRobolab(LibraryObject):
    """This orange is spherical with a textured, bright orange skin and a small, green stem. Its surface is dimpled, typical of citrus fruits, and it exudes a fresh, citrus scent."""
    name = "orange_01_fruits_veggies_robolab"
    tags = ["object", "graspable", "food", "fruit", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/fruits_veggies/orange_01.usd"
    )
    dims = (0.0725, 0.0725, 0.0832)
    # Physics material: dynamic_friction=5.0, static_friction=5.0, restitution=0.10000000149011612


@register_asset
class Orange02FruitsVeggiesRobolab(LibraryObject):
    """This orange has a vibrant, textured orange skin and a small, green stem. Its round shape and dimpled surface are characteristic of fresh citrus fruits."""
    name = "orange_02_fruits_veggies_robolab"
    tags = ["object", "graspable", "food", "fruit", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/fruits_veggies/orange_02.usd"
    )
    dims = (0.0718, 0.0718, 0.0735)
    # Physics material: dynamic_friction=5.0, static_friction=5.0, restitution=0.10000000149011612


@register_asset
class Pomegranate01FruitsVeggiesRobolab(LibraryObject):
    """This pomegranate is round and slightly uneven, with a deep red, leathery skin. It features a distinctive crown-like calyx at the top and contains numerous juicy, ruby-red seeds inside."""
    name = "pomegranate01_fruits_veggies_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/fruits_veggies/pomegranate01.usd"
    )
    dims = (0.1143, 0.1154, 0.1150)
    # Physics material: dynamic_friction=5.0, static_friction=5.0, restitution=0.10000000149011612


@register_asset
class RedOnionFruitsVeggiesRobolab(LibraryObject):
    """The red onion has a smooth, opaque purple-red skin with a spherical shape. Its layers are crisp and white with a hint of purple, offering a sharp, pungent flavor commonly used in cooking."""
    name = "red_onion_fruits_veggies_robolab"
    tags = ["object", "graspable", "food", "vegetable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/fruits_veggies/red_onion.usd"
    )
    dims = (0.0594, 0.0597, 0.0903)
    # Physics material: dynamic_friction=5.0, static_friction=5.0, restitution=0.10000000149011612


@register_asset
class HammerHandalRobolab(LibraryObject):
    """The hammer features a metallic head designed for striking, attached to a long wooden handle. The handle has a smooth finish and may bear some branding near the bottom."""
    name = "hammer_handal_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/handal/hammer.usd"
    )
    dims = (0.4104, 0.1460, 0.0383)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.5),
    }


@register_asset
class Hammer1HandalRobolab(LibraryObject):
    """This hammer features a metallic head with a smooth striking surface and a claw for nail removal. The handle is black with a textured grip, providing comfort during use."""
    name = "hammer_1_handal_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/handal/hammer_1.usd"
    )
    dims = (0.4087, 0.1489, 0.0361)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.5),
    }


@register_asset
class Hammer2HandalRobolab(LibraryObject):
    """The hammer features a metallic head with a smooth face and a curved claw, designed for driving nails and pulling them out. The handle is made of wood, displaying a natural finish with a light brown color."""
    name = "hammer_2_handal_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/handal/hammer_2.usd"
    )
    dims = (0.3311, 0.1297, 0.0313)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.5),
    }


@register_asset
class Hammer3HandalRobolab(LibraryObject):
    """This hammer features a gray metal head with a smooth face and a claw for pulling nails. The handle is ergonomic, predominantly black with red accents, providing a comfortable grip."""
    name = "hammer_3_handal_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/handal/hammer_3.usd"
    )
    dims = (0.3861, 0.1429, 0.0383)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.8999999761581421),
    }


@register_asset
class Hammer4HandalRobolab(LibraryObject):
    """This hammer features a metallic head with a smooth and shiny surface, positioned at both ends for versatility. The handle is made of a dark material with a textured blue grip for comfort and control."""
    name = "hammer_4_handal_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/handal/hammer_4.usd"
    )
    dims = (0.3308, 0.1281, 0.0333)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.5),
    }


@register_asset
class Hammer5HandalRobolab(LibraryObject):
    """This hammer features a metallic head with a smooth striking surface and a claw on one end. The handle is made of durable black material with an orange rubber grip for comfort."""
    name = "hammer_5_handal_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/handal/hammer_5.usd"
    )
    dims = (0.3367, 0.1267, 0.0344)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.8999999761581421),
    }


@register_asset
class Hammer6HandalRobolab(LibraryObject):
    """This hammer features a smooth, metallic head with a round end and a claw for nail removal. The handle is made of rubber with a blue color, providing a comfortable grip."""
    name = "hammer_6_handal_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/handal/hammer_6.usd"
    )
    dims = (0.4052, 0.1437, 0.0347)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.5),
    }


@register_asset
class Hammer7HandalRobolab(LibraryObject):
    """The hammer features a metal head with a smooth, flat striking surface and a claw for pulling nails. Its handle is bright red, ergonomically designed for a comfortable grip."""
    name = "hammer_7_handal_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/handal/hammer_7.usd"
    )
    dims = (0.3588, 0.1488, 0.0365)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.5),
    }


@register_asset
class Hammer8HandalRobolab(LibraryObject):
    """This hammer features a metallic head and a black rubberized handle for a comfortable grip. The head is designed with a smooth striking surface on one side and a claw for nail removal on the other."""
    name = "hammer_8_handal_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/handal/hammer_8.usd"
    )
    dims = (0.3346, 0.1462, 0.0348)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.5),
    }


@register_asset
class LadleHandalRobolab(LibraryObject):
    """The ladle features a broad, rounded scoop designed for serving liquids. It has a green plastic head and a wooden handle, giving it a modern yet rustic appearance."""
    name = "ladle_handal_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/handal/ladle.usd"
    )
    dims = (0.3199, 0.0779, 0.0623)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.15000000596046448),
    }


@register_asset
class MeasuringCupsHandalRobolab(LibraryObject):
    """These measuring cups are made of a soft green plastic. They feature a simple, rounded design with comfortable handles for easy pouring."""
    name = "measuring_cups_handal_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/handal/measuring_cups.usd"
    )
    dims = (0.1674, 0.0828, 0.0379)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.20000000298023224),
    }


@register_asset
class MeasuringCups1HandalRobolab(LibraryObject):
    """These measuring cups are made of flexible orange plastic and feature a handle for easy pouring. They are designed in a smooth, rounded shape, with a slight spout for accurate measurement."""
    name = "measuring_cups_1_handal_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/handal/measuring_cups_1.usd"
    )
    dims = (0.1734, 0.0952, 0.0521)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.5),
    }


@register_asset
class MeasuringSpoonHandalRobolab(LibraryObject):
    """These measuring spoons are bright yellow and have a smooth, glossy finish. They feature a rounded end for scooping, with a long handle that includes a hole for easy storage."""
    name = "measuring_spoon_handal_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/handal/measuring_spoon.usd"
    )
    dims = (0.1564, 0.0617, 0.0287)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.10000000149011612),
    }


@register_asset
class SaladTongsHandalRobolab(LibraryObject):
    """These salad tongs feature a pair of curved, claw-like ends for easy gripping. The handles are a warm brown color, contrasting with the pinkish tone of the gripping part."""
    name = "salad_tongs_handal_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/handal/salad_tongs.usd"
    )
    dims = (0.3160, 0.0603, 0.0402)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.30000001192092896),
    }


@register_asset
class ServingSpoonHandalRobolab(LibraryObject):
    """The serving spoon features a bright green, shallow bowl with elongated slots for draining liquids. It has a wooden handle that gives it a natural look, contrasting with the vibrant color of the spoon."""
    name = "serving_spoon_handal_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/handal/serving_spoon.usd"
    )
    dims = (0.3136, 0.0699, 0.0309)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.15000000596046448),
    }


@register_asset
class ServingSpoonsHandalRobolab(LibraryObject):
    """The serving spoons feature deep red silicone heads with a smooth texture. Their long handles are made of a light-colored wood, providing a comfortable grip."""
    name = "serving_spoons_handal_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/handal/serving_spoons.usd"
    )
    dims = (0.3152, 0.0689, 0.0310)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.15000000596046448),
    }


@register_asset
class SpoonHandalRobolab(LibraryObject):
    """This object consists of four spoons, each with a long handle and a shallow bowl. They are a muted gray color, with a smooth surface and a small hole in the handle for hanging."""
    name = "spoon_handal_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/handal/spoon.usd"
    )
    dims = (0.1240, 0.0342, 0.0176)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.05000000074505806),
    }


@register_asset
class Spoon1HandalRobolab(LibraryObject):
    """This object consists of four teal kitchen spoons with smooth, rounded bowls and elongated handles. Each spoon has a small hole at the end of the handle for easy hanging or storage."""
    name = "spoon_1_handal_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/handal/spoon_1.usd"
    )
    dims = (0.1284, 0.0419, 0.0213)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.10000000149011612),
    }


@register_asset
class Spoon2HandalRobolab(LibraryObject):
    """This object consists of two pairs of curved spoons with long handles. They feature a vibrant orange color and have a smooth, glossy finish."""
    name = "spoon_2_handal_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/handal/spoon_2.usd"
    )
    dims = (0.1166, 0.0208, 0.0108)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.10000000149011612),
    }


@register_asset
class BbqSauceBottleHopeRobolab(LibraryObject):
    """The bottle features a curved shape with a brownish-red color, suggesting its contents. It has a label that reads 'Tangy BBQ Sauce' in vibrant colors, with a star graphic, and a matching orange cap."""
    name = "bbq_sauce_bottle_hope_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/bbq_sauce_bottle.usd"
    )
    dims = (0.0435, 0.0646, 0.1483)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.4000000059604645),
    }


@register_asset
class ButterHopeRobolab(LibraryObject):
    """The object is a rectangular package of butter featuring a vibrant orange and red color scheme. It has images of a cow and text that reads 'FARM FRESH BUTTER' and 'UNSALTED'."""
    name = "butter_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/butter.usd"
    )
    dims = (0.0239, 0.1033, 0.0528)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.25),
    }


@register_asset
class CannedMushroomsHopeRobolab(LibraryObject):
    """The can features a vivid design with a purple and orange color scheme, prominently displaying the text 'Sliced Mushrooms'. It has a metallic finish and contains nutritional information on the label."""
    name = "canned_mushrooms_hope_robolab"
    tags = ["object", "graspable", "food", "vegetable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/canned_mushrooms.usd"
    )
    dims = (0.0708, 0.0659, 0.0333)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.4000000059604645),
    }


@register_asset
class CannedPeachesHopeRobolab(LibraryObject):
    """The can features a bright orange and yellow design with images of sliced peaches. It includes nutritional information and text indicating the product is 'Sliced Peaches in Their Own Natural Juices.'"""
    name = "canned_peaches_hope_robolab"
    tags = ["object", "graspable", "food", "fruit", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/canned_peaches.usd"
    )
    dims = (0.0710, 0.0659, 0.0578)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.3499999940395355),
    }


@register_asset
class CannedTunaHopeRobolab(LibraryObject):
    """The can is cylindrical with a colorful label featuring cartoon fish illustrations in shades of blue and green. The label includes text indicating 'Tuna Fish' and nutritional information on the back."""
    name = "canned_tuna_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/canned_tuna.usd"
    )
    dims = (0.0708, 0.0658, 0.0326)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.18000000715255737),
    }


@register_asset
class ChocolatePuddingMixHopeRobolab(LibraryObject):
    """The packaging features a rich brown color with the words 'Instant Chocolate Pudding' prominently displayed in white and red font. The boxes have images of creamy chocolate pudding and nutritional information on one side."""
    name = "chocolate_pudding_mix_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/chocolate_pudding_mix.usd"
    )
    dims = (0.0299, 0.0835, 0.0495)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.44999998807907104),
    }


@register_asset
class CornCanHopeRobolab(LibraryObject):
    """The can is cylindrical with a vibrant yellow and green design, prominently featuring the word 'CORN' in bold letters. One side displays nutritional facts and ingredients, printed in a clear, easy-to-read format."""
    name = "corn_can_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/corn_can.usd"
    )
    dims = (0.0709, 0.0661, 0.0580)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.3499999940395355),
    }


@register_asset
class CreamCheeseHopeRobolab(LibraryObject):
    """The object is a rectangular block of cream cheese, predominantly white with a smooth texture. It has blue and red text on the packaging, indicating the brand name and descriptions like 'Fresh Taste'."""
    name = "cream_cheese_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/cream_cheese.usd"
    )
    dims = (0.0242, 0.1036, 0.0532)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.22699999809265137),
    }


@register_asset
class GranolaBarsHopeRobolab(LibraryObject):
    """The granola bars are packaged in a colorful yellow box decorated with red and white text. The box displays images of oats and raisins, emphasizing the natural ingredients inside."""
    name = "granola_bars_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/granola_bars.usd"
    )
    dims = (0.0387, 0.1653, 0.1240)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.4000000059604645),
    }


@register_asset
class GreenBeansCanHopeRobolab(LibraryObject):
    """The can features a vibrant green design with illustrations of green beans and leaves. The label includes the text 'Green Beans' prominently, along with nutritional information on one side."""
    name = "green_beans_can_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/green_beans_can.usd"
    )
    dims = (0.0706, 0.0657, 0.0576)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.4000000059604645),
    }


@register_asset
class KetchupBottleHopeRobolab(LibraryObject):
    """The bottle is made of plastic with a smooth, rounded body and a squeeze top. It features a vibrant orange-red color and a label that reads 'Fancy Tomato Ketchup' in bold text."""
    name = "ketchup_bottle_hope_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/ketchup_bottle.usd"
    )
    dims = (0.0434, 0.0645, 0.1486)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.5),
    }


@register_asset
class MacaroniAndCheeseHopeRobolab(LibraryObject):
    """The packaging is a bright orange box with bold, playful text declaring 'SO CHEESY!' and 'MACARONI AND CHEESE.' It prominently features images of macaroni and cheese, suggesting a creamy texture and rich flavor."""
    name = "macaroni_and_cheese_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/macaroni_and_cheese.usd"
    )
    dims = (0.0402, 0.1235, 0.1663)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.21299999952316284),
    }


@register_asset
class MayonnaiseBottleHopeRobolab(LibraryObject):
    """The bottle is a squeeze-type with a slightly curved shape, finished in a light beige color. It features a blue cap and a label that prominently displays the word 'MAYO' along with graphics of mayonnaise."""
    name = "mayonnaise_bottle_hope_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/mayonnaise_bottle.usd"
    )
    dims = (0.0410, 0.0645, 0.1479)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.5),
    }


@register_asset
class MilkCartonHopeRobolab(LibraryObject):
    """The carton is predominantly orange with green and white accents, displaying the word 'Milk' prominently on all sides. It features a cartoon cow on the front and nutritional information on one side."""
    name = "milk_carton_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/milk_carton.usd"
    )
    dims = (0.0733, 0.0722, 0.1904)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.5),
    }


@register_asset
class MustardBottleHopeRobolab(LibraryObject):
    """The bottle is a vibrant yellow color with a glossy finish. It features a green label with white text that reads 'Spicy Yellow Mustard'."""
    name = "mustard_bottle_hope_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/mustard_bottle.usd"
    )
    dims = (0.0486, 0.0651, 0.1601)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.30000001192092896),
    }


@register_asset
class OatmealRaisinCookiesHopeRobolab(LibraryObject):
    """The packaging features a vibrant red and white color scheme with images of cookies prominently displayed. The front of the box is labeled 'Oatmeal Raisin Cookies' in bold letters, accompanied by graphics of the cookies themselves."""
    name = "oatmeal_raisin_cookies_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/oatmeal_raisin_cookies.usd"
    )
    dims = (0.0402, 0.1227, 0.1672)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.3400000035762787),
    }


@register_asset
class OrangeJuiceCartonHopeRobolab(LibraryObject):
    """The carton is primarily orange and yellow, featuring large images of sliced oranges. It has text that reads 'ORANGE JUICE' and indicates 'NO PULP' on the front."""
    name = "orange_juice_carton_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/orange_juice_carton.usd"
    )
    dims = (0.0728, 0.0716, 0.1925)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.5),
    }


@register_asset
class ParmesanCheeseCanisterHopeRobolab(LibraryObject):
    """The canister is cylindrical with a green and yellow striped design, featuring text that indicates it contains grated Parmesan cheese. It has a lid on top and nutritional facts printed on the side."""
    name = "parmesan_cheese_canister_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/parmesan_cheese_canister.usd"
    )
    dims = (0.0661, 0.0711, 0.1029)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.20000000298023224),
    }


@register_asset
class PeasAndCarrotsHopeRobolab(LibraryObject):
    """The object is a cylindrical can featuring a colorful label with green polka dots and orange accents. It has text that reads 'Peas & Carrots' on the front and nutritional information on the back."""
    name = "peas_and_carrots_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/peas_and_carrots.usd"
    )
    dims = (0.0706, 0.0659, 0.0585)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.4000000059604645),
    }


@register_asset
class PineappleSlicesCanHopeRobolab(LibraryObject):
    """The can features a bright blue background with yellow pineapple slice illustrations. It has a label with red text indicating 'Pineapple Slices' and nutritional information on the side."""
    name = "pineapple_slices_can_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/pineapple_slices_can.usd"
    )
    dims = (0.0696, 0.0657, 0.0576)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.5),
    }


@register_asset
class PittedCherriesHopeRobolab(LibraryObject):
    """The can is cylindrical with a purple label featuring bright red cherries and white text. The lid is metallic, and the nutritional facts are displayed on one side of the can."""
    name = "pitted_cherries_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/pitted_cherries.usd"
    )
    dims = (0.0711, 0.0662, 0.0582)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.5),
    }


@register_asset
class PopcornBoxHopeRobolab(LibraryObject):
    """The box is bright blue and features a playful design with yellow and white flowers. It prominently displays the word 'POPCORN' in bold red letters on the front."""
    name = "popcorn_box_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/popcorn_box.usd"
    )
    dims = (0.0383, 0.1265, 0.0850)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.20000000298023224),
    }


@register_asset
class RaisinBoxHopeRobolab(LibraryObject):
    """The box is brightly colored with an orange background and features a smiling sun graphic. It prominently displays the word 'RAISINS' along with illustrations of grapes."""
    name = "raisin_box_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/raisin_box.usd"
    )
    dims = (0.0398, 0.0859, 0.1232)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.25),
    }


@register_asset
class RanchDressingHopeRobolab(LibraryObject):
    """The bottle has a smooth, rounded shape with a prominent green cap. It features a label that displays the text 'Creamy Ranch Dressing' alongside images of vegetables."""
    name = "ranch_dressing_hope_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/ranch_dressing.usd"
    )
    dims = (0.0437, 0.0640, 0.1474)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.30000001192092896),
    }


@register_asset
class SpaghettiHopeRobolab(LibraryObject):
    """The packaging is a rectangular box with a vibrant red and green design. It prominently features the word 'SPAGHETTI' in bold letters along with the label 'Organic Whole Wheat'."""
    name = "spaghetti_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/spaghetti.usd"
    )
    dims = (0.0285, 0.2499, 0.0498)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.45399999618530273),
    }


@register_asset
class TomatoSauceCanHopeRobolab(LibraryObject):
    """The can is cylindrical with a glossy red and green label featuring the words 'TOMATO SAUCE' prominently displayed. It is adorned with images of tomatoes and includes nutrition facts and serving suggestions on one side."""
    name = "tomato_sauce_can_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/tomato_sauce_can.usd"
    )
    dims = (0.0702, 0.0665, 0.0828)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.4000000059604645),
    }


@register_asset
class YogurtCupHopeRobolab(LibraryObject):
    """The yogurt cup is cylindrical and features a vibrant orange color with a label indicating its strawberry flavor. It has a plastic lid on top and nutrition facts printed on the side."""
    name = "yogurt_cup_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/yogurt_cup.usd"
    )
    dims = (0.0680, 0.0679, 0.0537)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.20000000298023224),
    }


@register_asset
class BbqSauceBottleHot3dRobolab(LibraryObject):
    """The bottle has a mostly brown exterior with a red and yellow label featuring the text 'BBQ Sauce'. Its shape is distinctively tapered towards the bottom with a wide opening at the top."""
    name = "bbq_sauce_bottle_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/bbq_sauce_bottle.usd"
    )
    dims = (0.0438, 0.0644, 0.1452)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.5),
    }


@register_asset
class BirdhouseHot3dRobolab(LibraryObject):
    """This birdhouse features a vibrant blue roof and a cheerful yellow body, designed to resemble a small house. It includes decorative elements like windows, doors, and small plants, adding a whimsical touch to any garden."""
    name = "birdhouse_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/birdhouse.usd"
    )
    dims = (0.1800, 0.1808, 0.2328)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=1.5),
    }


@register_asset
class CeramicMugHot3dRobolab(LibraryObject):
    """The mug features a vibrant blue swirl pattern on a white ceramic surface, with a glossy finish. It has a contrasting yellow rim and a blue handle, creating a cheerful and decorative appearance."""
    name = "ceramic_mug_hot3d_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/ceramic_mug.usd"
    )
    dims = (0.0970, 0.1272, 0.0841)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.5),
    }


@register_asset
class ClayPlatesHot3dRobolab(LibraryObject):
    """These are circular plates made of clay, featuring a smooth, natural texture. They have a light brown color with subtle variations in tone and a slightly raised center."""
    name = "clay_plates_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/clay_plates.usd"
    )
    dims = (0.3006, 0.3020, 0.0471)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.5),
    }


@register_asset
class CoffeePotHot3dRobolab(LibraryObject):
    """The coffee pot has a sleek shape with a matte black finish, accented by a speckled pattern. It features a sturdy handle and a spout for easy pouring."""
    name = "coffee_pot_hot3d_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/coffee_pot.usd"
    )
    dims = (0.1457, 0.0849, 0.1509)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=1.5),
    }


@register_asset
class ComputerMouseHot3dRobolab(LibraryObject):
    """The object has a smooth, rounded shape with a matte black finish. It features buttons on the top and a scroll wheel, with a minimalist design that avoids bright colors or text."""
    name = "computer_mouse_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/computer_mouse.usd"
    )
    dims = (0.0700, 0.1098, 0.0422)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.10000000149011612),
    }


@register_asset
class DumbbellHot3dRobolab(LibraryObject):
    """These dumbbells feature a hexagonal shape with a matte green color. Each one is labeled with '5LB' to indicate its weight."""
    name = "dumbbell_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/dumbbell.usd"
    )
    dims = (0.0835, 0.1802, 0.0744)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=2.2699999809265137),
    }


@register_asset
class FoamRollerHot3dRobolab(LibraryObject):
    """The object has a rectangular shape with a textured surface on one side and a smooth side on the other. It features a dark brown color overall, with a lighter gray accent around the edges."""
    name = "foam_roller_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/foam_roller.usd"
    )
    dims = (0.0678, 0.1474, 0.0366)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=1.0),
    }


@register_asset
class FrozenVegetableBlockHot3dRobolab(LibraryObject):
    """The object is a rectangular block with a smooth surface, primarily green in color. It features a label on the side that includes text reading 'FLASH FROZEN' and 'MIXED VEGETABLES' with colorful dots across the design."""
    name = "frozen_vegetable_block_hot3d_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/frozen_vegetable_block.usd"
    )
    dims = (0.0673, 0.0967, 0.0204)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=1.0),
    }


@register_asset
class FrozenWafflesHot3dRobolab(LibraryObject):
    """The object is a rectangular block with a vibrant sky-blue color, reminiscent of a packaged food item. It features a playful design with yellow grid patterns and text that reads 'Frozen Waffles' along with other details printed on the sides."""
    name = "frozen_waffles_hot3d_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/frozen_waffles.usd"
    )
    dims = (0.0702, 0.1301, 0.0205)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.25),
    }


@register_asset
class GlassesHot3dRobolab(LibraryObject):
    """The object features a frame with two lenses and two arms extending from either side. The frame appears to be black, and the lenses are likely tinted."""
    name = "glasses_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/glasses.usd"
    )
    dims = (0.1469, 0.1638, 0.0459)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.10000000149011612),
    }


@register_asset
class KeyboardHot3dRobolab(LibraryObject):
    """The keyboard features a sleek, slim design with black keys and a matte finish. It includes a standard layout of letters, numbers, and function keys, with some symbols printed on the keys."""
    name = "keyboard_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/keyboard.usd"
    )
    dims = (0.4290, 0.1290, 0.0203)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.800000011920929),
    }


@register_asset
class LizardFigurineHot3dRobolab(LibraryObject):
    """The figurine resembles a lizard with a textured surface that mimics reptilian scales. It has a blend of green and brown colors, giving it a realistic appearance."""
    name = "lizard_figurine_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/lizard_figurine.usd"
    )
    dims = (0.2397, 0.1400, 0.1353)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.5),
    }


@register_asset
class MegaphoneHot3dRobolab(LibraryObject):
    """The megaphone has a streamlined cylindrical shape with a slightly tapered end and an open mouthpiece. It is colored in a subdued green tone, featuring a sturdy handle for easy grip."""
    name = "megaphone_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/megaphone.usd"
    )
    dims = (0.1198, 0.1602, 0.3012)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=1.5),
    }


@register_asset
class MugHot3dRobolab(LibraryObject):
    """This mug has a ribbed texture and a slightly curved handle. It is a solid light color, resembling unglazed ceramic."""
    name = "mug_hot3d_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/mug.usd"
    )
    dims = (0.1161, 0.0908, 0.0916)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.25),
    }


@register_asset
class PitcherHot3dRobolab(LibraryObject):
    """The object has a conical shape with a wide base that tapers towards the top. It is a light gray color and features a handle on one side for easy pouring."""
    name = "pitcher_hot3d_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/pitcher.usd"
    )
    dims = (0.1358, 0.0997, 0.1674)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=1.5),
    }


@register_asset
class PotatoMasherHot3dRobolab(LibraryObject):
    """The potato masher has a sturdy black handle with a wide, round mashing head featuring a circular design. It is predominantly black, designed for comfortable grip and effective mashing."""
    name = "potato_masher_hot3d_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/potato_masher.usd"
    )
    dims = (0.0899, 0.0901, 0.2830)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.25),
    }


@register_asset
class RemoteControlHot3dRobolab(LibraryObject):
    """The remote control features a sleek, curved design with a matte black finish. It has a few buttons on the top, including navigation and volume controls, all colored in subtle shades to blend with the overall dark aesthetic."""
    name = "remote_control_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/remote_control.usd"
    )
    dims = (0.0356, 0.0252, 0.1643)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.20000000298023224),
    }


@register_asset
class RubiksCubeHot3dRobolab(LibraryObject):
    """The object is a colorful 3D puzzle consisting of smaller cubes that can be rotated on each axis. Its faces feature a combination of colors, typically including red, green, blue, yellow, orange, and white."""
    name = "rubiks_cube_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/rubiks_cube.usd"
    )
    dims = (0.0583, 0.0578, 0.0580)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.20000000298023224),
    }


@register_asset
class SmartphoneHot3dRobolab(LibraryObject):
    """The object is a sleek, rectangular device with rounded edges, primarily in a black color. It features a smooth, glossy surface with no visible text or logos."""
    name = "smartphone_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/smartphone.usd"
    )
    dims = (0.0729, 0.1540, 0.0103)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.20000000298023224),
    }


@register_asset
class SoupCanHot3dRobolab(LibraryObject):
    """The can is cylindrical with a colorful design featuring vibrant orange and blue colors. It has text on the front that reads 'ALPHABET SOUP' and nutritional information on the back."""
    name = "soup_can_hot3d_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/soup_can.usd"
    )
    dims = (0.0707, 0.0658, 0.0818)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.3499999940395355),
    }


@register_asset
class SpatulaHot3dRobolab(LibraryObject):
    """The spatula has a flat, broad head with several slots for draining liquids. It features a long handle, both components being a deep red color."""
    name = "spatula_hot3d_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/spatula.usd"
    )
    dims = (0.3680, 0.3347, 0.0416)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.20000000298023224),
    }


@register_asset
class StorageBoxHot3dRobolab(LibraryObject):
    """This storage box has a rectangular shape with a smooth, light-colored wooden exterior. It features an open front for accessing contents and appears to have a soft padding or lining visible on the inside."""
    name = "storage_box_hot3d_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/storage_box.usd"
    )
    dims = (0.0856, 0.0659, 0.1236)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=1.5),
    }


@register_asset
class WoodenBowlHot3dRobolab(LibraryObject):
    """The bowl has a smooth, rounded shape with a warm, natural wood finish. Its surface features intricate patterns created by the grain of the wood, showcasing various shades of brown."""
    name = "wooden_bowl_hot3d_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/wooden_bowl.usd"
    )
    dims = (0.2800, 0.2801, 0.1300)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.5),
    }


@register_asset
class WoodenSpoonsHot3dRobolab(LibraryObject):
    """These wooden spoons have a smooth, light brown finish and feature a simple, elegant design. They come in various shapes, including flat and rounded heads, ideal for stirring and serving."""
    name = "wooden_spoons_hot3d_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/wooden_spoons.usd"
    )
    dims = (0.0684, 0.3094, 0.0179)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.20000000298023224),
    }


@register_asset
class AppleObjaverseRobolab(LibraryObject):
    name = "apple_objaverse_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/objaverse/apple_02.usd"
    )
    dims = (0.0702, 0.0754, 0.0733)
    # Physics material: dynamic_friction=10.0, static_friction=10.0, restitution=0.4000000059604645
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=830.0),
    }


@register_asset
class Apple01ObjaverseRobolab(LibraryObject):
    name = "apple_01_objaverse_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/objaverse/apple_01.usd"
    )
    dims = (0.0681, 0.0659, 0.0677)
    # Physics material: dynamic_friction=0.800000011920929, static_friction=0.800000011920929, restitution=0.0
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=980.0),
    }


@register_asset
class BagelObjaverseRobolab(LibraryObject):
    name = "bagel_objaverse_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/objaverse/bagel_06.usd"
    )
    dims = (0.0972, 0.1100, 0.0311)
    # Physics material: dynamic_friction=10.0, static_friction=10.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=400.0),
    }


@register_asset
class Bagel00ObjaverseRobolab(LibraryObject):
    name = "bagel_00_objaverse_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/objaverse/bagel_00.usd"
    )
    dims = (0.1000, 0.0948, 0.0312)
    # Physics material: dynamic_friction=10.0, static_friction=10.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=400.0),
    }


@register_asset
class GregorysCoffeeCupObjaverseRobolab(LibraryObject):
    name = "gregorys_coffee_cup_objaverse_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/objaverse/gregorys_coffee_cup.usd"
    )
    dims = (0.0701, 0.0701, 0.1100)


@register_asset
class LunchbagObjaverseRobolab(LibraryObject):
    name = "lunchbag_objaverse_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/objaverse/lunchbag.usd"
    )
    dims = (0.0995, 0.0525, 0.1450)
    # Physics material: dynamic_friction=10.0, static_friction=10.0, restitution=0.05000000074505806
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=650.0),
    }


@register_asset
class RedBellPepperObjaverseRobolab(LibraryObject):
    name = "red_bell_pepper_objaverse_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/objaverse/red_bell_pepper.usd"
    )
    dims = (0.0819, 0.0858, 0.0900)
    # Physics material: dynamic_friction=10.0, static_friction=10.0, restitution=0.4000000059604645
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=830.0),
    }


@register_asset
class SesameBagelObjaverseRobolab(LibraryObject):
    name = "sesame_bagel_objaverse_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/objaverse/bagel_07.usd"
    )
    dims = (50.0000, 50.0000, 0.0325)
    scale = (0.2, 0.2, 0.8)
    # Physics material: dynamic_friction=10.0, static_friction=10.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=400.0),
    }


@register_asset
class SnickersBarObjaverseRobolab(LibraryObject):
    name = "snickers_bar_objaverse_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/objaverse/snickers_bar.usd"
    )
    dims = (0.0283, 0.0825, 0.0172)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=400.0),
    }


@register_asset
class AntiquelvaseVompRobolab(LibraryObject):
    """This antique vase is crafted from opaque stone, featuring intricate carvings and a weathered patina. Its muted earth tones and classic silhouette evoke a sense of timeless elegance and historical charm."""
    name = "antiquelvase_vomp_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/antiquelvase/antiquelvase.usd"
    )
    dims = (1.1137, 0.9308, 0.9260)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=2850.0),
    }


@register_asset
class AntiquelvasesmallVompRobolab(LibraryObject):
    """This small antique vase is crafted from opaque stone, featuring intricate carvings and a smooth, rounded body. Its muted earth tones give it a timeless, elegant appearance, perfect for displaying delicate flowers."""
    name = "antiquelvasesmall_vomp_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/antiquelvasesmall/antiquelvasesmall.usd"
    )
    dims = (0.8632, 0.7140, 0.7347)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=2850.0),
    }


@register_asset
class AnzaLargeVompRobolab(LibraryObject):
    """This vase is crafted from opaque concrete, featuring a smooth, minimalist design with a large, cylindrical shape. Its neutral gray color complements modern interiors, emphasizing simplicity and elegance."""
    name = "anza_large_vomp_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/anza_large/anza_large.usd"
    )
    dims = (0.8636, 0.8636, 0.9172)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.009999999776482582
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=2500.0),
    }


@register_asset
class AnzaMediumVompRobolab(LibraryObject):
    """This vase is crafted from opaque concrete, featuring a smooth, minimalist design with a medium height. Its neutral gray color and clean lines make it a versatile piece for modern decor."""
    name = "anza_medium_vomp_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/anza_medium/anza_medium.usd"
    )
    dims = (0.6340, 0.6340, 0.4942)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.009999999776482582
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=2400.0),
    }


@register_asset
class BinA01VompRobolab(LibraryObject):
    """The bin is made of opaque plastic with a smooth, rectangular body and a secure-fitting lid. It features a neutral color, ideal for storage, and is designed for durability and easy stacking."""
    name = "bin_a01_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/bin_a01/bin_a01.usd"
    )
    dims = (0.0869, 0.1066, 0.0798)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class BinA02VompRobolab(LibraryObject):
    """This container is a sleek, cylindrical metal bin with a smooth, reflective surface and a simple, modern design. It features a flat lid and a sturdy base, ideal for waste disposal."""
    name = "bin_a02_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/bin_a02/bin_a02.usd"
    )
    dims = (0.1221, 0.1066, 0.0798)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=2700.0),
    }


@register_asset
class BinA03VompRobolab(LibraryObject):
    """This container is a sleek, cylindrical metal bin with a smooth, silver finish. It features a flat lid and a sturdy base, ideal for securely storing items."""
    name = "bin_a03_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/bin_a03/bin_a03.usd"
    )
    dims = (0.1571, 0.1011, 0.0798)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=2700.0),
    }


@register_asset
class BinA04VompRobolab(LibraryObject):
    """This container is a sleek, metallic bin with a cylindrical shape and a smooth, reflective surface. It features a flat lid and a simple, modern design suitable for various storage needs."""
    name = "bin_a04_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/bin_a04/bin_a04.usd"
    )
    dims = (0.2976, 0.1411, 0.1295)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=2700.0),
    }


@register_asset
class BinA05VompRobolab(LibraryObject):
    """This container is made of smooth metal with a cylindrical shape and a matte finish. It features a removable lid and a simple, modern design, suitable for storing various items."""
    name = "bin_a05_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/bin_a05/bin_a05.usd"
    )
    dims = (0.2976, 0.2004, 0.1573)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class BinA06VompRobolab(LibraryObject):
    """This container is a sleek, cylindrical metal bin with a smooth, reflective surface. It features a hinged lid for easy access and a minimalist design, ideal for modern spaces."""
    name = "bin_a06_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/bin_a06/bin_a06.usd"
    )
    scale = (0.689, 0.689, 0.689)
    dims = (0.5083, 0.3000, 0.1870)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=2700.0),
    }


@register_asset
class BinB01VompRobolab(LibraryObject):
    """This container is a cylindrical metal bin with a smooth, silver surface. It features a flat lid and a foot pedal for hands-free operation, ideal for waste disposal in kitchens or offices."""
    name = "bin_b01_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/bin_b01/bin_b01.usd"
    )
    dims = (0.0943, 0.0986, 0.0499)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=2700.0),
    }


@register_asset
class BinB02VompRobolab(LibraryObject):
    """This container is a sleek, cylindrical metal bin with a smooth, silver finish. It features a flat lid and a foot pedal for hands-free operation, ideal for waste disposal."""
    name = "bin_b02_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/bin_b02/bin_b02.usd"
    )
    dims = (0.1700, 0.1000, 0.0762)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1000.0),
    }


@register_asset
class BinB03VompRobolab(LibraryObject):
    """This container is a cylindrical metal bin with a smooth, silver finish. It features a flat lid and a simple, modern design, suitable for waste disposal or storage."""
    name = "bin_b03_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/bin_b03/bin_b03.usd"
    )
    dims = (0.2313, 0.1402, 0.1200)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=2700.0),
    }


@register_asset
class BinB04VompRobolab(LibraryObject):
    """This container is a metallic bin with a cylindrical shape, featuring a smooth, shiny surface and a flat lid. It is designed for durability and functionality, ideal for waste disposal or storage."""
    name = "bin_b04_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/bin_b04/bin_b04.usd"
    )
    dims = (0.3498, 0.1999, 0.1438)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class BinB06VompRobolab(LibraryObject):
    """This container is made of metal with a sleek, cylindrical shape and a smooth, silver finish. It features a hinged lid and a foot pedal for hands-free operation."""
    name = "bin_b06_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/bin_b06/bin_b06.usd"
    )
    scale = (0.49, 0.49, 0.49)
    dims = (0.7147, 0.4514, 0.3176)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=2700.0),
    }


@register_asset
class BinB07VompRobolab(LibraryObject):
    """This container is a cylindrical metal bin with a smooth, silver finish. It features a flat lid and a simple, functional design, suitable for waste disposal or storage purposes."""
    name = "bin_b07_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/bin_b07/bin_b07.usd"
    )
    scale = (0.49, 0.49, 0.49)
    dims = (0.7147, 0.4523, 0.3176)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=2700.0),
    }


@register_asset
class BlackandbrassbowlLargeVompRobolab(LibraryObject):
    """This large bowl features a sleek black exterior with a contrasting brass interior, crafted from opaque metal. Its smooth, rounded shape adds a touch of elegance to any setting."""
    name = "blackandbrassbowl_large_vomp_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/blackandbrassbowl_large/blackandbrassbowl_large.usd"
    )
    dims = (0.3017, 0.3017, 0.1522)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=2700.0),
    }


@register_asset
class BlackandbrassbowlSmallVompRobolab(LibraryObject):
    """This small bowl features a sleek, opaque metal design with a black exterior and a contrasting brass interior, offering a modern and elegant look."""
    name = "blackandbrassbowl_small_vomp_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/blackandbrassbowl_small/blackandbrassbowl_small.usd"
    )
    dims = (0.1803, 0.1803, 0.1013)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=2700.0),
    }


@register_asset
class BoxA01VompRobolab(LibraryObject):
    """This box is constructed from opaque plastic with alternating black and clear sections, featuring a sturdy rectangular shape. Its design includes multiple compartments, providing organized storage within its compact form."""
    name = "box_a01_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/box_a01/box_a01.usd"
    )
    scale = (0.583, 0.583, 0.583)
    dims = (0.5999, 0.3999, 0.3300)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=2700.0),
    }


@register_asset
class BoxA02VompRobolab(LibraryObject):
    """This box is made of opaque plastic with alternating black and regular bins. It has a smooth surface and a modular design, suitable for organizing or storing various items."""
    name = "box_a02_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/box_a02/box_a02.usd"
    )
    scale = (0.583, 0.583, 0.583)
    dims = (0.5999, 0.3999, 0.2900)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=7850.0),
    }


@register_asset
class BoxA03VompRobolab(LibraryObject):
    """The box is composed of opaque plastic with alternating black and standard bins, featuring a smooth surface and a sturdy, stackable design."""
    name = "box_a03_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/box_a03/box_a03.usd"
    )
    scale = (0.583, 0.583, 0.583)
    dims = (0.5999, 0.3999, 0.2299)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class BoxA04VompRobolab(LibraryObject):
    """The box is constructed from opaque plastic with alternating black and standard sections, featuring a sturdy, rectangular design. Its smooth surface and uniform color scheme provide a sleek, modern appearance."""
    name = "box_a04_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/box_a04/box_a04.usd"
    )
    scale = (0.875, 0.875, 0.875)
    dims = (0.4000, 0.3000, 0.4100)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=2700.0),
    }


@register_asset
class BoxA05VompRobolab(LibraryObject):
    """This box is constructed from opaque plastic with a mix of black and standard bins. It features a modular design, allowing for versatile storage and organization."""
    name = "box_a05_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/box_a05/box_a05.usd"
    )
    scale = (0.875, 0.875, 0.875)
    dims = (0.4000, 0.3000, 0.3300)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class BoxA06VompRobolab(LibraryObject):
    """This box is composed of opaque plastic with a mix of black and standard bins, featuring a sturdy, stackable design ideal for organizing and storing various items."""
    name = "box_a06_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/box_a06/box_a06.usd"
    )
    scale = (0.583, 0.583, 0.583)
    dims = (0.5999, 0.3999, 0.1850)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class BoxA07VompRobolab(LibraryObject):
    """The box is composed of opaque plastic with alternating black and standard sections. Its modular design features multiple bins, providing versatile storage options with a sleek, modern appearance."""
    name = "box_a07_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/box_a07/box_a07.usd"
    )
    scale = (0.583, 0.583, 0.583)
    dims = (0.5999, 0.4000, 0.1300)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class BoxA08VompRobolab(LibraryObject):
    """The box is composed of opaque plastic with alternating black and standard sections, featuring a sturdy rectangular shape. Its design suggests durability and practicality for storage purposes."""
    name = "box_a08_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/box_a08/box_a08.usd"
    )
    scale = (0.583, 0.583, 0.583)
    dims = (0.5999, 0.4123, 0.0850)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class BoxA09VompRobolab(LibraryObject):
    """The box is composed of opaque plastic with alternating black and standard bins, featuring a sturdy, stackable design ideal for organizing and storing items efficiently."""
    name = "box_a09_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/box_a09/box_a09.usd"
    )
    scale = (0.875, 0.875, 0.875)
    dims = (0.4000, 0.3000, 0.2448)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class BoxA10VompRobolab(LibraryObject):
    """The box is made of opaque plastic with a mix of black and standard bins, featuring a sturdy rectangular shape and a smooth finish."""
    name = "box_a10_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/box_a10/box_a10.usd"
    )
    scale = (0.875, 0.875, 0.875)
    dims = (0.4000, 0.3000, 0.1848)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class BoxA11VompRobolab(LibraryObject):
    """The box is composed of opaque plastic with alternating black and standard bins, featuring a sturdy, stackable design ideal for organizing and storing items efficiently."""
    name = "box_a11_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/box_a11/box_a11.usd"
    )
    scale = (0.873, 0.873, 0.873)
    dims = (0.4007, 0.2996, 0.1300)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class BulkstoragerackA01VompRobolab(LibraryObject):
    """A bulkstoragerack_a01 converted from SimReady."""
    name = "bulkstoragerack_a01_vomp_robolab"
    tags = ["object", "fixture", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/bulkstoragerack_a01/bulkstoragerack_a01.usd"
    )
    scale = (0.322, 0.322, 0.322)
    dims = (0.9063, 1.5520, 1.8362)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class CaseA01VompRobolab(LibraryObject):
    """This container is a glossy metal case with a sleek design, featuring two sturdy plastic handles for easy carrying. Its metallic surface is smooth and opaque, providing a durable and stylish appearance."""
    name = "case_a01_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/case_a01/case_a01.usd"
    )
    scale = (0.641, 0.641, 0.641)
    dims = (0.5464, 0.3555, 0.2294)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=2700.0),
    }


@register_asset
class CaseA02VompRobolab(LibraryObject):
    """The container is a glossy metal case with alternating sections of 'caseaglossya' and 'caseaglossyb' finishes, providing a sleek, durable appearance. Its opaque design suggests a sturdy, protective function."""
    name = "case_a02_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/case_a02/case_a02.usd"
    )
    scale = (0.588, 0.588, 0.588)
    dims = (0.5954, 0.4237, 0.2650)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class CaseA04VompRobolab(LibraryObject):
    """The container is a glossy metal case with alternating panels of 'caseaglossya' and 'caseaglossyb' finishes, creating a sleek, modern look. Its opaque design suggests durability and secure storage."""
    name = "case_a04_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/case_a04/case_a04.usd"
    )
    scale = (0.487, 0.487, 0.487)
    dims = (0.7181, 0.5268, 0.3455)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class CaseA05VompRobolab(LibraryObject):
    """This container is a glossy metal case with alternating sections of 'caseaglossya' and 'caseaglossyb' finishes, creating a sleek, modern appearance. It features an opaque design, emphasizing durability and style."""
    name = "case_a05_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/case_a05/case_a05.usd"
    )
    scale = (0.447, 0.447, 0.447)
    dims = (0.7827, 0.6045, 0.3806)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class CaseC01VompRobolab(LibraryObject):
    """The container features a glossy metal exterior with chrome accents and a textured section. It includes two glossy black plastic components, providing a sleek and modern appearance."""
    name = "case_c01_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/case_c01/case_c01.usd"
    )
    scale = (0.57, 0.57, 0.57)
    dims = (0.6142, 0.4324, 0.2661)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class CaseC02VompRobolab(LibraryObject):
    """The case features a glossy metal exterior with chrome accents and a rough-textured section. It includes two glossy black plastic components, providing a sleek and modern appearance."""
    name = "case_c02_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/case_c02/case_c02.usd"
    )
    scale = (0.427, 0.427, 0.427)
    dims = (0.8192, 0.4324, 0.3582)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=2700.0),
    }


@register_asset
class CaseC03VompRobolab(LibraryObject):
    """The case is a sturdy container made of glossy and rough metal, featuring chrome accents. It includes glossy black plastic components, providing a sleek and durable design suitable for secure storage."""
    name = "case_c03_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/case_c03/case_c03.usd"
    )
    scale = (0.381, 0.381, 0.381)
    dims = (0.9192, 0.5324, 0.4253)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=2700.0),
    }


@register_asset
class CaseC04VompRobolab(LibraryObject):
    """The container is a sturdy metal case with glossy and rough finishes, featuring chrome accents. It includes two glossy black plastic components, giving it a sleek, modern appearance."""
    name = "case_c04_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/case_c04/case_c04.usd"
    )
    scale = (0.427, 0.427, 0.427)
    dims = (0.8192, 0.6624, 0.4253)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class CaseC05VompRobolab(LibraryObject):
    """This container features a sleek design with glossy and rough metal surfaces, accented by glossy black plastic elements and chrome details, creating a modern and durable appearance."""
    name = "case_c05_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/case_c05/case_c05.usd"
    )
    scale = (0.427, 0.427, 0.427)
    dims = (0.8192, 0.6624, 0.5968)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class CaseD01VompRobolab(LibraryObject):
    """This container is a glossy metal and plastic case with a sleek design, featuring alternating sections of shiny metal and black plastic. Its robust construction suggests durability and secure storage."""
    name = "case_d01_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/case_d01/case_d01.usd"
    )
    scale = (0.856, 0.856, 0.856)
    dims = (0.4090, 0.3219, 0.3447)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class CaseD02VompRobolab(LibraryObject):
    """The container is a glossy metal case with alternating shiny silver and black plastic sections, featuring a sleek, modern design."""
    name = "case_d02_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/case_d02/case_d02.usd"
    )
    scale = (0.584, 0.584, 0.584)
    dims = (0.5995, 0.4012, 0.2648)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class CaseD03VompRobolab(LibraryObject):
    """The container features a glossy metal exterior with alternating panels of glossy black plastic. Its sleek design combines durable metal and plastic materials, providing a modern and sturdy appearance."""
    name = "case_d03_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/case_d03/case_d03.usd"
    )
    scale = (0.447, 0.447, 0.447)
    dims = (0.7824, 0.3938, 0.3797)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class CaseD04VompRobolab(LibraryObject):
    """The case is a glossy metal container with alternating glossy black plastic sections, featuring a sleek, modern design. Its metallic parts are polished, providing a reflective finish that contrasts with the matte black plastic."""
    name = "case_d04_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/case_d04/case_d04.usd"
    )
    scale = (0.387, 0.387, 0.387)
    dims = (0.9048, 0.5018, 0.3827)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class CaseE01VompRobolab(LibraryObject):
    """The container features a glossy metal exterior with chrome accents and black plastic components. Its sleek, opaque design combines durability with a modern aesthetic, ideal for secure storage or transport."""
    name = "case_e01_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/case_e01/case_e01.usd"
    )
    scale = (0.731, 0.731, 0.731)
    dims = (0.4788, 0.2769, 0.2099)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=2700.0),
    }


@register_asset
class CaseE02VompRobolab(LibraryObject):
    """The container features a glossy metal body with chrome accents and glossy black plastic components. Its sleek design combines durability with a modern aesthetic, making it both functional and visually appealing."""
    name = "case_e02_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/case_e02/case_e02.usd"
    )
    scale = (0.495, 0.495, 0.495)
    dims = (0.7070, 0.3218, 0.3199)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=2700.0),
    }


@register_asset
class CaseE03VompRobolab(LibraryObject):
    """The container features a glossy metal body with chrome accents and black plastic components. Its sleek design combines opaque metal and plastic parts, creating a modern, durable appearance."""
    name = "case_e03_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/case_e03/case_e03.usd"
    )
    scale = (0.462, 0.462, 0.462)
    dims = (0.7568, 0.3912, 0.3599)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=2700.0),
    }


@register_asset
class ClosedheadpailA01VompRobolab(LibraryObject):
    """This pail is made of opaque blue plastic with a glossy finish. It features a closed head design, providing a secure seal for contents, and includes a sturdy handle for easy carrying."""
    name = "closedheadpail_a01_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/closedheadpail_a01/closedheadpail_a01.usd"
    )
    dims = (0.2628, 0.2634, 0.3470)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=950.0),
    }


@register_asset
class ContainerA01VompRobolab(LibraryObject):
    """This container is made of metal with a cylindrical shape and a smooth, silver finish. It features a secure, fitted lid and is designed for storing various items."""
    name = "container_a01_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_a01/container_a01.usd"
    )
    scale = (0.769, 0.769, 0.769)
    dims = (0.4550, 0.3250, 0.2348)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=2700.0),
    }


@register_asset
class ContainerA02VompRobolab(LibraryObject):
    """The container features a sleek chrome metal body with a black plastic lid, offering a modern and durable design."""
    name = "container_a02_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_a02/container_a02.usd"
    )
    scale = (0.642, 0.642, 0.642)
    dims = (0.5450, 0.3850, 0.2600)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class ContainerA03VompRobolab(LibraryObject):
    """The container features a sleek, chrome metal body with a sturdy, opaque black plastic lid, combining durability with a modern aesthetic."""
    name = "container_a03_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_a03/container_a03.usd"
    )
    scale = (0.56, 0.56, 0.56)
    dims = (0.6250, 0.4450, 0.3300)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class ContainerA04VompRobolab(LibraryObject):
    """The container features a chrome metal body with a sleek, opaque finish, complemented by a black plastic lid. Its modern design and durable materials make it ideal for storing various items securely."""
    name = "container_a04_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_a04/container_a04.usd"
    )
    scale = (0.496, 0.496, 0.496)
    dims = (0.7050, 0.5050, 0.3750)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class ContainerB02VompRobolab(LibraryObject):
    """The container is made of opaque plastic, featuring a smooth, cylindrical shape with a secure, snap-on lid. It is designed for storage, with a minimalist design and no visible labels or markings."""
    name = "container_b02_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_b02/container_b02.usd"
    )
    scale = (0.875, 0.875, 0.875)
    dims = (0.4000, 0.3000, 0.0550)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class ContainerB03VompRobolab(LibraryObject):
    """This container is made of metal with a cylindrical shape and a smooth, silver finish. It features a secure, removable lid and is designed for storing various items efficiently."""
    name = "container_b03_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_b03/container_b03.usd"
    )
    scale = (0.875, 0.875, 0.875)
    dims = (0.4001, 0.3006, 0.0750)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class ContainerB04VompRobolab(LibraryObject):
    """The container features a sturdy metal construction with a smooth, cylindrical shape and a matte silver finish. It includes a secure, flat lid and subtle, horizontal grooves for added grip and style."""
    name = "container_b04_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_b04/container_b04.usd"
    )
    scale = (0.869, 0.869, 0.869)
    dims = (0.4027, 0.3006, 0.1200)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class ContainerB05VompRobolab(LibraryObject):
    """The container is metallic with a sleek, cylindrical shape and a smooth, reflective surface. It features a secure, fitted lid and a minimalist design, ideal for storing various items."""
    name = "container_b05_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_b05/container_b05.usd"
    )
    scale = (0.869, 0.869, 0.869)
    dims = (0.4027, 0.3006, 0.1700)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class ContainerB06VompRobolab(LibraryObject):
    """This container features a sleek metal body with a cylindrical shape and a smooth, polished finish. It has a secure lid and a minimalist design, ideal for storing various items."""
    name = "container_b06_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_b06/container_b06.usd"
    )
    scale = (0.875, 0.875, 0.875)
    dims = (0.4001, 0.3006, 0.1700)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=2700.0),
    }


@register_asset
class ContainerB07VompRobolab(LibraryObject):
    """This container is made of durable metal with a sleek, cylindrical shape and a smooth, silver finish. It features a secure, tight-fitting lid and is designed for efficient storage and transport."""
    name = "container_b07_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_b07/container_b07.usd"
    )
    scale = (0.874, 0.874, 0.874)
    dims = (0.4006, 0.3006, 0.2200)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=2700.0),
    }


@register_asset
class ContainerB08VompRobolab(LibraryObject):
    """The container is metallic with a cylindrical shape, featuring a smooth, silver surface and a secure, fitted lid. It is designed for durability and storage efficiency."""
    name = "container_b08_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_b08/container_b08.usd"
    )
    scale = (0.869, 0.869, 0.869)
    dims = (0.4027, 0.3006, 0.2200)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class ContainerB09VompRobolab(LibraryObject):
    """The container is metallic with a cylindrical shape and a smooth, shiny surface. It features a secure, flat lid and is designed for durability and storage efficiency."""
    name = "container_b09_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_b09/container_b09.usd"
    )
    scale = (0.876, 0.876, 0.876)
    dims = (0.3994, 0.2998, 0.2200)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class ContainerB10VompRobolab(LibraryObject):
    """The container is metallic with a smooth, cylindrical body and a secure, flat lid. It features a sleek silver finish, ideal for storing various items while maintaining a modern aesthetic."""
    name = "container_b10_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_b10/container_b10.usd"
    )
    scale = (0.875, 0.875, 0.875)
    dims = (0.4001, 0.3006, 0.2350)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class ContainerB11VompRobolab(LibraryObject):
    """The container is metallic with a sleek, cylindrical shape and a smooth surface. It features a secure, fitted lid and a minimalist design, ideal for storing various items."""
    name = "container_b11_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_b11/container_b11.usd"
    )
    scale = (0.876, 0.876, 0.876)
    dims = (0.3994, 0.2998, 0.2700)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class ContainerB12VompRobolab(LibraryObject):
    """The container is metallic with a sleek, cylindrical shape and a smooth, reflective surface. It features a secure, fitted lid and a minimalist design, ideal for storing various items."""
    name = "container_b12_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_b12/container_b12.usd"
    )
    scale = (0.876, 0.876, 0.876)
    dims = (0.3994, 0.2998, 0.2700)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class ContainerB13VompRobolab(LibraryObject):
    """The container is metallic with a cylindrical shape, featuring a smooth, silver surface. It has a secure, removable lid and is designed for durability and storage efficiency."""
    name = "container_b13_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_b13/container_b13.usd"
    )
    scale = (0.866, 0.866, 0.866)
    dims = (0.4044, 0.3006, 0.2700)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class ContainerB14VompRobolab(LibraryObject):
    """The container features an opaque plastic body with a smooth finish, paired with a glossy metal lid. Its minimalist design combines durability and style, ideal for storing various items securely."""
    name = "container_b14_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_b14/container_b14.usd"
    )
    scale = (0.875, 0.875, 0.875)
    dims = (0.4001, 0.3006, 0.2700)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class ContainerB15VompRobolab(LibraryObject):
    """The container is made of metal with a sleek, cylindrical shape and a smooth, silver finish. It features a secure, airtight lid and is designed for durability and storage efficiency."""
    name = "container_b15_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_b15/container_b15.usd"
    )
    scale = (0.875, 0.875, 0.875)
    dims = (0.4001, 0.3006, 0.2800)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class ContainerB16VompRobolab(LibraryObject):
    """The container is metallic with a cylindrical shape, featuring a smooth silver surface. It has a secure lid and a simple, industrial design, ideal for storing various items."""
    name = "container_b16_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_b16/container_b16.usd"
    )
    scale = (0.875, 0.875, 0.875)
    dims = (0.4001, 0.3006, 0.3200)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class ContainerB17VompRobolab(LibraryObject):
    """The container is metallic with a sleek, cylindrical shape and a brushed silver finish. It features a secure, airtight lid and is designed for durability and storage efficiency."""
    name = "container_b17_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_b17/container_b17.usd"
    )
    scale = (0.869, 0.869, 0.869)
    dims = (0.4027, 0.3006, 0.3200)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=2700.0),
    }


@register_asset
class ContainerB18VompRobolab(LibraryObject):
    """The container is metallic with a sleek, cylindrical shape and a smooth, reflective surface. It features a secure, fitted lid and a minimalist design, ideal for storing various items efficiently."""
    name = "container_b18_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_b18/container_b18.usd"
    )
    scale = (0.875, 0.875, 0.875)
    dims = (0.4001, 0.3006, 0.4000)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=950.0),
    }


@register_asset
class ContainerC01VompRobolab(LibraryObject):
    """The container is metallic with a cylindrical shape, featuring a smooth, shiny surface. It has a secure lid and a minimalist design, ideal for storing various items."""
    name = "container_c01_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_c01/container_c01.usd"
    )
    scale = (0.583, 0.583, 0.583)
    dims = (0.6000, 0.4004, 0.2288)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=2700.0),
    }


@register_asset
class ContainerC02VompRobolab(LibraryObject):
    """The container is metallic with a cylindrical shape and a smooth, reflective surface. It features a secure lid and a seamless design, emphasizing durability and industrial style."""
    name = "container_c02_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_c02/container_c02.usd"
    )
    scale = (0.583, 0.583, 0.583)
    dims = (0.6000, 0.4004, 0.2882)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class ContainerC03VompRobolab(LibraryObject):
    """The container is metallic with a cylindrical shape, featuring a smooth silver surface. It has a secure lid and a simple, industrial design, suitable for storing various items."""
    name = "container_c03_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_c03/container_c03.usd"
    )
    scale = (0.583, 0.583, 0.583)
    dims = (0.6000, 0.4004, 0.3480)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=2700.0),
    }


@register_asset
class ContainerC04VompRobolab(LibraryObject):
    """The container is metallic with a cylindrical shape, featuring a smooth silver surface. It has a secure, fitted lid and is designed for durability and storage efficiency."""
    name = "container_c04_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_c04/container_c04.usd"
    )
    scale = (0.583, 0.583, 0.583)
    dims = (0.6000, 0.4004, 0.4281)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class ContainerC05VompRobolab(LibraryObject):
    """The container is metallic with a cylindrical shape, featuring a smooth, reflective surface. It has a secure lid and a minimalist design, emphasizing durability and functionality."""
    name = "container_c05_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_c05/container_c05.usd"
    )
    scale = (0.583, 0.583, 0.583)
    dims = (0.6000, 0.4025, 0.2282)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class ContainerC06VompRobolab(LibraryObject):
    """The container is metallic with a cylindrical shape, featuring a smooth, shiny surface. It has a secure lid and a minimalist design, suitable for storing various items."""
    name = "container_c06_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_c06/container_c06.usd"
    )
    scale = (0.583, 0.583, 0.583)
    dims = (0.6000, 0.4025, 0.2876)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=950.0),
    }


@register_asset
class ContainerC07VompRobolab(LibraryObject):
    """This container is metallic with a cylindrical shape and a smooth, silver finish. It features a secure, matching lid and a seamless design, ideal for storing various items."""
    name = "container_c07_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_c07/container_c07.usd"
    )
    scale = (0.583, 0.583, 0.583)
    dims = (0.6000, 0.4025, 0.3475)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class ContainerC08VompRobolab(LibraryObject):
    """The container is metallic with a cylindrical shape, featuring a smooth, shiny surface. It has a secure, fitted lid and a minimalist design, emphasizing functionality and durability."""
    name = "container_c08_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_c08/container_c08.usd"
    )
    scale = (0.583, 0.583, 0.583)
    dims = (0.6000, 0.4025, 0.4275)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class ContainerC09VompRobolab(LibraryObject):
    """This container is metallic with a cylindrical shape and a smooth, reflective surface. It features a secure, fitted lid and is designed for durability and efficient storage."""
    name = "container_c09_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_c09/container_c09.usd"
    )
    scale = (0.862, 0.862, 0.862)
    dims = (0.4061, 0.3000, 0.1285)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=2700.0),
    }


@register_asset
class ContainerC10VompRobolab(LibraryObject):
    """The container is metallic with a cylindrical shape, featuring a smooth silver surface. It has a secure lid and is designed for durability, ideal for storing various items."""
    name = "container_c10_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_c10/container_c10.usd"
    )
    scale = (0.862, 0.862, 0.862)
    dims = (0.4062, 0.3000, 0.1779)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=2700.0),
    }


@register_asset
class ContainerC11VompRobolab(LibraryObject):
    """The container is metallic with a cylindrical shape and a smooth, shiny surface. It features a secure, matching lid, providing a sleek and functional design for storage purposes."""
    name = "container_c11_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_c11/container_c11.usd"
    )
    scale = (0.862, 0.862, 0.862)
    dims = (0.4062, 0.3000, 0.2280)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class ContainerC12VompRobolab(LibraryObject):
    """The container is metallic with a smooth, cylindrical shape and a silver finish. It features a secure, matching lid and is designed for durability and practical storage."""
    name = "container_c12_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_c12/container_c12.usd"
    )
    scale = (0.862, 0.862, 0.862)
    dims = (0.4062, 0.3000, 0.3280)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=2700.0),
    }


@register_asset
class ContainerC13VompRobolab(LibraryObject):
    """The container is metallic with a sleek, cylindrical shape and a smooth, reflective surface. It features a secure lid and a minimalist design, emphasizing functionality and durability."""
    name = "container_c13_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_c13/container_c13.usd"
    )
    scale = (0.862, 0.862, 0.862)
    dims = (0.4061, 0.3029, 0.1280)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class ContainerC14VompRobolab(LibraryObject):
    """This container is made of metal with a sleek, cylindrical shape and a smooth, silver finish. It features a secure lid and a minimalist design, suitable for storing various items."""
    name = "container_c14_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_c14/container_c14.usd"
    )
    scale = (0.862, 0.862, 0.862)
    dims = (0.4062, 0.3029, 0.1773)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class ContainerC15VompRobolab(LibraryObject):
    """The container is metallic with a cylindrical shape and a smooth, silver surface. It features a secure lid and a robust design, suitable for storing various items."""
    name = "container_c15_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_c15/container_c15.usd"
    )
    scale = (0.862, 0.862, 0.862)
    dims = (0.4062, 0.3029, 0.2275)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class ContainerC16VompRobolab(LibraryObject):
    """The container is metallic with a cylindrical shape and a smooth, shiny surface. It features a secure, tight-fitting lid and is designed for durability and storage efficiency."""
    name = "container_c16_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_c16/container_c16.usd"
    )
    scale = (0.862, 0.862, 0.862)
    dims = (0.4062, 0.3029, 0.3275)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=2700.0),
    }


@register_asset
class ContainerC17VompRobolab(LibraryObject):
    """The container is metallic with a cylindrical shape, featuring a smooth, shiny surface. It has a secure, flat lid and a minimalist design, emphasizing functionality and durability."""
    name = "container_c17_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_c17/container_c17.usd"
    )
    scale = (0.577, 0.577, 0.577)
    dims = (0.6062, 0.4013, 0.0832)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class ContainerC18VompRobolab(LibraryObject):
    """This container is metallic with a cylindrical shape and a smooth, reflective surface. It features a secure lid and a minimalist design, ideal for storing various items."""
    name = "container_c18_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_c18/container_c18.usd"
    )
    scale = (0.583, 0.583, 0.583)
    dims = (0.6000, 0.4006, 0.1274)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class ContainerC20VompRobolab(LibraryObject):
    """The container is metallic with a cylindrical shape, featuring a smooth, silver surface. It has a secure, fitted lid and is designed for durability and storage efficiency."""
    name = "container_c20_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_c20/container_c20.usd"
    )
    scale = (0.577, 0.577, 0.577)
    dims = (0.6062, 0.4008, 0.1782)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class ContainerC21VompRobolab(LibraryObject):
    """The container is metallic with a sleek, cylindrical shape and a smooth, reflective surface. It features a secure lid and a minimalist design, emphasizing functionality and modern aesthetics."""
    name = "container_c21_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_c21/container_c21.usd"
    )
    scale = (0.577, 0.577, 0.577)
    dims = (0.6062, 0.4031, 0.0827)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=2700.0),
    }


@register_asset
class ContainerC22VompRobolab(LibraryObject):
    """The container is metallic with a smooth, cylindrical body and a secure, fitted lid. It features a sleek silver finish, providing a modern and industrial look."""
    name = "container_c22_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_c22/container_c22.usd"
    )
    scale = (0.583, 0.583, 0.583)
    dims = (0.6000, 0.4025, 0.1269)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class ContainerC23VompRobolab(LibraryObject):
    """The container is metallic with a cylindrical shape and a smooth, shiny surface. It features a secure lid and a minimalist design, ideal for storing various items."""
    name = "container_c23_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_c23/container_c23.usd"
    )
    scale = (0.577, 0.577, 0.577)
    dims = (0.6062, 0.4025, 0.1577)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class ContainerC24VompRobolab(LibraryObject):
    """The container is metallic with a sleek, cylindrical shape and a smooth, silver finish. It features a secure lid and a minimalist design, ideal for storing various items."""
    name = "container_c24_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_c24/container_c24.usd"
    )
    scale = (0.577, 0.577, 0.577)
    dims = (0.6062, 0.4025, 0.1777)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class ContainerD01VompRobolab(LibraryObject):
    """The container is made of opaque black plastic, featuring a smooth, cylindrical body with a secure, snap-on lid. Its minimalist design is both functional and durable for storing various items."""
    name = "container_d01_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_d01/container_d01.usd"
    )
    scale = (0.166, 0.166, 0.166)
    dims = (2.1071, 1.4806, 0.9300)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class ContainerD02VompRobolab(LibraryObject):
    """The container is made of opaque black plastic with a smooth, cylindrical shape and a secure-fitting lid, designed for durable storage and easy handling."""
    name = "container_d02_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_d02/container_d02.usd"
    )
    scale = (0.193, 0.193, 0.193)
    dims = (1.8142, 1.3874, 0.8098)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class ContainerD03VompRobolab(LibraryObject):
    """The container is made of opaque black plastic, featuring a smooth, cylindrical shape with a secure-fitting lid. Its minimalist design is functional, ideal for storing various items discreetly."""
    name = "container_d03_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_d03/container_d03.usd"
    )
    scale = (0.217, 0.217, 0.217)
    dims = (1.6147, 1.1889, 0.8089)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class ContainerD04VompRobolab(LibraryObject):
    """The container is made of opaque black plastic with a smooth, cylindrical shape. It features a secure lid and a minimalist design, ideal for storing various items discreetly."""
    name = "container_d04_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_d04/container_d04.usd"
    )
    scale = (0.265, 0.265, 0.265)
    dims = (1.3184, 0.9697, 0.8115)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class ContainerD05VompRobolab(LibraryObject):
    """The container is made of opaque black plastic, featuring a smooth, cylindrical body with a secure, snap-on lid. Its minimalist design is functional for storage, emphasizing durability and simplicity."""
    name = "container_d05_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_d05/container_d05.usd"
    )
    scale = (0.266, 0.266, 0.266)
    dims = (1.3177, 0.9689, 0.6287)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class ContainerD06VompRobolab(LibraryObject):
    """The container is made of opaque black plastic, featuring a smooth, cylindrical body with a secure-fitting lid. Its minimalist design is functional, ideal for storing various items discreetly."""
    name = "container_d06_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_d06/container_d06.usd"
    )
    scale = (0.294, 0.294, 0.294)
    dims = (1.1888, 0.7906, 0.6018)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class ContainerD07VompRobolab(LibraryObject):
    """The container is made of opaque black plastic, featuring a smooth surface and a sturdy, rectangular shape. It is designed for durability and storage, with a secure-fitting lid to keep contents safe."""
    name = "container_d07_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_d07/container_d07.usd"
    )
    scale = (0.297, 0.297, 0.297)
    dims = (1.1777, 0.7005, 0.5296)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class ContainerD08VompRobolab(LibraryObject):
    """This container is made of opaque black plastic, featuring a smooth, rectangular shape with a secure-fitting lid. Its minimalist design is functional, ideal for storing various items discreetly."""
    name = "container_d08_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_d08/container_d08.usd"
    )
    scale = (0.398, 0.398, 0.398)
    dims = (0.8786, 0.5699, 0.5994)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class ContainerD09VompRobolab(LibraryObject):
    """The container is made of opaque black plastic, featuring a smooth, cylindrical shape with a secure-fitting lid. Its minimalist design is functional, ideal for storing various items discreetly."""
    name = "container_d09_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_d09/container_d09.usd"
    )
    scale = (0.398, 0.398, 0.398)
    dims = (0.8788, 0.5805, 0.2913)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class ContainerE01VompRobolab(LibraryObject):
    """The container is metallic with a sleek, cylindrical shape and a smooth, reflective surface. It features a secure, matching lid, providing a modern and functional design."""
    name = "container_e01_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_e01/container_e01.usd"
    )
    scale = (0.29, 0.29, 0.29)
    dims = (1.2087, 0.8320, 0.8423)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=950.0),
    }


@register_asset
class ContainerE02VompRobolab(LibraryObject):
    """This container is made of metal with a sleek, cylindrical shape and a smooth, reflective surface. It features a secure, tight-fitting lid and a minimalist design, ideal for storing various items."""
    name = "container_e02_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_e02/container_e02.usd"
    )
    scale = (0.29, 0.29, 0.29)
    dims = (1.2087, 0.8320, 0.7499)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=2700.0),
    }


@register_asset
class ContainerE03VompRobolab(LibraryObject):
    """The container is metallic with a sleek, cylindrical shape and a smooth, reflective surface. It features a secure, tight-fitting lid, making it ideal for storing various items safely."""
    name = "container_e03_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_e03/container_e03.usd"
    )
    scale = (0.29, 0.29, 0.29)
    dims = (1.2087, 0.8320, 0.6439)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=950.0),
    }


@register_asset
class ContainerE04VompRobolab(LibraryObject):
    """This container is metallic with a cylindrical shape and a smooth, shiny surface. It features a secure, fitted lid and a minimalist design, suitable for storing various items."""
    name = "container_e04_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_e04/container_e04.usd"
    )
    scale = (0.29, 0.29, 0.29)
    dims = (1.2087, 0.8320, 0.5445)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=2700.0),
    }


@register_asset
class ContainerE05VompRobolab(LibraryObject):
    """This container is made of metal with a sleek, cylindrical shape and a smooth, reflective surface. It features a secure lid and a minimalist design, emphasizing functionality and modern aesthetics."""
    name = "container_e05_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_e05/container_e05.usd"
    )
    scale = (0.29, 0.29, 0.29)
    dims = (1.2087, 0.8320, 0.4462)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=950.0),
    }


@register_asset
class ContainerE06VompRobolab(LibraryObject):
    """The container is metallic with a cylindrical shape, featuring a smooth, silver surface and a secure, fitted lid. It is designed for durability and efficient storage."""
    name = "container_e06_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_e06/container_e06.usd"
    )
    scale = (0.29, 0.29, 0.29)
    dims = (1.2087, 0.8320, 0.3437)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class ContainerF03VompRobolab(LibraryObject):
    """The container features an opaque plastic bin with a sleek black finish and chrome metal accents, providing a modern and durable storage solution."""
    name = "container_f03_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_f03/container_f03.usd"
    )
    scale = (0.291, 0.291, 0.291)
    dims = (1.2014, 0.8097, 0.8864)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=950.0),
    }


@register_asset
class ContainerF04VompRobolab(LibraryObject):
    """The container features an opaque plastic bin with a sleek black finish, complemented by chrome metal accents. Its sturdy construction and modern design make it ideal for organizing and storing various items."""
    name = "container_f04_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_f04/container_f04.usd"
    )
    scale = (0.291, 0.291, 0.291)
    dims = (1.2014, 0.8097, 0.8864)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class ContainerF05VompRobolab(LibraryObject):
    """The container features an opaque plastic bin with a sleek black finish, complemented by chrome metal accents. Its sturdy construction and modern design make it ideal for storage and organization."""
    name = "container_f05_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_f05/container_f05.usd"
    )
    scale = (0.291, 0.291, 0.291)
    dims = (1.2007, 0.8097, 0.8863)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class ContainerF08VompRobolab(LibraryObject):
    """This container features an opaque plastic bin with a sleek, black finish. It has a sturdy, rectangular shape and a secure lid, making it ideal for storage and organization."""
    name = "container_f08_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_f08/container_f08.usd"
    )
    scale = (0.291, 0.291, 0.291)
    dims = (1.2014, 0.8014, 0.8864)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=950.0),
    }


@register_asset
class ContainerF09VompRobolab(LibraryObject):
    """The container features an opaque plastic bin with a black lid, providing a sturdy and discreet storage solution. Its simple design and neutral color make it versatile for various organizational needs."""
    name = "container_f09_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_f09/container_f09.usd"
    )
    scale = (0.291, 0.291, 0.291)
    dims = (1.2014, 0.8014, 0.8864)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class ContainerF10VompRobolab(LibraryObject):
    """The container features an opaque plastic bin with a sleek black finish, designed for durability and efficient storage. Its minimalist design and sturdy construction make it ideal for organizing various items."""
    name = "container_f10_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_f10/container_f10.usd"
    )
    scale = (0.291, 0.291, 0.291)
    dims = (1.2007, 0.8014, 0.8863)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class ContainerF13VompRobolab(LibraryObject):
    """The container features an opaque plastic bin with a black lid. Its sturdy construction is ideal for storage, and the sleek black top provides a secure closure."""
    name = "container_f13_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_f13/container_f13.usd"
    )
    scale = (0.291, 0.291, 0.291)
    dims = (1.2014, 0.8014, 0.8864)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class ContainerF14VompRobolab(LibraryObject):
    """The container features an opaque plastic bin with a black lid, designed for secure storage. Its sturdy construction and simple design make it suitable for organizing various items."""
    name = "container_f14_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_f14/container_f14.usd"
    )
    scale = (0.291, 0.291, 0.291)
    dims = (1.2007, 0.8014, 0.8863)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class ContainerF15VompRobolab(LibraryObject):
    """The container features an opaque plastic bin with a black finish, providing a sturdy and discreet storage solution. Its simple design is functional, suitable for organizing various items while maintaining a sleek appearance."""
    name = "container_f15_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_f15/container_f15.usd"
    )
    scale = (0.301, 0.301, 0.301)
    dims = (1.1635, 0.7995, 0.8000)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class ContainerF16VompRobolab(LibraryObject):
    """This container features an opaque plastic bin with a sturdy, rectangular shape and a black lid. The materials provide durability and a sleek, modern appearance, ideal for storage and organization."""
    name = "container_f16_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_f16/container_f16.usd"
    )
    scale = (0.301, 0.301, 0.301)
    dims = (1.1635, 0.7995, 0.8000)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class ContainerF17VompRobolab(LibraryObject):
    """The container is a sturdy, opaque plastic bin with a sleek black finish. It features a secure-fitting lid and reinforced edges, ideal for storage and organization."""
    name = "container_f17_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_f17/container_f17.usd"
    )
    scale = (0.43, 0.43, 0.43)
    dims = (0.7995, 1.1634, 0.8000)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=950.0),
    }


@register_asset
class ContainerF18VompRobolab(LibraryObject):
    """This container features an opaque plastic bin with a black finish, providing a sturdy and durable storage solution. Its sleek design is ideal for organizing various items while maintaining a clean, modern appearance."""
    name = "container_f18_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_f18/container_f18.usd"
    )
    scale = (0.43, 0.43, 0.43)
    dims = (0.7995, 1.1634, 0.8000)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class ContainerF20VompRobolab(LibraryObject):
    """The container features an opaque plastic bin with a sleek black finish, designed for durability and discreet storage."""
    name = "container_f20_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_f20/container_f20.usd"
    )
    scale = (0.301, 0.301, 0.301)
    dims = (1.1634, 0.7995, 0.8000)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class ContainerF21VompRobolab(LibraryObject):
    """The container features an opaque plastic bin with a sleek, black finish. It is designed for durability and storage, with a simple, modern aesthetic."""
    name = "container_f21_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_f21/container_f21.usd"
    )
    scale = (0.301, 0.301, 0.301)
    dims = (1.1634, 0.7995, 0.8000)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=950.0),
    }


@register_asset
class ContainerF23VompRobolab(LibraryObject):
    """The container features an opaque black plastic bin with a sturdy, rectangular shape and a secure-fitting lid. Its durable construction is ideal for storage and organization, providing a sleek, minimalist appearance."""
    name = "container_f23_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_f23/container_f23.usd"
    )
    scale = (0.302, 0.302, 0.302)
    dims = (1.1584, 0.7958, 0.6700)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class ContainerF24VompRobolab(LibraryObject):
    """The container features an opaque plastic bin with a matte finish and a black lid. Its sturdy construction is ideal for storage, and the lid securely snaps on to keep contents protected."""
    name = "container_f24_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_f24/container_f24.usd"
    )
    scale = (0.302, 0.302, 0.302)
    dims = (1.1584, 0.7958, 0.6700)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=950.0),
    }


@register_asset
class ContainerF26VompRobolab(LibraryObject):
    """This container features an opaque plastic bin with a black lid. The sturdy construction is ideal for storage, offering durability and a sleek, modern appearance."""
    name = "container_f26_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/container_f26/container_f26.usd"
    )
    scale = (0.302, 0.302, 0.302)
    dims = (1.1584, 0.7958, 0.5220)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=950.0),
    }


@register_asset
class CrabbypenholderVompRobolab(LibraryObject):
    """The crab-shaped pen holder is made of opaque plastic, featuring a vibrant red color. It has detailed claws and legs, with multiple slots on its back to hold pens and pencils securely."""
    name = "crabbypenholder_vomp_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/crabbypenholder/crabbypenholder.usd"
    )
    dims = (0.0505, 0.0303, 0.0347)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class CubeboxA02VompRobolab(LibraryObject):
    """The cardboard box is opaque, featuring a natural brown color with a smooth surface. It includes a decal, adding a touch of detail to its otherwise plain exterior."""
    name = "cubebox_a02_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/cubebox_a02/cubebox_a02.usd"
    )
    dims = (0.1572, 0.1562, 0.1587)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=700.0),
    }


@register_asset
class CuttingBoardAVompRobolab(LibraryObject):
    """This cutting board is made of opaque wood with a smooth, rectangular surface. It features natural wood grain patterns, providing a sturdy and durable platform for food preparation."""
    name = "cutting_board_a_vomp_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/cutting_board_a/cutting_board_a.usd"
    )
    scale = (0.783, 0.783, 0.783)
    dims = (0.4469, 0.6145, 0.0608)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=600.0),
    }


@register_asset
class ForkBigVompRobolab(LibraryObject):
    """This large fork is made of opaque plastic with a smooth, sturdy handle and four evenly spaced tines. It features a simple, functional design suitable for serving or dining."""
    name = "fork_big_vomp_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/fork_big/fork_big.usd"
    )
    dims = (0.1841, 0.0249, 0.0133)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=7850.0),
    }


@register_asset
class ForkSmallVompRobolab(LibraryObject):
    """This small fork is made of opaque plastic with a smooth, lightweight design. It features four short tines and a simple handle, ideal for casual dining or picnics."""
    name = "fork_small_vomp_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/fork_small/fork_small.usd"
    )
    dims = (0.1626, 0.0287, 0.0132)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class GardenplanterLargeVompRobolab(LibraryObject):
    """This large garden planter is made of opaque concrete, featuring a rectangular shape with a smooth, gray surface. Its sturdy construction is ideal for outdoor use, providing ample space for various plants and flowers."""
    name = "gardenplanter_large_vomp_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/gardenplanter_large/gardenplanter_large.usd"
    )
    scale = (0.787, 0.787, 0.787)
    dims = (0.4446, 0.4446, 0.4829)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.009999999776482582
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=2400.0),
    }


@register_asset
class GardenplanterMediumVompRobolab(LibraryObject):
    """The garden planter is a medium-sized, rectangular flower box made of opaque concrete with a smooth, gray finish. It features clean lines and a sturdy build, ideal for outdoor plant arrangements."""
    name = "gardenplanter_medium_vomp_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/gardenplanter_medium/gardenplanter_medium.usd"
    )
    scale = (0.985, 0.985, 0.985)
    dims = (0.3552, 0.3552, 0.3810)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.009999999776482582
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=2400.0),
    }


@register_asset
class GardenplanterSmallVompRobolab(LibraryObject):
    """This small garden planter is made of opaque concrete, featuring a rectangular shape with a smooth, minimalist design. Its neutral gray color complements various outdoor settings, ideal for housing flowers or small plants."""
    name = "gardenplanter_small_vomp_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/gardenplanter_small/gardenplanter_small.usd"
    )
    dims = (0.3048, 0.3048, 0.3048)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.009999999776482582
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=2500.0),
    }


@register_asset
class HeavydutysteelshelvingA01VompRobolab(LibraryObject):
    """A heavydutysteelshelving_a01 converted from SimReady."""
    name = "heavydutysteelshelving_a01_vomp_robolab"
    tags = ["object", "fixture", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/heavydutysteelshelving_a01/heavydutysteelshelving_a01.usd"
    )
    scale = (0.544, 0.544, 0.544)
    dims = (0.4624, 0.9196, 1.8288)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=7850.0),
    }


@register_asset
class LargeStorageRackVompRobolab(LibraryObject):
    name = "large_storage_rack_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/large_storage_rack/large_storage_rack.usd"
    )
    scale = (0.322, 0.322, 0.322)
    dims = (0.9063, 1.5520, 1.8362)


@register_asset
class LiquidscrewtoppailA01VompRobolab(LibraryObject):
    """This pail features a glossy, opaque plastic body with a red and black screw top. Its design is sturdy and functional, suitable for securely storing and transporting liquids."""
    name = "liquidscrewtoppail_a01_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/liquidscrewtoppail_a01/liquidscrewtoppail_a01.usd"
    )
    scale = (0.923, 0.923, 0.923)
    dims = (0.3792, 0.3866, 0.4299)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=950.0),
    }


@register_asset
class LiquidscrewtoppailA02VompRobolab(LibraryObject):
    """This pail features a glossy, opaque plastic body with a red hue and a black screw top lid. Its sturdy design is ideal for securely storing and transporting liquids."""
    name = "liquidscrewtoppail_a02_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/liquidscrewtoppail_a02/liquidscrewtoppail_a02.usd"
    )
    scale = (0.923, 0.923, 0.923)
    dims = (0.3792, 0.3866, 0.4299)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class MetalfstylecanA01VompRobolab(LibraryObject):
    """This jug is made of glossy, opaque metal with a sleek, cylindrical body and a sturdy handle. Its metallic finish gives it a modern, reflective appearance."""
    name = "metalfstylecan_a01_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/metalfstylecan_a01/metalfstylecan_a01.usd"
    )
    dims = (0.1099, 0.1673, 0.2580)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=7850.0),
    }


@register_asset
class MilkjugA01VompRobolab(LibraryObject):
    """This milk jug is made of translucent natural plastic with a glossy white cap. It features a sturdy handle and a wide mouth for easy pouring."""
    name = "milkjug_a01_vomp_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/milkjug_a01/milkjug_a01.usd"
    )
    dims = (0.0768, 0.0766, 0.1724)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class MilkjugA02VompRobolab(LibraryObject):
    """This milk jug is made of opaque plastic with a natural translucent body and a glossy white handle. It features a screw-on cap and a rectangular shape with rounded edges for easy pouring."""
    name = "milkjug_a02_vomp_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/milkjug_a02/milkjug_a02.usd"
    )
    dims = (0.1093, 0.1093, 0.2168)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=950.0),
    }


@register_asset
class MilkjugA03VompRobolab(LibraryObject):
    """This milk jug features a natural translucent body with an opaque glossy white handle and cap. Made of plastic, it has a classic rectangular shape with a convenient pour spout for easy handling."""
    name = "milkjug_a03_vomp_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/milkjug_a03/milkjug_a03.usd"
    )
    dims = (0.1502, 0.1502, 0.2177)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=950.0),
    }


@register_asset
class MoppingbucketB01VompRobolab(LibraryObject):
    """This mop bucket is made of opaque blue plastic with a black handle and features a decal. It has a sturdy construction, designed for durability and ease of use during cleaning tasks."""
    name = "moppingbucket_b01_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/moppingbucket_b01/moppingbucket_b01.usd"
    )
    scale = (0.934, 0.934, 0.934)
    dims = (0.3715, 0.5356, 0.4777)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1000.0),
    }


@register_asset
class NaturalbostonroundbottleA01VompRobolab(LibraryObject):
    """The bottle features a translucent natural plastic body with a rounded shape and a glossy white opaque plastic cap, providing a sleek and functional design."""
    name = "naturalbostonroundbottle_a01_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/naturalbostonroundbottle_a01/naturalbostonroundbottle_a01.usd"
    )
    dims = (0.0569, 0.0569, 0.1323)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class NaturalbostonroundbottleA02VompRobolab(LibraryObject):
    """The bottle features a natural, translucent plastic body with a smooth, rounded shape and a glossy white opaque cap, providing a clean and simple appearance."""
    name = "naturalbostonroundbottle_a02_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/naturalbostonroundbottle_a02/naturalbostonroundbottle_a02.usd"
    )
    dims = (0.0775, 0.0775, 0.1518)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class NaturalbostonroundbottleA03VompRobolab(LibraryObject):
    """The bottle has a rounded, natural-colored translucent plastic body and a glossy white opaque plastic cap, creating a sleek and modern appearance."""
    name = "naturalbostonroundbottle_a03_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/naturalbostonroundbottle_a03/naturalbostonroundbottle_a03.usd"
    )
    dims = (0.0901, 0.0901, 0.1907)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class PlasticjerricanAVompRobolab(LibraryObject):
    """This 20-liter jerry can is made of durable opaque plastic with a translucent section for monitoring liquid levels. It features a sturdy handle and a secure screw cap for easy transport and storage."""
    name = "plasticjerrican_a_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/plasticjerrican_a/plasticjerrican_a.usd"
    )
    dims = (0.1872, 0.2045, 0.2270)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=950.0),
    }


@register_asset
class PlasticjerricanA01VompRobolab(LibraryObject):
    """This 20-liter jerry can is made of durable opaque plastic with a translucent section for monitoring liquid levels. It features a sturdy handle and a secure screw cap for easy transport and storage."""
    name = "plasticjerrican_a01_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/plasticjerrican_a01/plasticjerrican_a01.usd"
    )
    dims = (0.1872, 0.2045, 0.2270)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=950.0),
    }


@register_asset
class PlasticjerricanA02VompRobolab(LibraryObject):
    """This 20-liter jerry can is made of durable opaque plastic, featuring a blue color. It has a rectangular shape with a built-in handle and a secure screw cap for easy transportation and storage of liquids."""
    name = "plasticjerrican_a02_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/plasticjerrican_a02/plasticjerrican_a02.usd"
    )
    dims = (0.2171, 0.2370, 0.2932)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class PlasticjerricanA03VompRobolab(LibraryObject):
    """This 20-liter jerry can features a sturdy, opaque plastic construction with a blue body and a white cap. Its rectangular shape includes a built-in handle for easy carrying and a spout for pouring."""
    name = "plasticjerrican_a03_vomp_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/plasticjerrican_a03/plasticjerrican_a03.usd"
    )
    dims = (0.2525, 0.2793, 0.3598)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=950.0),
    }


@register_asset
class PlasticjerricanA04VompRobolab(LibraryObject):
    """This 20-liter jerry can features a sturdy, opaque plastic construction with a blue body and a white cap. Its rectangular shape and built-in handle ensure easy handling and efficient storage."""
    name = "plasticjerrican_a04_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/plasticjerrican_a04/plasticjerrican_a04.usd"
    )
    dims = (0.2525, 0.2793, 0.5103)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class PlasticpailA01VompRobolab(LibraryObject):
    """This pail is made of opaque, glossy plastic with a smooth, rounded body and a sturdy handle. It features a bright color and a simple design, ideal for carrying liquids or small items."""
    name = "plasticpail_a01_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/plasticpail_a01/plasticpail_a01.usd"
    )
    dims = (0.1290, 0.1290, 0.1238)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=950.0),
    }


@register_asset
class PlasticpailA02VompRobolab(LibraryObject):
    """This pail is made of opaque green plastic with a glossy finish. It features a sturdy handle for easy carrying and a smooth, rounded body, ideal for various household or gardening tasks."""
    name = "plasticpail_a02_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/plasticpail_a02/plasticpail_a02.usd"
    )
    dims = (0.1960, 0.1844, 0.2000)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class PlasticpailA03VompRobolab(LibraryObject):
    """This pail is made of glossy black plastic with a smooth, opaque finish. It features a shiny metal handle for easy carrying, combining durability with a sleek, modern appearance."""
    name = "plasticpail_a03_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/plasticpail_a03/plasticpail_a03.usd"
    )
    dims = (0.3329, 0.3236, 0.2984)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=950.0),
    }


@register_asset
class PlasticpailA04VompRobolab(LibraryObject):
    """This pail is made of opaque orange plastic with a glossy finish and features a shiny metal handle. Its sturdy construction makes it suitable for carrying liquids or other materials."""
    name = "plasticpail_a04_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/plasticpail_a04/plasticpail_a04.usd"
    )
    dims = (0.3329, 0.3236, 0.4508)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=950.0),
    }


@register_asset
class PlateLargeVompRobolab(LibraryObject):
    """This large plate is made of opaque plastic with a smooth, flat surface and a slightly raised rim. It features a simple, solid color, ideal for serving meals or displaying food."""
    name = "plate_large_vomp_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/plate_large/plate_large.usd"
    )
    dims = (0.3308, 0.3308, 0.0258)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class PlateSmallVompRobolab(LibraryObject):
    """This small plate is made of opaque plastic with a smooth, flat surface and slightly raised edges. It is lightweight and durable, suitable for serving small portions or snacks."""
    name = "plate_small_vomp_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/plate_small/plate_small.usd"
    )
    dims = (0.2976, 0.2976, 0.0212)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class PumpkinlargeVompRobolab(LibraryObject):
    """This large pumpkin is round and ribbed, with a vibrant orange skin and a sturdy green stem. Its opaque, organic surface is smooth, making it ideal for carving or autumn decoration."""
    name = "pumpkinlarge_vomp_robolab"
    tags = ["object", "graspable", "food", "vegetable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/pumpkinlarge/pumpkinlarge.usd"
    )
    dims = (0.0890, 0.0901, 0.0890)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.0
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=900.0),
    }


@register_asset
class PumpkinsmallVompRobolab(LibraryObject):
    """This small pumpkin has a round, ribbed shape with a vibrant orange color and a sturdy green stem. Its smooth, opaque skin gives it a classic autumnal appearance, perfect for decoration or cooking."""
    name = "pumpkinsmall_vomp_robolab"
    tags = ["object", "graspable", "food", "vegetable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/pumpkinsmall/pumpkinsmall.usd"
    )
    dims = (0.0760, 0.0760, 0.0752)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.0
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=2800.0),
    }


@register_asset
class RackL04VompRobolab(LibraryObject):
    """A rack_l04 converted from SimReady."""
    name = "rack_l04_vomp_robolab"
    tags = ["object", "fixture", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/rack_l04/rack_l04.usd"
    )
    scale = (0.312, 0.312, 0.312)
    dims = (0.7000, 1.6000, 2.5000)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=7850.0),
    }


@register_asset
class ScrewtoppailA01VompRobolab(LibraryObject):
    """This pail features a glossy, opaque plastic body with a screw top. It comes in two color options: blue and red, offering a secure seal for storing various materials."""
    name = "screwtoppail_a01_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/screwtoppail_a01/screwtoppail_a01.usd"
    )
    dims = (0.2712, 0.2703, 0.2483)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=950.0),
    }


@register_asset
class ScrewtoppailA02VompRobolab(LibraryObject):
    """This pail features a glossy, opaque plastic body with a screw-top lid. It comes in three colors: white, blue, and red, providing a sturdy and secure storage solution."""
    name = "screwtoppail_a02_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/screwtoppail_a02/screwtoppail_a02.usd"
    )
    dims = (0.3295, 0.3295, 0.2716)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=950.0),
    }


@register_asset
class ScrewtoppailA03VompRobolab(LibraryObject):
    """This pail features a glossy, opaque plastic body with a screw-top lid. It comes in vibrant colors, including blue and red, providing a secure and colorful storage solution."""
    name = "screwtoppail_a03_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/screwtoppail_a03/screwtoppail_a03.usd"
    )
    dims = (0.3295, 0.3295, 0.3229)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=950.0),
    }


@register_asset
class ScrewtoppailA04VompRobolab(LibraryObject):
    """This pail features a glossy, opaque plastic body with a screw top lid. It is available in blue and red variations, offering a durable and secure storage solution."""
    name = "screwtoppail_a04_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/screwtoppail_a04/screwtoppail_a04.usd"
    )
    dims = (0.3295, 0.3295, 0.4781)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=950.0),
    }


@register_asset
class SeahornVaseVompRobolab(LibraryObject):
    """The seahorn vase is crafted from opaque concrete, featuring a unique design that resembles a twisted sea horn. Its textured surface and muted gray color give it a modern, sculptural appearance."""
    name = "seahorn_vase_vomp_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/seahorn_vase/seahorn_vase.usd"
    )
    dims = (0.3365, 0.3365, 0.3680)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.009999999776482582
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=2500.0),
    }


@register_asset
class ServingBowlVompRobolab(LibraryObject):
    """The serving bowl is made of opaque metal with a smooth, polished finish. It features a wide, shallow design, ideal for presenting food, and has a subtle metallic sheen that adds elegance to any table setting."""
    name = "serving_bowl_vomp_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/serving_bowl/serving_bowl.usd"
    )
    dims = (0.1597, 0.1597, 0.0520)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=2700.0),
    }


@register_asset
class SmRackM01VompRobolab(LibraryObject):
    """A sm_rack_m01 converted from SimReady."""
    name = "sm_rack_m01_vomp_robolab"
    tags = ["object", "fixture", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/sm_rack_m01/sm_rack_m01.usd"
    )
    scale = (0.25, 0.25, 0.25)
    dims = (1.1000, 2.0000, 2.5000)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class Spatula01VompRobolab(LibraryObject):
    """The spatula features a smooth wooden handle and a flat, wide wooden blade. Its natural wood grain and warm brown color give it a rustic appearance, ideal for cooking and serving."""
    name = "spatula_01_vomp_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/spatula_01/spatula_01.usd"
    )
    dims = (0.2653, 0.0858, 0.0135)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class Spatula02VompRobolab(LibraryObject):
    """This spatula features a flat, wide wooden blade and a smooth wooden handle, both with a natural wood grain finish, providing a sturdy and heat-resistant tool for cooking."""
    name = "spatula_02_vomp_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/spatula_02/spatula_02.usd"
    )
    dims = (0.3558, 0.0914, 0.0173)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1100.0),
    }


@register_asset
class Spatula03VompRobolab(LibraryObject):
    """This spatula has a smooth wooden handle and a flat wooden blade, both in a natural wood finish. It's designed for flipping and serving, with a sturdy, ergonomic grip."""
    name = "spatula_03_vomp_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/spatula_03/spatula_03.usd"
    )
    dims = (0.2691, 0.0621, 0.0135)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=600.0),
    }


@register_asset
class Spatula04VompRobolab(LibraryObject):
    """This spatula features a smooth wooden handle and a matching wooden flat blade, both with a natural finish. Its simple design is ideal for flipping and serving food."""
    name = "spatula_04_vomp_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/spatula_04/spatula_04.usd"
    )
    dims = (0.2701, 0.0664, 0.0135)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1100.0),
    }


@register_asset
class Spatula05VompRobolab(LibraryObject):
    """This spatula features a wooden handle and a matching wooden flat blade, both with a smooth, natural finish. It is designed for flipping and turning food while cooking."""
    name = "spatula_05_vomp_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/spatula_05/spatula_05.usd"
    )
    dims = (0.2752, 0.0752, 0.0074)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1100.0),
    }


@register_asset
class Spatula06VompRobolab(LibraryObject):
    """This spatula features a smooth wooden handle and a flat wooden blade, both with a natural finish. It is designed for flipping and serving, showcasing a simple and functional design."""
    name = "spatula_06_vomp_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/spatula_06/spatula_06.usd"
    )
    dims = (0.2749, 0.0772, 0.0074)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class Spatula07VompRobolab(LibraryObject):
    """This spatula features a flat, rectangular wooden head and a matching wooden handle. The natural wood grain is visible, giving it a rustic appearance, and it is designed for flipping and serving food."""
    name = "spatula_07_vomp_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/spatula_07/spatula_07.usd"
    )
    dims = (0.2756, 0.0699, 0.0160)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class Spatula08VompRobolab(LibraryObject):
    """This spatula features a smooth wooden handle and a flat wooden blade, both with a natural finish. Its simple design highlights the warm, earthy tones of the wood, ideal for cooking and serving."""
    name = "spatula_08_vomp_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/spatula_08/spatula_08.usd"
    )
    dims = (0.2727, 0.0808, 0.0135)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=600.0),
    }


@register_asset
class Spatula09VompRobolab(LibraryObject):
    """This spatula features a smooth wooden handle and a flat wooden blade, both with a natural finish. It is designed for flipping and stirring, offering a comfortable grip and sturdy construction."""
    name = "spatula_09_vomp_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/spatula_09/spatula_09.usd"
    )
    dims = (0.2687, 0.0620, 0.0135)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=600.0),
    }


@register_asset
class Spatula10VompRobolab(LibraryObject):
    """This spatula features a smooth wooden handle and a flat wooden blade, both with a natural finish. It is designed for flipping and serving, offering a sturdy and heat-resistant tool for cooking."""
    name = "spatula_10_vomp_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/spatula_10/spatula_10.usd"
    )
    dims = (0.2690, 0.0620, 0.0135)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class Spatula11VompRobolab(LibraryObject):
    """This spatula features a flat, rectangular wooden head and a matching wooden handle, both with a smooth, natural finish, ideal for flipping and serving food."""
    name = "spatula_11_vomp_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/spatula_11/spatula_11.usd"
    )
    dims = (0.0620, 0.2687, 0.0135)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class Spatula12VompRobolab(LibraryObject):
    """This spatula features a flat, rectangular wooden head and a smooth wooden handle. The natural wood grain is visible, giving it a rustic appearance, and it is ideal for flipping or serving food."""
    name = "spatula_12_vomp_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/spatula_12/spatula_12.usd"
    )
    dims = (0.2746, 0.0667, 0.0073)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class Spatula13VompRobolab(LibraryObject):
    """This spatula features a flat, wooden head and handle, both with a smooth, natural finish. The design is simple and functional, ideal for flipping or stirring in cooking."""
    name = "spatula_13_vomp_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/spatula_13/spatula_13.usd"
    )
    dims = (0.0587, 0.2728, 0.0073)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class Spatula14VompRobolab(LibraryObject):
    """This spatula features a smooth wooden handle and a flat wooden head, both with a natural finish. It's designed for flipping and turning food, offering durability and a comfortable grip."""
    name = "spatula_14_vomp_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/spatula_14/spatula_14.usd"
    )
    dims = (0.0588, 0.2730, 0.0073)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=600.0),
    }


@register_asset
class Spatula15VompRobolab(LibraryObject):
    """This spatula features a flat, rectangular wooden head and a smooth wooden handle, both with a natural finish. The seamless design highlights its simplicity and functionality for cooking and serving."""
    name = "spatula_15_vomp_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/spatula_15/spatula_15.usd"
    )
    dims = (0.2728, 0.0587, 0.0073)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class Spatula16VompRobolab(LibraryObject):
    """This spatula features a wooden handle and a flat wooden blade, both with a smooth, natural finish. It's designed for flipping and serving, showcasing the warm tones and grain patterns typical of wood."""
    name = "spatula_16_vomp_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/spatula_16/spatula_16.usd"
    )
    dims = (0.0588, 0.2730, 0.0073)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1100.0),
    }


@register_asset
class Spatula17VompRobolab(LibraryObject):
    """This spatula features a smooth wooden handle and a flat wooden head, both with a natural finish. It is designed for flipping and serving, offering a rustic and durable kitchen tool."""
    name = "spatula_17_vomp_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/spatula_17/spatula_17.usd"
    )
    dims = (0.2731, 0.0588, 0.0074)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1100.0),
    }


@register_asset
class Spatula18VompRobolab(LibraryObject):
    """This spatula features a smooth wooden handle and a flat, wide wooden blade. The natural wood grain is visible, giving it a rustic appearance, and it's designed for flipping or serving food."""
    name = "spatula_18_vomp_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/spatula_18/spatula_18.usd"
    )
    dims = (0.2728, 0.0587, 0.0074)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=600.0),
    }


@register_asset
class Spatula19VompRobolab(LibraryObject):
    """This spatula features a flat, wide wooden head and a smooth wooden handle, both in a natural wood finish, providing a sturdy and heat-resistant tool for cooking and flipping."""
    name = "spatula_19_vomp_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/spatula_19/spatula_19.usd"
    )
    dims = (0.2731, 0.0588, 0.0074)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class SpoonBigVompRobolab(LibraryObject):
    """This large spoon is made of opaque steel with a smooth, shiny finish. It features a deep, rounded bowl and a long, sturdy handle, ideal for serving or cooking."""
    name = "spoon_big_vomp_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/spoon_big/spoon_big.usd"
    )
    dims = (0.1695, 0.0421, 0.0171)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=7850.0),
    }


@register_asset
class SpoonSmallVompRobolab(LibraryObject):
    """This small spoon is crafted from opaque steel, featuring a smooth, shiny surface. Its bowl is rounded, and the handle is slender and slightly curved, ideal for stirring or serving small portions."""
    name = "spoon_small_vomp_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/spoon_small/spoon_small.usd"
    )
    dims = (0.1478, 0.0353, 0.0132)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class SquarepailA01VompRobolab(LibraryObject):
    """This pail has a glossy, square-shaped body made of opaque plastic, complemented by a shiny metal handle. Its modern design combines durability with a sleek appearance."""
    name = "squarepail_a01_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/squarepail_a01/squarepail_a01.usd"
    )
    dims = (0.2812, 0.2477, 0.2255)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class SquarepailA02VompRobolab(LibraryObject):
    """This pail features a square shape with a glossy black plastic body and a shiny metal handle. Its sleek design combines durable materials, making it suitable for various carrying and storage tasks."""
    name = "squarepail_a02_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/squarepail_a02/squarepail_a02.usd"
    )
    dims = (0.2547, 0.2386, 0.3612)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class UtilitybucketA01VompRobolab(LibraryObject):
    """The utility bucket features a glossy metal handle and a red opaque plastic body with gray decals. It includes a gray plastic lid, providing a sturdy and functional design for various uses."""
    name = "utilitybucket_a01_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/utilitybucket_a01/utilitybucket_a01.usd"
    )
    dims = (0.3383, 0.3281, 0.2588)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class UtilitybucketA02VompRobolab(LibraryObject):
    """The utility bucket features a glossy metal handle and a red opaque plastic body with grey decals. Its sturdy construction is ideal for carrying heavy loads, and the grey plastic adds a sleek, modern touch."""
    name = "utilitybucket_a02_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/utilitybucket_a02/utilitybucket_a02.usd"
    )
    scale = (0.96, 0.96, 0.96)
    dims = (0.3645, 0.3534, 0.3019)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
    }


@register_asset
class UtilityjugA01VompRobolab(LibraryObject):
    """This 20-liter jerry can is made of durable, opaque plastic with a translucent section for monitoring liquid levels. It features a sturdy handle and a secure screw cap for easy transport and storage."""
    name = "utilityjug_a01_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/utilityjug_a01/utilityjug_a01.usd"
    )
    dims = (0.0965, 0.0969, 0.2803)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=950.0),
    }


@register_asset
class UtilityjugA02VompRobolab(LibraryObject):
    """The utility jug is a 20-liter translucent jerry can made of natural plastic, featuring an opaque section for durability. Its design includes a sturdy handle and a secure screw cap for easy transport and storage."""
    name = "utilityjug_a02_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/utilityjug_a02/utilityjug_a02.usd"
    )
    dims = (0.1242, 0.1245, 0.2662)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=950.0),
    }


@register_asset
class UtilityjugA03VompRobolab(LibraryObject):
    """The utility jug is a 20-liter jerry can with a translucent body and white accents. Made from durable opaque plastic, it features a sturdy handle and a secure screw cap for easy transport and storage."""
    name = "utilityjug_a03_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/utilityjug_a03/utilityjug_a03.usd"
    )
    dims = (0.1713, 0.1715, 0.3357)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=950.0),
    }


@register_asset
class WhitepackerbottleA01VompRobolab(LibraryObject):
    """This bottle is made of glossy, opaque white plastic with a smooth, rounded body and a secure screw-on cap, designed for packaging various products."""
    name = "whitepackerbottle_a01_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/whitepackerbottle_a01/whitepackerbottle_a01.usd"
    )
    dims = (0.0433, 0.0433, 0.0768)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class WhitepackerbottleA02VompRobolab(LibraryObject):
    """The bottle is made of glossy, opaque white plastic with a smooth, cylindrical shape and a secure screw-on cap, ideal for packaging pills or supplements."""
    name = "whitepackerbottle_a02_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/whitepackerbottle_a02/whitepackerbottle_a02.usd"
    )
    dims = (0.0511, 0.0511, 0.0871)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=950.0),
    }


@register_asset
class WhitepackerbottleA03VompRobolab(LibraryObject):
    """This bottle is made of glossy, opaque white plastic with a smooth, rounded body and a secure screw-on cap, ideal for packaging pills or supplements."""
    name = "whitepackerbottle_a03_vomp_robolab"
    tags = ["object", "container", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/whitepackerbottle_a03/whitepackerbottle_a03.usd"
    )
    dims = (0.0615, 0.0615, 0.1086)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.30000001192092896
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=800.0),
    }


@register_asset
class WireshelvingA01VompRobolab(LibraryObject):
    """A wireshelving_a01 converted from SimReady."""
    name = "wireshelving_a01_vomp_robolab"
    tags = ["object", "fixture", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/vomp/wireshelving_a01/wireshelving_a01.usd"
    )
    scale = (0.652, 0.652, 0.652)
    dims = (0.4635, 0.7664, 0.8701)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(density=7850.0),
    }


@register_asset
class BananaYcbRobolab(LibraryObject):
    """These bananas are long and curved, with a smooth yellow skin and slightly green tips. They are commonly known for their sweet taste and are often eaten raw or used in various dishes."""
    name = "banana_ycb_robolab"
    tags = ["object", "graspable", "food", "fruit", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/ycb/banana.usd"
    )
    dims = (0.1089, 0.1784, 0.0367)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.11999999731779099),
    }


@register_asset
class BowlYcbRobolab(LibraryObject):
    """The bowls are round with a smooth, glossy surface and a vibrant red color. They feature a speckled design and a slightly raised rim, adding a decorative touch."""
    name = "bowl_ycb_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/ycb/bowl.usd"
    )
    dims = (0.1614, 0.1611, 0.0550)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.5),
    }


@register_asset
class Bowl2YcbRobolab(LibraryObject):
    """The bowls are round with a smooth, glossy surface and a vibrant red color. They feature a speckled design and a slightly raised rim, adding a decorative touch."""
    name = "bowl2_ycb_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/ycb/bowl2.usd"
    )
    dims = (0.1614, 0.1611, 0.0550)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.5),
    }


@register_asset
class BrickYcbRobolab(LibraryObject):
    """The object is a rectangular brick with a deep red-brown color. It features three circular indentations on one side, giving it a textured appearance."""
    name = "brick_ycb_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/ycb/brick.usd"
    )
    dims = (0.0526, 0.0778, 0.0511)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=2.5),
    }


@register_asset
class CheezItYcbRobolab(LibraryObject):
    """The package is predominantly red with white and yellow accents, featuring bold text that reads 'Cheez-It' and 'Original.' It showcases images of the square crackers and includes nutritional information and serving suggestions on the sides."""
    name = "cheez_it_ycb_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/ycb/cheez_it.usd"
    )
    dims = (0.0717, 0.1640, 0.2135)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.22499999403953552),
    }


@register_asset
class ChocolatePuddingYcbRobolab(LibraryObject):
    """The package is primarily red and brown with the prominent text 'JELL-O' in bold letters. There are nutritional facts and ingredient details printed on the sides, and the overall shape is a rectangular box."""
    name = "chocolate_pudding_ycb_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/ycb/chocolate_pudding.usd"
    )
    dims = (0.1378, 0.1288, 0.0388)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.25),
    }


@register_asset
class ClampYcbRobolab(LibraryObject):
    """The clamp features a sturdy black body with a bright orange tip for added visibility. Its ergonomic design includes grips that provide comfort during use."""
    name = "clamp_ycb_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/ycb/clamp.usd"
    )
    dims = (0.1200, 0.1713, 0.0392)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.5),
    }


@register_asset
class CoffeeCanYcbRobolab(LibraryObject):
    """The can is cylindrical with a shiny blue exterior, featuring the text 'Master Chef' prominently displayed on the front. It has white and yellow accents and contains ground coffee, with brewing instructions printed on the label."""
    name = "coffee_can_ycb_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/ycb/coffee_can.usd"
    )
    dims = (0.1023, 0.1024, 0.1401)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.296999990940094),
    }


@register_asset
class CordlessDrillYcbRobolab(LibraryObject):
    """The cordless drill features a sleek design with a predominantly orange body and black grip. It includes a chuck at the front for securing drill bits and has a battery pack attached to the base."""
    name = "cordless_drill_ycb_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/ycb/cordless_drill.usd"
    )
    dims = (0.1842, 0.1874, 0.0573)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=1.5),
    }


@register_asset
class DryEraseMarkerYcbRobolab(LibraryObject):
    """The marker features a cylindrical body with a black cap and a white barrel, prominently displaying the brand name 'Expo' in bold text. The tip is chiseled, designed for both broad and fine lines, and the overall color scheme emphasizes its dry-erase functionality."""
    name = "dry_erase_marker_ycb_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/ycb/dry_erase_marker.usd"
    )
    dims = (0.0210, 0.1208, 0.0189)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.019999999552965164),
    }


@register_asset
class JelloYcbRobolab(LibraryObject):
    """The object is a rectangular box with a bright red color featuring the brand name 'JELL-O' prominently displayed on the top. The sides contain additional information such as flavors and nutritional facts, with a glossy finish."""
    name = "jello_ycb_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/ycb/jello.usd"
    )
    dims = (0.0894, 0.1011, 0.0301)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.20000000298023224),
    }


@register_asset
class MugYcbRobolab(LibraryObject):
    """The object is a pair of red ceramic mugs with a glossy finish. They feature a curved handle and have subtle vertical stripes along the body."""
    name = "mug_ycb_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/ycb/mug.usd"
    )
    dims = (0.1170, 0.0931, 0.0814)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.30000001192092896),
    }


@register_asset
class MustardYcbRobolab(LibraryObject):
    """The object features a bright yellow plastic squeeze bottle with a conical cap. It is labeled with bold text and nutritional information, highlighting its flavor and usage."""
    name = "mustard_ycb_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/ycb/mustard.usd"
    )
    dims = (0.0971, 0.0666, 0.1914)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.5),
    }


@register_asset
class PitcherYcbRobolab(LibraryObject):
    """This pitcher has a smooth, rounded body and a curved handle for easy pouring. It features a deep blue color with a tapered spout at the top."""
    name = "pitcher_ycb_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/ycb/pitcher.usd"
    )
    dims = (0.1490, 0.1448, 0.2426)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.5),
    }


@register_asset
class ScissorsYcbRobolab(LibraryObject):
    """These scissors have a grey plastic handle with orange accents. The blades are shiny metal and are designed for cutting through paper and other materials."""
    name = "scissors_ycb_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/ycb/scissors.usd"
    )
    dims = (0.0961, 0.2015, 0.0157)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.20000000298023224),
    }


@register_asset
class SoftScrubYcbRobolab(LibraryObject):
    """The container is a tall, white plastic bottle with a slightly curved shape. It features a colorful label with blue and green hues, including the text 'Soft Scrub' prominently displayed on the front."""
    name = "soft_scrub_ycb_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/ycb/soft_scrub.usd"
    )
    dims = (0.1024, 0.0677, 0.2506)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=1.5),
    }


@register_asset
class SpamCanYcbRobolab(LibraryObject):
    """The can is metallic and rectangular with a glossy finish, predominantly featuring a vibrant green and gold color scheme. It prominently displays the brand name 'SPAM' in bold, white letters, along with colorful images of food dishes on its label."""
    name = "spam_can_ycb_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/ycb/spam_can.usd"
    )
    dims = (0.1021, 0.0603, 0.0836)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.3499999940395355),
    }


@register_asset
class SpringClampYcbRobolab(LibraryObject):
    """This object features a robust black plastic design with a textured grip for easy handling. The clamp has bright orange tips that are used to secure items together firmly."""
    name = "spring_clamp_ycb_robolab"
    tags = ["object", "graspable", "tool", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/ycb/spring_clamp.usd"
    )
    dims = (0.2098, 0.1644, 0.0363)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.5),
    }


@register_asset
class SugarBoxYcbRobolab(LibraryObject):
    """The box is primarily yellow and white, featuring the word 'Domino' prominently on the front. It has nutritional information and usage instructions printed on the sides in black text."""
    name = "sugar_box_ycb_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/ycb/sugar_box.usd"
    )
    dims = (0.0495, 0.0940, 0.1760)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=1.0),
    }


@register_asset
class TomatoSoupCanYcbRobolab(LibraryObject):
    """The can is made of metal and features a vibrant red and white label. It displays the text 'Campbell's Tomato Soup' prominently along with nutritional information on the back."""
    name = "tomato_soup_can_ycb_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/ycb/tomato_soup_can.usd"
    )
    dims = (0.0679, 0.0677, 0.1020)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.44999998807907104),
    }


@register_asset
class TunaCanYcbRobolab(LibraryObject):
    """The can is cylindrical with a metallic surface featuring a vibrant blue label. It has text indicating the contents, nutritional information, and branding, along with a design that includes a cartoon fish."""
    name = "tuna_can_ycb_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/ycb/tuna_can.usd"
    )
    dims = (0.0855, 0.0856, 0.0335)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.20000000298023224),
    }


@register_asset
class WoodBlockYcbRobolab(LibraryObject):
    """The object is a rectangular block made of light-colored wood, with a smooth surface. It has straight edges and a natural wood grain texture, showcasing a warm, beige hue."""
    name = "wood_block_ycb_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/ycb/wood_block.usd"
    )
    dims = (0.1040, 0.1035, 0.2059)
    # Physics material: dynamic_friction=2.0, static_friction=2.0, restitution=0.10000000149011612
    spawn_cfg_addon = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=2.5),
    }

# --- Lightwheel RIGID objects ---


@register_asset
class Alcohol008(LibraryObject):
    name = "alcohol_008"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Alcohol008/Alcohol008.usd")
    object_type = ObjectType.RIGID

@register_asset
class Alcohol014(LibraryObject):
    name = "alcohol_014"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Alcohol014/Alcohol014.usd")
    object_type = ObjectType.RIGID

@register_asset
class AlphabetSoup001(LibraryObject):
    name = "alphabet_soup_001"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/AlphabetSoup001/AlphabetSoup001.usd")
    object_type = ObjectType.RIGID

@register_asset
class Apple006(LibraryObject):
    name = "apple_006"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Apple006/Apple006.usd")
    object_type = ObjectType.RIGID

@register_asset
class Apple012(LibraryObject):
    name = "apple_012"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Apple012/Apple012.usd")
    object_type = ObjectType.RIGID

@register_asset
class Apricot001(LibraryObject):
    name = "apricot_001"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Apricot001/Apricot001.usd")
    object_type = ObjectType.RIGID

@register_asset
class Apricot002(LibraryObject):
    name = "apricot_002"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Apricot002/Apricot002.usd")
    object_type = ObjectType.RIGID

@register_asset
class Artichoke007(LibraryObject):
    name = "artichoke_007"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Artichoke007/Artichoke007.usd")
    object_type = ObjectType.RIGID

@register_asset
class Artichoke010(LibraryObject):
    name = "artichoke_010"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Artichoke010/Artichoke010.usd")
    object_type = ObjectType.RIGID

@register_asset
class Asparagus008(LibraryObject):
    name = "asparagus_008"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Asparagus008/Asparagus008.usd")
    object_type = ObjectType.RIGID

@register_asset
class Asparagus011(LibraryObject):
    name = "asparagus_011"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Asparagus011/Asparagus011.usd")
    object_type = ObjectType.RIGID

@register_asset
class Avocado003(LibraryObject):
    name = "avocado_003"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Avocado003/Avocado003.usd")
    object_type = ObjectType.RIGID

@register_asset
class Avocado016(LibraryObject):
    name = "avocado_016"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Avocado016/Avocado016.usd")
    object_type = ObjectType.RIGID

@register_asset
class Bacon004(LibraryObject):
    name = "bacon_004"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Bacon004/Bacon004.usd")
    object_type = ObjectType.RIGID

@register_asset
class Bacon007(LibraryObject):
    name = "bacon_007"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Bacon007/Bacon007.usd")
    object_type = ObjectType.RIGID

@register_asset
class Bagel005(LibraryObject):
    name = "bagel_005"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Bagel005/Bagel005.usd")
    object_type = ObjectType.RIGID

@register_asset
class Bagel018(LibraryObject):
    name = "bagel_018"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Bagel018/Bagel018.usd")
    object_type = ObjectType.RIGID

@register_asset
class BaggedFood007(LibraryObject):
    name = "bagged_food_007"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/BaggedFood007/BaggedFood007.usd")
    object_type = ObjectType.RIGID

@register_asset
class BaggedFood009(LibraryObject):
    name = "bagged_food_009"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/BaggedFood009/BaggedFood009.usd")
    object_type = ObjectType.RIGID

@register_asset
class Baguette015(LibraryObject):
    name = "baguette_015"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Baguette015/Baguette015.usd")
    object_type = ObjectType.RIGID

@register_asset
class Baguette020(LibraryObject):
    name = "baguette_020"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Baguette020/Baguette020.usd")
    object_type = ObjectType.RIGID

@register_asset
class BakingSheet006(LibraryObject):
    name = "baking_sheet_006"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/BakingSheet006/BakingSheet006.usd")
    object_type = ObjectType.RIGID

@register_asset
class BakingSheet008(LibraryObject):
    name = "baking_sheet_008"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/BakingSheet008/BakingSheet008.usd")
    object_type = ObjectType.RIGID

@register_asset
class Banana005(LibraryObject):
    name = "banana_005"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Banana005/Banana005.usd")
    object_type = ObjectType.RIGID

@register_asset
class Banana031(LibraryObject):
    name = "banana_031"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Banana031/Banana031.usd")
    object_type = ObjectType.RIGID

@register_asset
class Bar015(LibraryObject):
    name = "bar_015"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Bar015/Bar015.usd")
    object_type = ObjectType.RIGID

@register_asset
class Bar016(LibraryObject):
    name = "bar_016"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Bar016/Bar016.usd")
    object_type = ObjectType.RIGID

@register_asset
class Basket058(LibraryObject):
    name = "basket_058"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Basket058/Basket058.usd")
    object_type = ObjectType.RIGID

@register_asset
class Beer009(LibraryObject):
    name = "beer_009"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Beer009/Beer009.usd")
    object_type = ObjectType.RIGID

@register_asset
class Beer016(LibraryObject):
    name = "beer_016"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Beer016/Beer016.usd")
    object_type = ObjectType.RIGID

@register_asset
class Beer019(LibraryObject):
    name = "beer_019"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Beer019/Beer019.usd")
    object_type = ObjectType.RIGID

@register_asset
class Beet007(LibraryObject):
    name = "beet_007"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Beet007/Beet007.usd")
    object_type = ObjectType.RIGID

@register_asset
class Beet009(LibraryObject):
    name = "beet_009"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Beet009/Beet009.usd")
    object_type = ObjectType.RIGID

@register_asset
class BellPepper004(LibraryObject):
    name = "bell_pepper_004"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/BellPepper004/BellPepper004.usd")
    object_type = ObjectType.RIGID

@register_asset
class BellPepper006(LibraryObject):
    name = "bell_pepper_006"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/BellPepper006/BellPepper006.usd")
    object_type = ObjectType.RIGID

@register_asset
class Book042(LibraryObject):
    name = "book_042"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Book042/Book042.usd")
    object_type = ObjectType.RIGID

@register_asset
class Book043(LibraryObject):
    name = "book_043"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Book043/Book043.usd")
    object_type = ObjectType.RIGID

@register_asset
class Bottle054(LibraryObject):
    name = "bottle_054"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Bottle054/Bottle054.usd")
    object_type = ObjectType.RIGID

@register_asset
class BottleOpener001(LibraryObject):
    name = "bottle_opener_001"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/BottleOpener001/BottleOpener001.usd")
    object_type = ObjectType.RIGID

@register_asset
class BottleOpener005(LibraryObject):
    name = "bottle_opener_005"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/BottleOpener005/BottleOpener005.usd")
    object_type = ObjectType.RIGID

@register_asset
class BottledDrink005(LibraryObject):
    name = "bottled_drink_005"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/BottledDrink005/BottledDrink005.usd")
    object_type = ObjectType.RIGID

@register_asset
class BottledDrink012(LibraryObject):
    name = "bottled_drink_012"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/BottledDrink012/BottledDrink012.usd")
    object_type = ObjectType.RIGID

@register_asset
class BottledWater009(LibraryObject):
    name = "bottled_water_009"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/BottledWater009/BottledWater009.usd")
    object_type = ObjectType.RIGID

@register_asset
class BottledWater020(LibraryObject):
    name = "bottled_water_020"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/BottledWater020/BottledWater020.usd")
    object_type = ObjectType.RIGID

@register_asset
class Bowl008(LibraryObject):
    name = "bowl_008"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Bowl008/Bowl008.usd")
    object_type = ObjectType.RIGID

@register_asset
class Bowl040(LibraryObject):
    name = "bowl_040"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Bowl040/Bowl040.usd")
    object_type = ObjectType.RIGID

@register_asset
class BoxedDrink007(LibraryObject):
    name = "boxed_drink_007"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/BoxedDrink007/BoxedDrink007.usd")
    object_type = ObjectType.RIGID

@register_asset
class BoxedDrink017(LibraryObject):
    name = "boxed_drink_017"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/BoxedDrink017/BoxedDrink017.usd")
    object_type = ObjectType.RIGID

@register_asset
class BoxedFood014(LibraryObject):
    name = "boxed_food_014"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/BoxedFood014/BoxedFood014.usd")
    object_type = ObjectType.RIGID

@register_asset
class BoxedFood017(LibraryObject):
    name = "boxed_food_017"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/BoxedFood017/BoxedFood017.usd")
    object_type = ObjectType.RIGID

@register_asset
class Bread018(LibraryObject):
    name = "bread_018"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Bread018/Bread018.usd")
    object_type = ObjectType.RIGID

@register_asset
class Bread028(LibraryObject):
    name = "bread_028"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Bread028/Bread028.usd")
    object_type = ObjectType.RIGID

@register_asset
class Bread030(LibraryObject):
    name = "bread_030"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Bread030/Bread030.usd")
    object_type = ObjectType.RIGID

@register_asset
class Bread032(LibraryObject):
    name = "bread_032"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Bread032/Bread032.usd")
    object_type = ObjectType.RIGID

@register_asset
class Broccoli001(LibraryObject):
    name = "broccoli_001"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Broccoli001/Broccoli001.usd")
    object_type = ObjectType.RIGID

@register_asset
class Broccoli002(LibraryObject):
    name = "broccoli_002"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Broccoli002/Broccoli002.usd")
    object_type = ObjectType.RIGID

@register_asset
class Broccoli003(LibraryObject):
    name = "broccoli_003"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Broccoli003/Broccoli003.usd")
    object_type = ObjectType.RIGID

@register_asset
class Broccoli004(LibraryObject):
    name = "broccoli_004"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Broccoli004/Broccoli004.usd")
    object_type = ObjectType.RIGID

@register_asset
class Broccoli005(LibraryObject):
    name = "broccoli_005"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Broccoli005/Broccoli005.usd")
    object_type = ObjectType.RIGID

@register_asset
class Broccoli006(LibraryObject):
    name = "broccoli_006"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Broccoli006/Broccoli006.usd")
    object_type = ObjectType.RIGID

@register_asset
class Broccoli007(LibraryObject):
    name = "broccoli_007"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Broccoli007/Broccoli007.usd")
    object_type = ObjectType.RIGID

@register_asset
class Broccoli008(LibraryObject):
    name = "broccoli_008"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Broccoli008/Broccoli008.usd")
    object_type = ObjectType.RIGID

@register_asset
class Broccoli009(LibraryObject):
    name = "broccoli_009"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Broccoli009/Broccoli009.usd")
    object_type = ObjectType.RIGID

@register_asset
class Broccoli010(LibraryObject):
    name = "broccoli_010"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Broccoli010/Broccoli010.usd")
    object_type = ObjectType.RIGID

@register_asset
class Broccoli011(LibraryObject):
    name = "broccoli_011"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Broccoli011/Broccoli011.usd")
    object_type = ObjectType.RIGID

@register_asset
class Broccoli012(LibraryObject):
    name = "broccoli_012"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Broccoli012/Broccoli012.usd")
    object_type = ObjectType.RIGID

@register_asset
class Broccoli013(LibraryObject):
    name = "broccoli_013"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Broccoli013/Broccoli013.usd")
    object_type = ObjectType.RIGID

@register_asset
class Broccoli014(LibraryObject):
    name = "broccoli_014"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Broccoli014/Broccoli014.usd")
    object_type = ObjectType.RIGID

@register_asset
class Broccoli015(LibraryObject):
    name = "broccoli_015"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Broccoli015/Broccoli015.usd")
    object_type = ObjectType.RIGID

@register_asset
class Broccoli016(LibraryObject):
    name = "broccoli_016"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Broccoli016/Broccoli016.usd")
    object_type = ObjectType.RIGID

@register_asset
class Broccoli017(LibraryObject):
    name = "broccoli_017"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Broccoli017/Broccoli017.usd")
    object_type = ObjectType.RIGID

@register_asset
class Broccoli018(LibraryObject):
    name = "broccoli_018"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Broccoli018/Broccoli018.usd")
    object_type = ObjectType.RIGID

@register_asset
class BrusselSprout005(LibraryObject):
    name = "brussel_sprout_005"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/BrusselSprout005/BrusselSprout005.usd")
    object_type = ObjectType.RIGID

@register_asset
class BrusselSprout012(LibraryObject):
    name = "brussel_sprout_012"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/BrusselSprout012/BrusselSprout012.usd")
    object_type = ObjectType.RIGID

@register_asset
class BuildingBlock003(LibraryObject):
    name = "building_block_003"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/BuildingBlock003/BuildingBlock003.usd")
    object_type = ObjectType.RIGID

@register_asset
class Burrito001(LibraryObject):
    name = "burrito_001"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Burrito001/Burrito001.usd")
    object_type = ObjectType.RIGID

@register_asset
class Burrito006(LibraryObject):
    name = "burrito_006"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Burrito006/Burrito006.usd")
    object_type = ObjectType.RIGID

@register_asset
class Butter001(LibraryObject):
    name = "butter_001"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Butter001/Butter001.usd")
    object_type = ObjectType.RIGID

@register_asset
class ButterStick005(LibraryObject):
    name = "butter_stick_005"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/ButterStick005/ButterStick005.usd")
    object_type = ObjectType.RIGID

@register_asset
class ButterStick012(LibraryObject):
    name = "butter_stick_012"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/ButterStick012/ButterStick012.usd")
    object_type = ObjectType.RIGID

@register_asset
class Cabbage010(LibraryObject):
    name = "cabbage_010"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Cabbage010/Cabbage010.usd")
    object_type = ObjectType.RIGID

@register_asset
class Cabbage012(LibraryObject):
    name = "cabbage_012"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Cabbage012/Cabbage012.usd")
    object_type = ObjectType.RIGID

@register_asset
class Cake003(LibraryObject):
    name = "cake_003"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Cake003/Cake003.usd")
    object_type = ObjectType.RIGID

@register_asset
class Cake015(LibraryObject):
    name = "cake_015"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Cake015/Cake015.usd")
    object_type = ObjectType.RIGID

@register_asset
class CanOpener001(LibraryObject):
    name = "can_opener_001"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/CanOpener001/CanOpener001.usd")
    object_type = ObjectType.RIGID

@register_asset
class CanOpener003(LibraryObject):
    name = "can_opener_003"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/CanOpener003/CanOpener003.usd")
    object_type = ObjectType.RIGID

@register_asset
class Candle017(LibraryObject):
    name = "candle_017"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Candle017/Candle017.usd")
    object_type = ObjectType.RIGID

@register_asset
class Candle027(LibraryObject):
    name = "candle_027"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Candle027/Candle027.usd")
    object_type = ObjectType.RIGID

@register_asset
class Candle030(LibraryObject):
    name = "candle_030"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Candle030/Candle030.usd")
    object_type = ObjectType.RIGID

@register_asset
class Candle044(LibraryObject):
    name = "candle_044"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Candle044/Candle044.usd")
    object_type = ObjectType.RIGID

@register_asset
class Candy003(LibraryObject):
    name = "candy_003"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Candy003/Candy003.usd")
    object_type = ObjectType.RIGID

@register_asset
class Candy004(LibraryObject):
    name = "candy_004"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Candy004/Candy004.usd")
    object_type = ObjectType.RIGID

@register_asset
class CannedFood003(LibraryObject):
    name = "canned_food_003"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/CannedFood003/CannedFood003.usd")
    object_type = ObjectType.RIGID

@register_asset
class CannedFood013(LibraryObject):
    name = "canned_food_013"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/CannedFood013/CannedFood013.usd")
    object_type = ObjectType.RIGID

@register_asset
class CannedFood042(LibraryObject):
    name = "canned_food_042"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/CannedFood042/CannedFood042.usd")
    object_type = ObjectType.RIGID

@register_asset
class CannedFood043(LibraryObject):
    name = "canned_food_043"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/CannedFood043/CannedFood043.usd")
    object_type = ObjectType.RIGID

@register_asset
class CanolaOil004(LibraryObject):
    name = "canola_oil_004"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/CanolaOil004/CanolaOil004.usd")
    object_type = ObjectType.RIGID

@register_asset
class CanolaOil013(LibraryObject):
    name = "canola_oil_013"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/CanolaOil013/CanolaOil013.usd")
    object_type = ObjectType.RIGID

@register_asset
class Cantaloupe004(LibraryObject):
    name = "cantaloupe_004"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Cantaloupe004/Cantaloupe004.usd")
    object_type = ObjectType.RIGID

@register_asset
class Cantaloupe009(LibraryObject):
    name = "cantaloupe_009"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Cantaloupe009/Cantaloupe009.usd")
    object_type = ObjectType.RIGID

@register_asset
class Carrot003(LibraryObject):
    name = "carrot_003"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Carrot003/Carrot003.usd")
    object_type = ObjectType.RIGID

@register_asset
class Carrot014(LibraryObject):
    name = "carrot_014"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Carrot014/Carrot014.usd")
    object_type = ObjectType.RIGID

@register_asset
class Cauliflower007(LibraryObject):
    name = "cauliflower_007"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Cauliflower007/Cauliflower007.usd")
    object_type = ObjectType.RIGID

@register_asset
class Cauliflower008(LibraryObject):
    name = "cauliflower_008"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Cauliflower008/Cauliflower008.usd")
    object_type = ObjectType.RIGID

@register_asset
class Celery004(LibraryObject):
    name = "celery_004"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Celery004/Celery004.usd")
    object_type = ObjectType.RIGID

@register_asset
class Celery009(LibraryObject):
    name = "celery_009"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Celery009/Celery009.usd")
    object_type = ObjectType.RIGID

@register_asset
class Cereal025(LibraryObject):
    name = "cereal_025"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Cereal025/Cereal025.usd")
    object_type = ObjectType.RIGID

@register_asset
class Cereal031(LibraryObject):
    name = "cereal_031"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Cereal031/Cereal031.usd")
    object_type = ObjectType.RIGID

@register_asset
class Cheese009(LibraryObject):
    name = "cheese_009"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Cheese009/Cheese009.usd")
    object_type = ObjectType.RIGID

@register_asset
class Cheese017(LibraryObject):
    name = "cheese_017"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Cheese017/Cheese017.usd")
    object_type = ObjectType.RIGID

@register_asset
class CheeseGrater013(LibraryObject):
    name = "cheese_grater_013"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/CheeseGrater013/CheeseGrater013.usd")
    object_type = ObjectType.RIGID

@register_asset
class CheeseGrater020(LibraryObject):
    name = "cheese_grater_020"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/CheeseGrater020/CheeseGrater020.usd")
    object_type = ObjectType.RIGID

@register_asset
class Cherry001(LibraryObject):
    name = "cherry_001"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Cherry001/Cherry001.usd")
    object_type = ObjectType.RIGID

@register_asset
class Cherry009(LibraryObject):
    name = "cherry_009"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Cherry009/Cherry009.usd")
    object_type = ObjectType.RIGID

@register_asset
class ChickenBreast003(LibraryObject):
    name = "chicken_breast_003"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/ChickenBreast003/ChickenBreast003.usd")
    object_type = ObjectType.RIGID

@register_asset
class ChickenBreast011(LibraryObject):
    name = "chicken_breast_011"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/ChickenBreast011/ChickenBreast011.usd")
    object_type = ObjectType.RIGID

@register_asset
class ChiliPepper007(LibraryObject):
    name = "chili_pepper_007"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/ChiliPepper007/ChiliPepper007.usd")
    object_type = ObjectType.RIGID

@register_asset
class Chip023(LibraryObject):
    name = "chip_023"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Chip023/Chip023.usd")
    object_type = ObjectType.RIGID

@register_asset
class Chocolate015(LibraryObject):
    name = "chocolate_015"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Chocolate015/Chocolate015.usd")
    object_type = ObjectType.RIGID

@register_asset
class Chocolate016(LibraryObject):
    name = "chocolate_016"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Chocolate016/Chocolate016.usd")
    object_type = ObjectType.RIGID

@register_asset
class ChocolatePudding001(LibraryObject):
    name = "chocolate_pudding_001"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/ChocolatePudding001/ChocolatePudding001.usd")
    object_type = ObjectType.RIGID

@register_asset
class Coconut002(LibraryObject):
    name = "coconut_002"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Coconut002/Coconut002.usd")
    object_type = ObjectType.RIGID

@register_asset
class Coconut014(LibraryObject):
    name = "coconut_014"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Coconut014/Coconut014.usd")
    object_type = ObjectType.RIGID

@register_asset
class CoffeeCup010(LibraryObject):
    name = "coffee_cup_010"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/CoffeeCup010/CoffeeCup010.usd")
    object_type = ObjectType.RIGID

@register_asset
class CoffeeCup025(LibraryObject):
    name = "coffee_cup_025"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/CoffeeCup025/CoffeeCup025.usd")
    object_type = ObjectType.RIGID

@register_asset
class Condiment006(LibraryObject):
    name = "condiment_006"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Condiment006/Condiment006.usd")
    object_type = ObjectType.RIGID

@register_asset
class Condiment013(LibraryObject):
    name = "condiment_013"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Condiment013/Condiment013.usd")
    object_type = ObjectType.RIGID

@register_asset
class Cookies002(LibraryObject):
    name = "cookies_002"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Cookies002/Cookies002.usd")
    object_type = ObjectType.RIGID

@register_asset
class Corn006(LibraryObject):
    name = "corn_006"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Corn006/Corn006.usd")
    object_type = ObjectType.RIGID

@register_asset
class Corn007(LibraryObject):
    name = "corn_007"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Corn007/Corn007.usd")
    object_type = ObjectType.RIGID

@register_asset
class CreamCheeseStick013(LibraryObject):
    name = "cream_cheese_stick_013"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/CreamCheeseStick013/CreamCheeseStick013.usd")
    object_type = ObjectType.RIGID

@register_asset
class Croissant011(LibraryObject):
    name = "croissant_011"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Croissant011/Croissant011.usd")
    object_type = ObjectType.RIGID

@register_asset
class Croissant016(LibraryObject):
    name = "croissant_016"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Croissant016/Croissant016.usd")
    object_type = ObjectType.RIGID

@register_asset
class Cucumber004(LibraryObject):
    name = "cucumber_004"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Cucumber004/Cucumber004.usd")
    object_type = ObjectType.RIGID

@register_asset
class Cucumber013(LibraryObject):
    name = "cucumber_013"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Cucumber013/Cucumber013.usd")
    object_type = ObjectType.RIGID

@register_asset
class Cup012(LibraryObject):
    name = "cup_012"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Cup012/Cup012.usd")
    object_type = ObjectType.RIGID

@register_asset
class Cup044(LibraryObject):
    name = "cup_044"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Cup044/Cup044.usd")
    object_type = ObjectType.RIGID

@register_asset
class Cupcake005(LibraryObject):
    name = "cupcake_005"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Cupcake005/Cupcake005.usd")
    object_type = ObjectType.RIGID

@register_asset
class Cupcake027(LibraryObject):
    name = "cupcake_027"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Cupcake027/Cupcake027.usd")
    object_type = ObjectType.RIGID

@register_asset
class CuttingBoard009(LibraryObject):
    name = "cutting_board_009"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/CuttingBoard009/CuttingBoard009.usd")
    object_type = ObjectType.RIGID

@register_asset
class CuttingBoard031(LibraryObject):
    name = "cutting_board_031"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/CuttingBoard031/CuttingBoard031.usd")
    object_type = ObjectType.RIGID

@register_asset
class DeskCaddy001(LibraryObject):
    name = "desk_caddy_001"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/DeskCaddy001/DeskCaddy001.usd")
    object_type = ObjectType.RIGID

@register_asset
class Donut021(LibraryObject):
    name = "donut_021"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Donut021/Donut021.usd")
    object_type = ObjectType.RIGID

@register_asset
class Donut022(LibraryObject):
    name = "donut_022"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Donut022/Donut022.usd")
    object_type = ObjectType.RIGID

@register_asset
class Dumpling018(LibraryObject):
    name = "dumpling_018"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Dumpling018/Dumpling018.usd")
    object_type = ObjectType.RIGID

@register_asset
class Dumpling022(LibraryObject):
    name = "dumpling_022"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Dumpling022/Dumpling022.usd")
    object_type = ObjectType.RIGID

@register_asset
class Egg002(LibraryObject):
    name = "egg_002"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Egg002/Egg002.usd")
    object_type = ObjectType.RIGID

@register_asset
class Egg024(LibraryObject):
    name = "egg_024"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Egg024/Egg024.usd")
    object_type = ObjectType.RIGID

@register_asset
class Eggplant001(LibraryObject):
    name = "eggplant_001"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Eggplant001/Eggplant001.usd")
    object_type = ObjectType.RIGID

@register_asset
class Eggplant009(LibraryObject):
    name = "eggplant_009"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Eggplant009/Eggplant009.usd")
    object_type = ObjectType.RIGID

@register_asset
class Fish016(LibraryObject):
    name = "fish_016"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Fish016/Fish016.usd")
    object_type = ObjectType.RIGID

@register_asset
class Fork010(LibraryObject):
    name = "fork_010"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Fork010/Fork010.usd")
    object_type = ObjectType.RIGID

@register_asset
class Fork016(LibraryObject):
    name = "fork_016"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Fork016/Fork016.usd")
    object_type = ObjectType.RIGID

@register_asset
class Garlic001(LibraryObject):
    name = "garlic_001"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Garlic001/Garlic001.usd")
    object_type = ObjectType.RIGID

@register_asset
class Garlic012(LibraryObject):
    name = "garlic_012"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Garlic012/Garlic012.usd")
    object_type = ObjectType.RIGID

@register_asset
class Ginger003(LibraryObject):
    name = "ginger_003"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Ginger003/Ginger003.usd")
    object_type = ObjectType.RIGID

@register_asset
class Ginger005(LibraryObject):
    name = "ginger_005"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Ginger005/Ginger005.usd")
    object_type = ObjectType.RIGID

@register_asset
class Grape002(LibraryObject):
    name = "grape_002"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Grape002/Grape002.usd")
    object_type = ObjectType.RIGID

@register_asset
class Grape010(LibraryObject):
    name = "grape_010"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Grape010/Grape010.usd")
    object_type = ObjectType.RIGID

@register_asset
class Ham005(LibraryObject):
    name = "ham_005"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Ham005/Ham005.usd")
    object_type = ObjectType.RIGID

@register_asset
class Ham009(LibraryObject):
    name = "ham_009"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Ham009/Ham009.usd")
    object_type = ObjectType.RIGID

@register_asset
class Hamburger014(LibraryObject):
    name = "hamburger_014"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Hamburger014/Hamburger014.usd")
    object_type = ObjectType.RIGID

@register_asset
class Hamburger015(LibraryObject):
    name = "hamburger_015"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Hamburger015/Hamburger015.usd")
    object_type = ObjectType.RIGID

@register_asset
class HoneyBottle016(LibraryObject):
    name = "honey_bottle_016"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/HoneyBottle016/HoneyBottle016.usd")
    object_type = ObjectType.RIGID

@register_asset
class HoneyBottle018(LibraryObject):
    name = "honey_bottle_018"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/HoneyBottle018/HoneyBottle018.usd")
    object_type = ObjectType.RIGID

@register_asset
class HotDog003(LibraryObject):
    name = "hot_dog_003"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/HotDog003/HotDog003.usd")
    object_type = ObjectType.RIGID

@register_asset
class HotDog005(LibraryObject):
    name = "hot_dog_005"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/HotDog005/HotDog005.usd")
    object_type = ObjectType.RIGID

@register_asset
class IceCream002(LibraryObject):
    name = "ice_cream_002"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/IceCream002/IceCream002.usd")
    object_type = ObjectType.RIGID

@register_asset
class IceCream009(LibraryObject):
    name = "ice_cream_009"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/IceCream009/IceCream009.usd")
    object_type = ObjectType.RIGID

@register_asset
class IceCubeTray010(LibraryObject):
    name = "ice_cube_tray_010"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/IceCubeTray010/IceCubeTray010.usd")
    object_type = ObjectType.RIGID

@register_asset
class IceCubeTray011(LibraryObject):
    name = "ice_cube_tray_011"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/IceCubeTray011/IceCubeTray011.usd")
    object_type = ObjectType.RIGID

@register_asset
class Jam010(LibraryObject):
    name = "jam_010"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Jam010/Jam010.usd")
    object_type = ObjectType.RIGID

@register_asset
class JelloCup001(LibraryObject):
    name = "jello_cup_001"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/JelloCup001/JelloCup001.usd")
    object_type = ObjectType.RIGID

@register_asset
class JelloCup004(LibraryObject):
    name = "jello_cup_004"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/JelloCup004/JelloCup004.usd")
    object_type = ObjectType.RIGID

@register_asset
class Jug005(LibraryObject):
    name = "jug_005"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Jug005/Jug005.usd")
    object_type = ObjectType.RIGID

@register_asset
class Jug008(LibraryObject):
    name = "jug_008"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Jug008/Jug008.usd")
    object_type = ObjectType.RIGID

@register_asset
class Jug015(LibraryObject):
    name = "jug_015"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Jug015/Jug015.usd")
    object_type = ObjectType.RIGID

@register_asset
class Kebabs010(LibraryObject):
    name = "kebabs_010"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Kebabs010/Kebabs010.usd")
    object_type = ObjectType.RIGID

@register_asset
class Kebabs011(LibraryObject):
    name = "kebabs_011"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Kebabs011/Kebabs011.usd")
    object_type = ObjectType.RIGID

@register_asset
class Ketchup007(LibraryObject):
    name = "ketchup_007"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Ketchup007/Ketchup007.usd")
    object_type = ObjectType.RIGID

@register_asset
class Ketchup023(LibraryObject):
    name = "ketchup_023"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Ketchup023/Ketchup023.usd")
    object_type = ObjectType.RIGID

@register_asset
class Kettle047(LibraryObject):
    name = "kettle_047"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Kettle047/Kettle047.usd")
    object_type = ObjectType.RIGID

@register_asset
class Kettle052(LibraryObject):
    name = "kettle_052"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Kettle052/Kettle052.usd")
    object_type = ObjectType.RIGID

@register_asset
class Kettle062(LibraryObject):
    name = "kettle_062"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Kettle062/Kettle062.usd")
    object_type = ObjectType.RIGID

@register_asset
class Kiwi013(LibraryObject):
    name = "kiwi_013"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Kiwi013/Kiwi013.usd")
    object_type = ObjectType.RIGID

@register_asset
class Kiwi018(LibraryObject):
    name = "kiwi_018"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Kiwi018/Kiwi018.usd")
    object_type = ObjectType.RIGID

@register_asset
class Knife025(LibraryObject):
    name = "knife_025"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Knife025/Knife025.usd")
    object_type = ObjectType.RIGID

@register_asset
class Knife077(LibraryObject):
    name = "knife_077"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Knife077/Knife077.usd")
    object_type = ObjectType.RIGID

@register_asset
class KnifeBlock006(LibraryObject):
    name = "knife_block_006"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/KnifeBlock006/KnifeBlock006.usd")
    object_type = ObjectType.RIGID

@register_asset
class KnifeBlock010(LibraryObject):
    name = "knife_block_010"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/KnifeBlock010/KnifeBlock010.usd")
    object_type = ObjectType.RIGID

@register_asset
class Ladle001(LibraryObject):
    name = "ladle_001"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Ladle001/Ladle001.usd")
    object_type = ObjectType.RIGID

@register_asset
class LambChop006(LibraryObject):
    name = "lamb_chop_006"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/LambChop006/LambChop006.usd")
    object_type = ObjectType.RIGID

@register_asset
class LambChop007(LibraryObject):
    name = "lamb_chop_007"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/LambChop007/LambChop007.usd")
    object_type = ObjectType.RIGID

@register_asset
class Lemon003(LibraryObject):
    name = "lemon_003"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Lemon003/Lemon003.usd")
    object_type = ObjectType.RIGID

@register_asset
class Lemon035(LibraryObject):
    name = "lemon_035"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Lemon035/Lemon035.usd")
    object_type = ObjectType.RIGID

@register_asset
class Lemonade002(LibraryObject):
    name = "lemonade_002"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Lemonade002/Lemonade002.usd")
    object_type = ObjectType.RIGID

@register_asset
class Lettuce015(LibraryObject):
    name = "lettuce_015"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Lettuce015/Lettuce015.usd")
    object_type = ObjectType.RIGID

@register_asset
class Lettuce017(LibraryObject):
    name = "lettuce_017"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Lettuce017/Lettuce017.usd")
    object_type = ObjectType.RIGID

@register_asset
class Lime003(LibraryObject):
    name = "lime_003"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Lime003/Lime003.usd")
    object_type = ObjectType.RIGID

@register_asset
class Lime005(LibraryObject):
    name = "lime_005"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Lime005/Lime005.usd")
    object_type = ObjectType.RIGID

@register_asset
class Lobster003(LibraryObject):
    name = "lobster_003"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Lobster003/Lobster003.usd")
    object_type = ObjectType.RIGID

@register_asset
class Lobster004(LibraryObject):
    name = "lobster_004"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Lobster004/Lobster004.usd")
    object_type = ObjectType.RIGID

@register_asset
class Lollipop005(LibraryObject):
    name = "lollipop_005"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Lollipop005/Lollipop005.usd")
    object_type = ObjectType.RIGID

@register_asset
class Lollipop006(LibraryObject):
    name = "lollipop_006"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Lollipop006/Lollipop006.usd")
    object_type = ObjectType.RIGID

@register_asset
class MacaroniAndCheese001(LibraryObject):
    name = "macaroni_and_cheese_001"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/MacaroniAndCheese001/MacaroniAndCheese001.usd")
    object_type = ObjectType.RIGID

@register_asset
class Mango005(LibraryObject):
    name = "mango_005"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Mango005/Mango005.usd")
    object_type = ObjectType.RIGID

@register_asset
class Mango022(LibraryObject):
    name = "mango_022"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Mango022/Mango022.usd")
    object_type = ObjectType.RIGID

@register_asset
class Milk012(LibraryObject):
    name = "milk_012"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Milk012/Milk012.usd")
    object_type = ObjectType.RIGID

@register_asset
class Milk016(LibraryObject):
    name = "milk_016"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Milk016/Milk016.usd")
    object_type = ObjectType.RIGID

@register_asset
class MilkDrink009(LibraryObject):
    name = "milk_drink_009"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/MilkDrink009/MilkDrink009.usd")
    object_type = ObjectType.RIGID

@register_asset
class Mug020(LibraryObject):
    name = "mug_020"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Mug020/Mug020.usd")
    object_type = ObjectType.RIGID

@register_asset
class Mug022(LibraryObject):
    name = "mug_022"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Mug022/Mug022.usd")
    object_type = ObjectType.RIGID

@register_asset
class Mushroom002(LibraryObject):
    name = "mushroom_002"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Mushroom002/Mushroom002.usd")
    object_type = ObjectType.RIGID

@register_asset
class Mushroom022(LibraryObject):
    name = "mushroom_022"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Mushroom022/Mushroom022.usd")
    object_type = ObjectType.RIGID

@register_asset
class OliveOilBottle006(LibraryObject):
    name = "olive_oil_bottle_006"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/OliveOilBottle006/OliveOilBottle006.usd")
    object_type = ObjectType.RIGID

@register_asset
class OliveOilBottle009(LibraryObject):
    name = "olive_oil_bottle_009"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/OliveOilBottle009/OliveOilBottle009.usd")
    object_type = ObjectType.RIGID

@register_asset
class Onion004(LibraryObject):
    name = "onion_004"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Onion004/Onion004.usd")
    object_type = ObjectType.RIGID

@register_asset
class Onion017(LibraryObject):
    name = "onion_017"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Onion017/Onion017.usd")
    object_type = ObjectType.RIGID

@register_asset
class Orange017(LibraryObject):
    name = "orange_017"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Orange017/Orange017.usd")
    object_type = ObjectType.RIGID

@register_asset
class OrangeJuice001(LibraryObject):
    name = "orange_juice_001"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/OrangeJuice001/OrangeJuice001.usd")
    object_type = ObjectType.RIGID

@register_asset
class Pan006(LibraryObject):
    name = "pan_006"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Pan006/Pan006.usd")
    object_type = ObjectType.RIGID

@register_asset
class Pan022(LibraryObject):
    name = "pan_022"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Pan022/Pan022.usd")
    object_type = ObjectType.RIGID

@register_asset
class PaperTowelHolder008(LibraryObject):
    name = "paper_towel_holder_008"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/PaperTowelHolder008/PaperTowelHolder008.usd")
    object_type = ObjectType.RIGID

@register_asset
class PaperTowelHolder010(LibraryObject):
    name = "paper_towel_holder_010"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/PaperTowelHolder010/PaperTowelHolder010.usd")
    object_type = ObjectType.RIGID

@register_asset
class Paprika006(LibraryObject):
    name = "paprika_006"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Paprika006/Paprika006.usd")
    object_type = ObjectType.RIGID

@register_asset
class Peach003(LibraryObject):
    name = "peach_003"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Peach003/Peach003.usd")
    object_type = ObjectType.RIGID

@register_asset
class Peach010(LibraryObject):
    name = "peach_010"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Peach010/Peach010.usd")
    object_type = ObjectType.RIGID

@register_asset
class PeanutButterJar010(LibraryObject):
    name = "peanut_butter_jar_010"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/PeanutButterJar010/PeanutButterJar010.usd")
    object_type = ObjectType.RIGID

@register_asset
class PeanutButterJar013(LibraryObject):
    name = "peanut_butter_jar_013"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/PeanutButterJar013/PeanutButterJar013.usd")
    object_type = ObjectType.RIGID

@register_asset
class Pear006(LibraryObject):
    name = "pear_006"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Pear006/Pear006.usd")
    object_type = ObjectType.RIGID

@register_asset
class Pear029(LibraryObject):
    name = "pear_029"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Pear029/Pear029.usd")
    object_type = ObjectType.RIGID

@register_asset
class Phone017(LibraryObject):
    name = "phone_017"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Phone017/Phone017.usd")
    object_type = ObjectType.RIGID

@register_asset
class PhotoFrame011(LibraryObject):
    name = "photo_frame_011"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/PhotoFrame011/PhotoFrame011.usd")
    object_type = ObjectType.RIGID

@register_asset
class Pickle001(LibraryObject):
    name = "pickle_001"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Pickle001/Pickle001.usd")
    object_type = ObjectType.RIGID

@register_asset
class Pickle004(LibraryObject):
    name = "pickle_004"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Pickle004/Pickle004.usd")
    object_type = ObjectType.RIGID

@register_asset
class Pineapple002(LibraryObject):
    name = "pineapple_002"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Pineapple002/Pineapple002.usd")
    object_type = ObjectType.RIGID

@register_asset
class Pineapple004(LibraryObject):
    name = "pineapple_004"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Pineapple004/Pineapple004.usd")
    object_type = ObjectType.RIGID

@register_asset
class Pitcher028(LibraryObject):
    name = "pitcher_028"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Pitcher028/Pitcher028.usd")
    object_type = ObjectType.RIGID

@register_asset
class Pitcher029(LibraryObject):
    name = "pitcher_029"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Pitcher029/Pitcher029.usd")
    object_type = ObjectType.RIGID

@register_asset
class PizzaCutter011(LibraryObject):
    name = "pizza_cutter_011"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/PizzaCutter011/PizzaCutter011.usd")
    object_type = ObjectType.RIGID

@register_asset
class PizzaCutter014(LibraryObject):
    name = "pizza_cutter_014"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/PizzaCutter014/PizzaCutter014.usd")
    object_type = ObjectType.RIGID

@register_asset
class Plant008(LibraryObject):
    name = "plant_008"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Plant008/Plant008.usd")
    object_type = ObjectType.RIGID

@register_asset
class Plant041(LibraryObject):
    name = "plant_041"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Plant041/Plant041.usd")
    scale = (0.467, 0.467, 0.467)
    object_type = ObjectType.RIGID

@register_asset
class Plate031(LibraryObject):
    name = "plate_031"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Plate031/Plate031.usd")
    object_type = ObjectType.RIGID

@register_asset
class Plate051(LibraryObject):
    name = "plate_051"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Plate051/Plate051.usd")
    object_type = ObjectType.RIGID

@register_asset
class Pomegranate004(LibraryObject):
    name = "pomegranate_004"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Pomegranate004/Pomegranate004.usd")
    object_type = ObjectType.RIGID

@register_asset
class Pomegranate007(LibraryObject):
    name = "pomegranate_007"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Pomegranate007/Pomegranate007.usd")
    object_type = ObjectType.RIGID

@register_asset
class Popcorn001(LibraryObject):
    name = "popcorn_001"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Popcorn001/Popcorn001.usd")
    object_type = ObjectType.RIGID

@register_asset
class PorkChop002(LibraryObject):
    name = "pork_chop_002"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/PorkChop002/PorkChop002.usd")
    object_type = ObjectType.RIGID

@register_asset
class PorkChop003(LibraryObject):
    name = "pork_chop_003"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/PorkChop003/PorkChop003.usd")
    object_type = ObjectType.RIGID

@register_asset
class PorkLoin003(LibraryObject):
    name = "pork_loin_003"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/PorkLoin003/PorkLoin003.usd")
    object_type = ObjectType.RIGID

@register_asset
class PorkLoin006(LibraryObject):
    name = "pork_loin_006"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/PorkLoin006/PorkLoin006.usd")
    object_type = ObjectType.RIGID

@register_asset
class Pot052(LibraryObject):
    name = "pot_052"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Pot052/Pot052.usd")
    object_type = ObjectType.RIGID

@register_asset
class Pot090(LibraryObject):
    name = "pot_090"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Pot090/Pot090.usd")
    object_type = ObjectType.RIGID

@register_asset
class Potato004(LibraryObject):
    name = "potato_004"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Potato004/Potato004.usd")
    object_type = ObjectType.RIGID

@register_asset
class Potato026(LibraryObject):
    name = "potato_026"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Potato026/Potato026.usd")
    object_type = ObjectType.RIGID

@register_asset
class Radish002(LibraryObject):
    name = "radish_002"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Radish002/Radish002.usd")
    object_type = ObjectType.RIGID

@register_asset
class Radish005(LibraryObject):
    name = "radish_005"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Radish005/Radish005.usd")
    object_type = ObjectType.RIGID

@register_asset
class Raspberry007(LibraryObject):
    name = "raspberry_007"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Raspberry007/Raspberry007.usd")
    object_type = ObjectType.RIGID

@register_asset
class Raspberry008(LibraryObject):
    name = "raspberry_008"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Raspberry008/Raspberry008.usd")
    object_type = ObjectType.RIGID

@register_asset
class RollingPin009(LibraryObject):
    name = "rolling_pin_009"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/RollingPin009/RollingPin009.usd")
    object_type = ObjectType.RIGID

@register_asset
class RollingPin012(LibraryObject):
    name = "rolling_pin_012"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/RollingPin012/RollingPin012.usd")
    object_type = ObjectType.RIGID

@register_asset
class Salami003(LibraryObject):
    name = "salami_003"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Salami003/Salami003.usd")
    object_type = ObjectType.RIGID

@register_asset
class Salami005(LibraryObject):
    name = "salami_005"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Salami005/Salami005.usd")
    object_type = ObjectType.RIGID

@register_asset
class Salsa010(LibraryObject):
    name = "salsa_010"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Salsa010/Salsa010.usd")
    object_type = ObjectType.RIGID

@register_asset
class Salsa013(LibraryObject):
    name = "salsa_013"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Salsa013/Salsa013.usd")
    object_type = ObjectType.RIGID

@register_asset
class SandwichBread010(LibraryObject):
    name = "sandwich_bread_010"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/SandwichBread010/SandwichBread010.usd")
    object_type = ObjectType.RIGID

@register_asset
class SandwichBread012(LibraryObject):
    name = "sandwich_bread_012"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/SandwichBread012/SandwichBread012.usd")
    object_type = ObjectType.RIGID

@register_asset
class Sausage004(LibraryObject):
    name = "sausage_004"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Sausage004/Sausage004.usd")
    object_type = ObjectType.RIGID

@register_asset
class Sausage012(LibraryObject):
    name = "sausage_012"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Sausage012/Sausage012.usd")
    object_type = ObjectType.RIGID

@register_asset
class Scallop006(LibraryObject):
    name = "scallop_006"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Scallop006/Scallop006.usd")
    object_type = ObjectType.RIGID

@register_asset
class Scallop012(LibraryObject):
    name = "scallop_012"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Scallop012/Scallop012.usd")
    object_type = ObjectType.RIGID

@register_asset
class Scissor037(LibraryObject):
    name = "scissor_037"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Scissor037/Scissor037.usd")
    object_type = ObjectType.RIGID

@register_asset
class Scissor038(LibraryObject):
    name = "scissor_038"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Scissor038/Scissor038.usd")
    object_type = ObjectType.RIGID

@register_asset
class Scone008(LibraryObject):
    name = "scone_008"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Scone008/Scone008.usd")
    object_type = ObjectType.RIGID

@register_asset
class Scone011(LibraryObject):
    name = "scone_011"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Scone011/Scone011.usd")
    object_type = ObjectType.RIGID

@register_asset
class Shaker001(LibraryObject):
    name = "shaker_001"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Shaker001/Shaker001.usd")
    object_type = ObjectType.RIGID

@register_asset
class Shaker004(LibraryObject):
    name = "shaker_004"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Shaker004/Shaker004.usd")
    object_type = ObjectType.RIGID

@register_asset
class Shrimp017(LibraryObject):
    name = "shrimp_017"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Shrimp017/Shrimp017.usd")
    object_type = ObjectType.RIGID

@register_asset
class Shrimp021(LibraryObject):
    name = "shrimp_021"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Shrimp021/Shrimp021.usd")
    object_type = ObjectType.RIGID

@register_asset
class Skewer001(LibraryObject):
    name = "skewer_001"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Skewer001/Skewer001.usd")
    object_type = ObjectType.RIGID

@register_asset
class Skewer004(LibraryObject):
    name = "skewer_004"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Skewer004/Skewer004.usd")
    object_type = ObjectType.RIGID

@register_asset
class Soap005(LibraryObject):
    name = "soap_005"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Soap005/Soap005.usd")
    object_type = ObjectType.RIGID

@register_asset
class Soap011(LibraryObject):
    name = "soap_011"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Soap011/Soap011.usd")
    object_type = ObjectType.RIGID

@register_asset
class SoapDispenser018(LibraryObject):
    name = "soap_dispenser_018"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/SoapDispenser018/SoapDispenser018.usd")
    object_type = ObjectType.RIGID

@register_asset
class SoapDispenser027(LibraryObject):
    name = "soap_dispenser_027"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/SoapDispenser027/SoapDispenser027.usd")
    object_type = ObjectType.RIGID

@register_asset
class SpaghettiBox001(LibraryObject):
    name = "spaghetti_box_001"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/SpaghettiBox001/SpaghettiBox001.usd")
    object_type = ObjectType.RIGID

@register_asset
class SpaghettiBox005(LibraryObject):
    name = "spaghetti_box_005"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/SpaghettiBox005/SpaghettiBox005.usd")
    object_type = ObjectType.RIGID

@register_asset
class Spatula017(LibraryObject):
    name = "spatula_017"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Spatula017/Spatula017.usd")
    object_type = ObjectType.RIGID

@register_asset
class Spatula018(LibraryObject):
    name = "spatula_018"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Spatula018/Spatula018.usd")
    object_type = ObjectType.RIGID

@register_asset
class Sponge002(LibraryObject):
    name = "sponge_002"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Sponge002/Sponge002.usd")
    object_type = ObjectType.RIGID

@register_asset
class Sponge009(LibraryObject):
    name = "sponge_009"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Sponge009/Sponge009.usd")
    object_type = ObjectType.RIGID

@register_asset
class Spoon019(LibraryObject):
    name = "spoon_019"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Spoon019/Spoon019.usd")
    object_type = ObjectType.RIGID

@register_asset
class Spoon022(LibraryObject):
    name = "spoon_022"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Spoon022/Spoon022.usd")
    object_type = ObjectType.RIGID

@register_asset
class Spray016(LibraryObject):
    name = "spray_016"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Spray016/Spray016.usd")
    object_type = ObjectType.RIGID

@register_asset
class Spray017(LibraryObject):
    name = "spray_017"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Spray017/Spray017.usd")
    object_type = ObjectType.RIGID

@register_asset
class Squash009(LibraryObject):
    name = "squash_009"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Squash009/Squash009.usd")
    object_type = ObjectType.RIGID

@register_asset
class Squash014(LibraryObject):
    name = "squash_014"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Squash014/Squash014.usd")
    object_type = ObjectType.RIGID

@register_asset
class Steak006(LibraryObject):
    name = "steak_006"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Steak006/Steak006.usd")
    object_type = ObjectType.RIGID

@register_asset
class Steak010(LibraryObject):
    name = "steak_010"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Steak010/Steak010.usd")
    object_type = ObjectType.RIGID

@register_asset
class Strawberry003(LibraryObject):
    name = "strawberry_003"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Strawberry003/Strawberry003.usd")
    object_type = ObjectType.RIGID

@register_asset
class Strawberry010(LibraryObject):
    name = "strawberry_010"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Strawberry010/Strawberry010.usd")
    object_type = ObjectType.RIGID

@register_asset
class Sushi004(LibraryObject):
    name = "sushi_004"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Sushi004/Sushi004.usd")
    object_type = ObjectType.RIGID

@register_asset
class Sushi008(LibraryObject):
    name = "sushi_008"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Sushi008/Sushi008.usd")
    object_type = ObjectType.RIGID

@register_asset
class SweetPotato005(LibraryObject):
    name = "sweet_potato_005"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/SweetPotato005/SweetPotato005.usd")
    object_type = ObjectType.RIGID

@register_asset
class SweetPotato007(LibraryObject):
    name = "sweet_potato_007"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/SweetPotato007/SweetPotato007.usd")
    object_type = ObjectType.RIGID

@register_asset
class SweetPotato014(LibraryObject):
    name = "sweet_potato_014"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/SweetPotato014/SweetPotato014.usd")
    object_type = ObjectType.RIGID

@register_asset
class SyrupBottle013(LibraryObject):
    name = "syrup_bottle_013"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/SyrupBottle013/SyrupBottle013.usd")
    object_type = ObjectType.RIGID

@register_asset
class SyrupBottle017(LibraryObject):
    name = "syrup_bottle_017"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/SyrupBottle017/SyrupBottle017.usd")
    object_type = ObjectType.RIGID

@register_asset
class TableKnife006(LibraryObject):
    name = "table_knife_006"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/TableKnife006/TableKnife006.usd")
    object_type = ObjectType.RIGID

@register_asset
class Tacos001(LibraryObject):
    name = "tacos_001"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Tacos001/Tacos001.usd")
    object_type = ObjectType.RIGID

@register_asset
class Tacos006(LibraryObject):
    name = "tacos_006"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Tacos006/Tacos006.usd")
    object_type = ObjectType.RIGID

@register_asset
class Tangerine014(LibraryObject):
    name = "tangerine_014"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Tangerine014/Tangerine014.usd")
    object_type = ObjectType.RIGID

@register_asset
class Tangerine019(LibraryObject):
    name = "tangerine_019"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Tangerine019/Tangerine019.usd")
    object_type = ObjectType.RIGID

@register_asset
class Teapot009(LibraryObject):
    name = "teapot_009"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Teapot009/Teapot009.usd")
    object_type = ObjectType.RIGID

@register_asset
class Teapot010(LibraryObject):
    name = "teapot_010"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Teapot010/Teapot010.usd")
    object_type = ObjectType.RIGID

@register_asset
class Thermos007(LibraryObject):
    name = "thermos_007"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Thermos007/Thermos007.usd")
    object_type = ObjectType.RIGID

@register_asset
class Tofu005(LibraryObject):
    name = "tofu_005"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Tofu005/Tofu005.usd")
    object_type = ObjectType.RIGID

@register_asset
class Tofu016(LibraryObject):
    name = "tofu_016"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Tofu016/Tofu016.usd")
    object_type = ObjectType.RIGID

@register_asset
class Tomato012(LibraryObject):
    name = "tomato_012"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Tomato012/Tomato012.usd")
    object_type = ObjectType.RIGID

@register_asset
class Tomato017(LibraryObject):
    name = "tomato_017"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Tomato017/Tomato017.usd")
    object_type = ObjectType.RIGID

@register_asset
class Tong016(LibraryObject):
    name = "tong_016"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Tong016/Tong016.usd")
    object_type = ObjectType.RIGID

@register_asset
class Tong017(LibraryObject):
    name = "tong_017"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Tong017/Tong017.usd")
    object_type = ObjectType.RIGID

@register_asset
class Turmeric003(LibraryObject):
    name = "turmeric_003"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Turmeric003/Turmeric003.usd")
    object_type = ObjectType.RIGID

@register_asset
class Turmeric005(LibraryObject):
    name = "turmeric_005"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Turmeric005/Turmeric005.usd")
    object_type = ObjectType.RIGID

@register_asset
class Turmeric006(LibraryObject):
    name = "turmeric_006"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Turmeric006/Turmeric006.usd")
    object_type = ObjectType.RIGID

@register_asset
class Turmeric009(LibraryObject):
    name = "turmeric_009"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Turmeric009/Turmeric009.usd")
    object_type = ObjectType.RIGID

@register_asset
class Turmeric010(LibraryObject):
    name = "turmeric_010"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Turmeric010/Turmeric010.usd")
    object_type = ObjectType.RIGID

@register_asset
class Turmeric012(LibraryObject):
    name = "turmeric_012"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Turmeric012/Turmeric012.usd")
    object_type = ObjectType.RIGID

@register_asset
class Vinegar003(LibraryObject):
    name = "vinegar_003"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Vinegar003/Vinegar003.usd")
    object_type = ObjectType.RIGID

@register_asset
class Vinegar010(LibraryObject):
    name = "vinegar_010"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Vinegar010/Vinegar010.usd")
    object_type = ObjectType.RIGID

@register_asset
class Waffle002(LibraryObject):
    name = "waffle_002"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Waffle002/Waffle002.usd")
    object_type = ObjectType.RIGID

@register_asset
class Waffle011(LibraryObject):
    name = "waffle_011"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Waffle011/Waffle011.usd")
    object_type = ObjectType.RIGID

@register_asset
class Walnut001(LibraryObject):
    name = "walnut_001"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Walnut001/Walnut001.usd")
    object_type = ObjectType.RIGID

@register_asset
class Walnut006(LibraryObject):
    name = "walnut_006"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Walnut006/Walnut006.usd")
    object_type = ObjectType.RIGID

@register_asset
class WaterBottle013(LibraryObject):
    name = "water_bottle_013"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/WaterBottle013/WaterBottle013.usd")
    object_type = ObjectType.RIGID

@register_asset
class WaterBottle023(LibraryObject):
    name = "water_bottle_023"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/WaterBottle023/WaterBottle023.usd")
    object_type = ObjectType.RIGID

@register_asset
class Watermelon007(LibraryObject):
    name = "watermelon_007"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Watermelon007/Watermelon007.usd")
    object_type = ObjectType.RIGID

@register_asset
class Watermelon011(LibraryObject):
    name = "watermelon_011"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Watermelon011/Watermelon011.usd")
    object_type = ObjectType.RIGID

@register_asset
class Whisk014(LibraryObject):
    name = "whisk_014"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Whisk014/Whisk014.usd")
    object_type = ObjectType.RIGID

@register_asset
class Wine013(LibraryObject):
    name = "wine_013"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Wine013/Wine013.usd")
    object_type = ObjectType.RIGID

@register_asset
class Wine021(LibraryObject):
    name = "wine_021"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Wine021/Wine021.usd")
    object_type = ObjectType.RIGID

@register_asset
class Yogurt008(LibraryObject):
    name = "yogurt_008"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Yogurt008/Yogurt008.usd")
    object_type = ObjectType.RIGID

@register_asset
class Zucchini002(LibraryObject):
    name = "zucchini_002"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Zucchini002/Zucchini002.usd")
    object_type = ObjectType.RIGID

@register_asset
class Zucchini006(LibraryObject):
    name = "zucchini_006"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/Zucchini006/Zucchini006.usd")
    object_type = ObjectType.RIGID

@register_asset
class bar_soap_0(LibraryObject):
    name = "bar_soap_0"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/bar_soap_0/bar_soap_0.usd")
    object_type = ObjectType.RIGID

@register_asset
class bar_soap_1(LibraryObject):
    name = "bar_soap_1"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/bar_soap_1/bar_soap_1.usd")
    object_type = ObjectType.RIGID

@register_asset
class bar_soap_3(LibraryObject):
    name = "bar_soap_3"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/bar_soap_3/bar_soap_3.usd")
    object_type = ObjectType.RIGID

@register_asset
class bar_soap_4(LibraryObject):
    name = "bar_soap_4"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/bar_soap_4/bar_soap_4.usd")
    object_type = ObjectType.RIGID

@register_asset
class chips_11(LibraryObject):
    name = "chips_11"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/chips_11/chips_11.usd")
    object_type = ObjectType.RIGID

@register_asset
class chips_5(LibraryObject):
    name = "chips_5"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/chips_5/chips_5.usd")
    object_type = ObjectType.RIGID

@register_asset
class dates_12(LibraryObject):
    name = "dates_12"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/dates_12/dates_12.usd")
    object_type = ObjectType.RIGID

@register_asset
class dates_4(LibraryObject):
    name = "dates_4"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/dates_4/dates_4.usd")
    object_type = ObjectType.RIGID

@register_asset
class scissors_11(LibraryObject):
    name = "scissors_11"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/scissors_11/scissors_11.usd")
    object_type = ObjectType.RIGID

@register_asset
class scissors_9(LibraryObject):
    name = "scissors_9"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/scissors_9/scissors_9.usd")
    object_type = ObjectType.RIGID

@register_asset
class wine_glass_2(LibraryObject):
    name = "wine_glass_2"
    tags = ["object", "lightwheel"]
    usd_path = os.path.expanduser("~/.cache/lightwheel_sdk/object/wine_glass_2/wine_glass_2.usd")
    object_type = ObjectType.RIGID



# Objects from object_library.py not in RoboLab catalog


@register_asset
class AlphabetSoupCanHopeRobolab(LibraryObject):
    name = "alphabet_soup_can_hope_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hope/alphabet_soup_can.usd"
    spawn_cfg_addon={"mass_props": sim_utils.MassPropertiesCfg(mass=0.01)}

@register_asset
class MarkerHot3DRobolab(LibraryObject):
    name = "marker_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/marker.usd"

@register_asset
class MilkCartonHot3DRobolab(LibraryObject):
    name = "milk_carton_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/milk_carton.usd"
    )

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
class SaladDressingBottleHot3DRobolab(LibraryObject):
    name = "salad_dressing_bottle_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/salad_dressing_bottle.usd"

@register_asset
class TomatoSauceCanHot3DRobolab(LibraryObject):
    name = "tomato_sauce_can_hot3d_robolab"
    tags = ["object", "graspable", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/tomato_sauce_can.usd"

@register_asset
class Apple02ObjaverseRobolab(LibraryObject):
    name = "apple_02_objaverse_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/objaverse/apple_02.usd"
    )

@register_asset
class Bagel06ObjaverseRobolab(LibraryObject):
    name = "bagel_06_objaverse_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = (
        f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/objaverse/bagel_06.usd"
    )


@register_asset
class Baguette02ObjaverseRobolab(LibraryObject):
    name = "baguette_02_objaverse_robolab"
    tags = ["object", "graspable", "food", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/objects/objaverse/baguette_02.usd"

@register_asset
class TableMapleRobolab(LibraryObject):
    name = "table_maple_robolab"
    tags = ["background", "fixture", "robolab"]
    usd_path = f"{ISAACLAB_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/srl_robolab_assets/fixtures/table_maple.usd"

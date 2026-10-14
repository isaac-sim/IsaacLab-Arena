# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import isaaclab.sim as sim_utils

from isaaclab_arena.affordances.pressable import Pressable
from isaaclab_arena.assets.background import Background
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose


class FactoryObject(Object):
    """Base class for G1 Factory-local USD objects."""

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


class FactoryBackgroundBase(Background):
    """Base class for G1 Factory-local backgrounds."""

    name: str
    tags: list[str]
    usd_path: str
    initial_pose: Pose | None = None
    object_min_z: float
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __init__(self, **kwargs):
        super().__init__(
            name=self.name,
            tags=self.tags,
            usd_path=self.usd_path,
            initial_pose=self.initial_pose,
            object_min_z=self.object_min_z,
            scale=self.scale,
            **kwargs,
        )


@register_asset
class FactoryBackground(FactoryBackgroundBase):
    name = "factory_room"
    tags = ["background"]
    usd_path = "/datasets/assets/room/room.usd"
    initial_pose = Pose(position_xyz=(7, 0, -0.785))
    object_min_z = -0.2
    scale = (300.0, 300.0, 100.0)


@register_asset
class LocomanipWhiteTable(FactoryObject):
    name = "locomanip_white_table"
    tags = ["object"]
    usd_path = "/datasets/assets/locomanip_assets/Collected_factory_ergo_table_01/factory_ergo_table_01.usd"
    default_prim_path = "{ENV_REGEX_NS}/table"
    default_initial_pose = Pose(position_xyz=(0.9, 0, -0.79), rotation_xyzw=(0.70711, 0, 0, 0.70711))

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None, **kwargs):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose or self.default_initial_pose, **kwargs)


@register_asset
class LocomanipPowerDrill(FactoryObject):
    name = "locomanip_power_drill"
    tags = ["object"]
    usd_path = "/datasets/assets/locomanip_assets/Collected_sm_powerdrill_b01/sm_powerdrill_b01.usd"
    default_prim_path = "{ENV_REGEX_NS}/power_drill"
    default_initial_pose = Pose(position_xyz=(0.9, 0, -0.79), rotation_xyzw=(0.70711, 0, 0, 0.70711))

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None, **kwargs):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose or self.default_initial_pose, **kwargs)


@register_asset
class LocomanipPowerDrillHolderTask(FactoryObject):
    name = "locomanip_power_drill_holder_task"
    tags = ["object"]
    usd_path = "/datasets/assets/locomanip_assets/Collected_sm_powerdrill_b01_holdertask/sm_powerdrill_b01.usd"
    default_prim_path = "{ENV_REGEX_NS}/power_drill"
    default_initial_pose = Pose(position_xyz=(0.9, 0, -0.79), rotation_xyzw=(0.70711, 0, 0, 0.70711))

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None, **kwargs):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose or self.default_initial_pose, **kwargs)


@register_asset
class LocomanipIndustrialShelf(FactoryObject):
    name = "locomanip_industrial_shelf"
    tags = ["object"]
    usd_path = "/datasets/assets/locomanip_assets/Collected_openindustrialsteelshelving_a12/openindustrialsteelshelving_a12.usd"
    default_prim_path = "{ENV_REGEX_NS}/industrial_shelf"
    default_initial_pose = Pose(position_xyz=(0.9, 0, -0.79), rotation_xyzw=(0.70711, 0, 0, 0.70711))

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None, **kwargs):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose or self.default_initial_pose, **kwargs)


@register_asset
class LocomanipCardbox(FactoryObject):
    name = "locomanip_cardbox"
    tags = ["object"]
    usd_path = "/datasets/assets/locomanip_assets/Collected_cardbox_a1_01/cardbox_a1_01.usd"
    default_prim_path = "{ENV_REGEX_NS}/locomanip_cardbox"
    scale = (0.8, 0.8, 0.8)
    default_initial_pose = Pose(position_xyz=(0.9, 0, -0.79), rotation_xyzw=(0.70711, 0, 0, 0.70711))

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None, **kwargs):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose or self.default_initial_pose, **kwargs)


@register_asset
class LocomanipLongbox(FactoryObject):
    name = "locomanip_longbox"
    tags = ["object"]
    usd_path = "/datasets/assets/locomanip_assets/Collected_longbox_a08/longbox_a08.usd"
    default_prim_path = "{ENV_REGEX_NS}/locomanip_longbox"
    default_initial_pose = Pose(position_xyz=(0.9, 0, -0.79), rotation_xyzw=(0.70711, 0, 0, 0.70711))

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None, **kwargs):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose or self.default_initial_pose, **kwargs)


@register_asset
class LocomanipPowerdrillHolder(FactoryObject):
    name = "locomanip_powerdrill_holder"
    tags = ["object"]
    usd_path = "/datasets/assets/locomanip_assets/Collected_powerdrill_holder/powerdrill_holder.usd"
    default_prim_path = "{ENV_REGEX_NS}/locomanip_powerdrill_holder"
    default_initial_pose = Pose(position_xyz=(0.9, 0, -0.79), rotation_xyzw=(0.70711, 0, 0, 0.70711))

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None, **kwargs):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose or self.default_initial_pose, **kwargs)


@register_asset
class LocomanipMobileCart(FactoryObject):
    name = "locomanip_mobile_cart"
    tags = ["object"]
    usd_path = "/datasets/assets/locomanip_assets/Collected_sm_mobileshelvingcart_a01/sm_mobileshelvingcart_a01.usd"
    default_prim_path = "{ENV_REGEX_NS}/locomanip_mobile_cart"
    object_type: ObjectType = ObjectType.ARTICULATION
    default_initial_pose = Pose(position_xyz=(0.9, 0, -0.79), rotation_xyzw=(1.0, 0, 0, 0.0))
    spawn_cfg_addon = {
        "articulation_props": sim_utils.ArticulationRootPropertiesCfg(fix_root_link=False),
    }

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None, **kwargs):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose or self.default_initial_pose, **kwargs)


@register_asset
class LocomanipTargetZone(FactoryObject):
    name = "locomanip_target_zone"
    tags = ["object"]
    usd_path = "/datasets/assets/locomanip_assets/Collected_target_zone/target_zone.usd"
    default_prim_path = "{ENV_REGEX_NS}/locomanip_target_zone"
    default_initial_pose = Pose(position_xyz=(0.9, 0, -0.79), rotation_xyzw=(0.70711, 0, 0, 0.70711))

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None, **kwargs):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose or self.default_initial_pose, **kwargs)


@register_asset
class Button(FactoryObject, Pressable):
    name = "button"
    tags = ["object", "pressable"]
    usd_path = "/datasets/assets/locomanip_assets_collected_usd/Collected_control_box/button.usd"
    default_prim_path = "{ENV_REGEX_NS}/button"
    object_type: ObjectType = ObjectType.ARTICULATION
    default_initial_pose = Pose(position_xyz=(0.9, 0, -0.79), rotation_xyzw=(1, 0, 0, 0.0))
    pressable_joint_name = "button_joint"
    pressedness_threshold = 0.5

    def __init__(
        self,
        instance_name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        **kwargs,
    ):
        super().__init__(
            instance_name=instance_name,
            prim_path=prim_path,
            initial_pose=initial_pose or self.default_initial_pose,
            pressable_joint_name=self.pressable_joint_name,
            pressedness_threshold=self.pressedness_threshold,
            **kwargs,
        )


@register_asset
class ControlBox(FactoryObject):
    name = "control_box"
    tags = ["object"]
    usd_path = "/datasets/assets/locomanip_assets_collected_usd/Collected_control_box/control_box.usd"
    default_prim_path = "{ENV_REGEX_NS}/control_box"
    default_initial_pose = Pose(position_xyz=(0.9, 0, -0.79), rotation_xyzw=(0.70711, 0, 0, 0.70711))

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None, **kwargs):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose or self.default_initial_pose, **kwargs)

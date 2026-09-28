# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Nucleus-hosted assets used by the ported Isaac Cap gear environment."""

from __future__ import annotations

from typing import Any, ClassVar

import isaaclab.sim as sim_utils

from isaaclab_arena.assets.background import Background
from isaaclab_arena.assets.hdr_image_library import EmptyWarehouseHDRRobolab
from isaaclab_arena.assets.nucleus import ARENA_NUCLEUS_DIR
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_library import DomeLight
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.register import register_asset, register_asset_factory
from isaaclab_arena.utils.pose import Pose

_ASSET_ROOT = f"{ARENA_NUCLEUS_DIR}/Arena/assets/object_library/temp_newton_envs/cap_envs/gear_assembly/assets"

FR3_WORKCELL_TABLE_USD_PATH = f"{_ASSET_ROOT}/industrial__fr3_workcell_table/industrial__fr3_workcell_table.usda"
HDR_SHADOW_RECEIVER_USD_PATH = f"{_ASSET_ROOT}/industrial__hdr_shadow_receiver/industrial__hdr_shadow_receiver.usda"
GEAR_ASSET_PATHS = {
    f"factory_gear_{size}": f"{_ASSET_ROOT}/industrial__factory_gear_{size}/industrial__factory_gear_{size}.usda"
    for size in ("base", "small", "medium", "large")
}


def _make_factory_gear(
    name: str,
    prim_name: str,
    usd_leaf: str,
    initial_pose: Pose | None,
) -> Object:
    """Create one Factory gear from its fully authored Cap package USD."""
    gear = Object(
        name=name,
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        object_type=ObjectType.RIGID,
        usd_path=GEAR_ASSET_PATHS[usd_leaf],
        initial_pose=initial_pose,
    )
    gear.disable_reset_pose()
    return gear


@register_asset_factory(name="factory_gear_base", object_type=ObjectType.RIGID)
def make_factory_gear_base(
    instance_name: str = "gear_base",
    initial_pose: Pose | None = None,
    **_ignored: Any,
) -> Object:
    """Create the fixed Factory gear base."""
    return _make_factory_gear(
        instance_name,
        "FactoryGearBase",
        "factory_gear_base",
        initial_pose,
    )


@register_asset_factory(name="factory_gear_small", object_type=ObjectType.RIGID)
def make_factory_gear_small(
    instance_name: str = "gear_small",
    initial_pose: Pose | None = None,
    **_ignored: Any,
) -> Object:
    """Create the source small Factory gear."""
    return _make_factory_gear(
        instance_name,
        "FactoryGearSmall",
        "factory_gear_small",
        initial_pose,
    )


@register_asset_factory(name="factory_gear_medium", object_type=ObjectType.RIGID)
def make_factory_gear_medium(
    instance_name: str = "gear_medium",
    initial_pose: Pose | None = None,
    **_ignored: Any,
) -> Object:
    """Create the source medium Factory gear."""
    return _make_factory_gear(
        instance_name,
        "FactoryGearMedium",
        "factory_gear_medium",
        initial_pose,
    )


@register_asset_factory(name="factory_gear_large", object_type=ObjectType.RIGID)
def make_factory_gear_large(
    instance_name: str = "gear_large",
    initial_pose: Pose | None = None,
    **_ignored: Any,
) -> Object:
    """Create the source large Factory gear."""
    return _make_factory_gear(
        instance_name,
        "FactoryGearLarge",
        "factory_gear_large",
        initial_pose,
    )


@register_asset
class IndustrialFr3WorkcellTable(Background):
    """Cap's FR3 workcell table background."""

    name = "fr3_workcell_table"
    tags: ClassVar[list[str]] = ["background"]
    usd_path = FR3_WORKCELL_TABLE_USD_PATH
    object_min_z = 0.0

    def __init__(
        self,
        initial_pose: Pose | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=self.name,
            usd_path=self.usd_path,
            object_min_z=self.object_min_z,
            initial_pose=initial_pose,
            tags=self.tags,
            **kwargs,
        )


@register_asset
class IndustrialHdrShadowReceiver(Object):
    """Invisible collision-free floor anchor for the HDR scene."""

    name = "hdr_shadow_receiver"
    tags: ClassVar[list[str]] = ["floor", "visual"]

    def __init__(
        self,
        instance_name: str = "hdr_shadow_receiver",
        ground_z: float = 0.0,
        initial_pose: Pose | None = None,
        **kwargs: Any,
    ) -> None:
        spawn_cfg_addon = dict(kwargs.pop("spawn_cfg_addon", {}) or {})
        spawn_cfg_addon["visible"] = False
        if initial_pose is None:
            initial_pose = Pose(position_xyz=(0.0, 0.0, ground_z + 0.0005))
        super().__init__(
            name=instance_name,
            usd_path=HDR_SHADOW_RECEIVER_USD_PATH,
            object_type=ObjectType.BASE,
            initial_pose=initial_pose,
            spawn_cfg_addon=spawn_cfg_addon,
            tags=self.tags,
            **kwargs,
        )


@register_asset
class IndustrialEmptyWarehouseDomeLight(DomeLight):
    """Cap's shared empty-warehouse HDR dome light."""

    name = "empty_warehouse_dome_light"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            spawner_cfg=sim_utils.DomeLightCfg(
                color=(0.75, 0.75, 0.75),
                intensity=1500.0,
                texture_file=EmptyWarehouseHDRRobolab.texture_file,
                texture_format=EmptyWarehouseHDRRobolab.texture_format,
                visible_in_primary_ray=True,
            ),
            **kwargs,
        )

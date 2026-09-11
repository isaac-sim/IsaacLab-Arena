# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Nucleus-hosted assets used by the ported Isaac Cap gear environment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import isaaclab.sim as sim_utils

from isaaclab_arena.assets.background import Background
from isaaclab_arena.assets.hdr_image_library import EmptyWarehouseHDRRobolab
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_library import DomeLight
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.utils.pose import Pose

_ASSET_ROOT = (
    "omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/isaac_arena/newton_envs/cap_envs/gear_assembly/assets"
)

FR3_WORKCELL_TABLE_USD_PATH = f"{_ASSET_ROOT}/industrial__fr3_workcell_table/industrial__fr3_workcell_table.usda"
HDR_SHADOW_RECEIVER_USD_PATH = f"{_ASSET_ROOT}/industrial__hdr_shadow_receiver/industrial__hdr_shadow_receiver.usda"
GEAR_ASSET_PATHS = {
    f"factory_gear_{size}": f"{_ASSET_ROOT}/industrial__factory_gear_{size}/industrial__factory_gear_{size}.usda"
    for size in ("base", "small", "medium", "large")
}


def _normalize_initial_pose(
    initial_pose: Pose | Mapping[str, Sequence[float]] | None,
) -> Pose | None:
    """Normalize graph/YAML pose mappings to Arena poses."""
    if initial_pose is None or isinstance(initial_pose, Pose):
        return initial_pose
    return Pose(
        position_xyz=tuple(float(value) for value in initial_pose["position_xyz"]),
        rotation_xyzw=tuple(float(value) for value in initial_pose["rotation_xyzw"]),
    )


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


def make_factory_gear_base(
    instance_name: str = "gear_base",
    initial_pose: Pose | Mapping[str, Sequence[float]] | None = None,
    **_ignored: Any,
) -> Object:
    """Create the fixed Factory gear base."""
    return _make_factory_gear(
        instance_name,
        "FactoryGearBase",
        "factory_gear_base",
        _normalize_initial_pose(initial_pose),
    )


def make_factory_gear_small(
    instance_name: str = "gear_small",
    initial_pose: Pose | Mapping[str, Sequence[float]] | None = None,
    **_ignored: Any,
) -> Object:
    """Create the source small Factory gear."""
    return _make_factory_gear(
        instance_name,
        "FactoryGearSmall",
        "factory_gear_small",
        _normalize_initial_pose(initial_pose),
    )


def make_factory_gear_medium(
    instance_name: str = "gear_medium",
    initial_pose: Pose | Mapping[str, Sequence[float]] | None = None,
    **_ignored: Any,
) -> Object:
    """Create the source medium Factory gear."""
    return _make_factory_gear(
        instance_name,
        "FactoryGearMedium",
        "factory_gear_medium",
        _normalize_initial_pose(initial_pose),
    )


def make_factory_gear_large(
    instance_name: str = "gear_large",
    initial_pose: Pose | Mapping[str, Sequence[float]] | None = None,
    **_ignored: Any,
) -> Object:
    """Create the source large Factory gear."""
    return _make_factory_gear(
        instance_name,
        "FactoryGearLarge",
        "factory_gear_large",
        _normalize_initial_pose(initial_pose),
    )


for _name, _factory in {
    "industrial__factory_gear_base": make_factory_gear_base,
    "industrial__factory_gear_small": make_factory_gear_small,
    "industrial__factory_gear_medium": make_factory_gear_medium,
    "industrial__factory_gear_large": make_factory_gear_large,
}.items():
    _factory.name = _name
    _factory.tags = ("object",)
    _factory.object_type = ObjectType.RIGID
del _name, _factory


class IndustrialFr3WorkcellTable(Background):
    """Cap's FR3 workcell table background."""

    name = "industrial__fr3_workcell_table"
    tags: ClassVar[list[str]] = ["background"]
    usd_path = FR3_WORKCELL_TABLE_USD_PATH
    object_min_z = 0.0

    def __init__(
        self,
        initial_pose: Pose | Mapping[str, Sequence[float]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=self.name,
            usd_path=self.usd_path,
            object_min_z=self.object_min_z,
            initial_pose=_normalize_initial_pose(initial_pose),
            tags=self.tags,
            **kwargs,
        )


class IndustrialHdrShadowReceiver(Object):
    """Invisible collision-free floor anchor for the HDR scene."""

    name = "industrial__hdr_shadow_receiver"
    tags: ClassVar[list[str]] = ["floor", "visual"]

    def __init__(
        self,
        instance_name: str = "hdr_shadow_receiver",
        ground_z: float = 0.0,
        **kwargs: Any,
    ) -> None:
        initial_pose = kwargs.pop("initial_pose", None)
        spawn_cfg_addon = dict(kwargs.pop("spawn_cfg_addon", {}) or {})
        spawn_cfg_addon["visible"] = False
        if initial_pose is None:
            initial_pose = Pose(position_xyz=(0.0, 0.0, ground_z + 0.0005))
        super().__init__(
            name=instance_name,
            usd_path=HDR_SHADOW_RECEIVER_USD_PATH,
            object_type=ObjectType.BASE,
            initial_pose=_normalize_initial_pose(initial_pose),
            spawn_cfg_addon=spawn_cfg_addon,
            tags=self.tags,
            **kwargs,
        )


class IndustrialEmptyWarehouseDomeLight(DomeLight):
    """Cap's shared empty-warehouse HDR dome light."""

    name = "industrial__empty_warehouse_dome_light"

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


GEAR_ASSET_ENTRY_POINTS = {
    make_factory_gear_base.name: make_factory_gear_base,
    make_factory_gear_small.name: make_factory_gear_small,
    make_factory_gear_medium.name: make_factory_gear_medium,
    make_factory_gear_large.name: make_factory_gear_large,
}

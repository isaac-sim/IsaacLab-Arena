# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Nucleus-hosted assets for the CAP easy tool-sorting environments."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

import isaaclab.sim as sim_utils

from isaaclab_arena.assets.nucleus import ARENA_NUCLEUS_DIR
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.relations.collision_mode import CollisionMode
from isaaclab_arena.utils.pose import Pose

TOOL_SORT_ASSET_ROOT = f"{ARENA_NUCLEUS_DIR}/Arena/assets/object_library/temp_newton_envs/cap_envs/tool_sorting/assets"
"""Published tool-sorting asset tree."""


class ToolSortBinAppearance(StrEnum):
    """Available destination-bin label sets."""

    DEFAULT = "default"
    BENCH = "bench"
    ELECTRICAL = "electrical"
    WIRING = "wiring"


def _tool_usd_path(name: str) -> str:
    return f"{TOOL_SORT_ASSET_ROOT}/{name}/{name}.usda"


class IndustrialToolSortObject(LibraryObject):
    """Base class for rigid tools in the CAP sorting scenes."""

    tags = ["object", "graspable", "industrial", "tool_sort"]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.disable_reset_pose()


class IndustrialToolSortAdjustableWrench(IndustrialToolSortObject):
    name = "vabar_tool_sort__adjustable_wrench"
    usd_path = _tool_usd_path(name)


class IndustrialToolSortBattery(IndustrialToolSortObject):
    name = "vabar_tool_sort__battery"
    usd_path = _tool_usd_path(name)


class IndustrialToolSortBreadboard(IndustrialToolSortObject):
    name = "vabar_tool_sort__breadboard"
    usd_path = _tool_usd_path(name)


class IndustrialToolSortCombinationPliers(IndustrialToolSortObject):
    name = "vabar_tool_sort__combination_pliers"
    usd_path = _tool_usd_path(name)


class IndustrialToolSortCuttingPliers(IndustrialToolSortObject):
    name = "vabar_tool_sort__cutting_pliers"
    usd_path = _tool_usd_path(name)


class IndustrialToolSortFlashlight(IndustrialToolSortObject):
    name = "vabar_tool_sort__flashlight"
    usd_path = _tool_usd_path(name)


class IndustrialToolSortInsulatingTape(IndustrialToolSortObject):
    name = "vabar_tool_sort__insulating_tape"
    usd_path = _tool_usd_path(name)


class IndustrialToolSortMultimeter(IndustrialToolSortObject):
    name = "vabar_tool_sort__multimeter"
    usd_path = _tool_usd_path(name)


class IndustrialToolSortSafetyGlasses(IndustrialToolSortObject):
    name = "vabar_tool_sort__safety_glasses"
    usd_path = _tool_usd_path(name)


class IndustrialToolSortSlottedScrewdriver(IndustrialToolSortObject):
    name = "vabar_tool_sort__slotted_screwdriver"
    usd_path = _tool_usd_path(name)


class IndustrialToolSortTapeMeasure(IndustrialToolSortObject):
    name = "vabar_tool_sort__tape_measure"
    usd_path = _tool_usd_path(name)


class IndustrialToolSortWireSpool(IndustrialToolSortObject):
    name = "vabar_tool_sort__wire_spool"
    usd_path = _tool_usd_path(name)


class IndustrialToolSortBin(Object):
    """Kinematic source or compartmented destination bin."""

    name = "industrial__tool_sort_bin"
    tags = ["object", "container", "industrial", "tool_sort"]
    object_type = ObjectType.RIGID

    def __init__(
        self,
        instance_name: str = "tool_sort_bin",
        side: Literal["source", "destination"] = "destination",
        appearance: ToolSortBinAppearance | str = ToolSortBinAppearance.DEFAULT,
        initial_pose: Pose | None = None,
    ) -> None:
        assert side in {"source", "destination"}, f"Invalid tool-sort bin side: {side!r}"
        appearance = ToolSortBinAppearance(appearance)
        leaf = "bin1.usda" if side == "source" else f"bin2_{appearance}.usda"
        super().__init__(
            name=instance_name,
            prim_path=f"{{ENV_REGEX_NS}}/{instance_name}",
            object_type=ObjectType.RIGID,
            usd_path=f"{TOOL_SORT_ASSET_ROOT}/{self.name}/{leaf}",
            initial_pose=initial_pose,
            spawn_cfg_addon={
                "rigid_props": sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            },
            tags=self.tags,
        )
        self.collision_mode = CollisionMode.MESH


TOOL_SORT_ASSET_CLASSES = (
    IndustrialToolSortAdjustableWrench,
    IndustrialToolSortBattery,
    IndustrialToolSortBreadboard,
    IndustrialToolSortCombinationPliers,
    IndustrialToolSortCuttingPliers,
    IndustrialToolSortFlashlight,
    IndustrialToolSortInsulatingTape,
    IndustrialToolSortMultimeter,
    IndustrialToolSortSafetyGlasses,
    IndustrialToolSortSlottedScrewdriver,
    IndustrialToolSortTapeMeasure,
    IndustrialToolSortWireSpool,
    IndustrialToolSortBin,
)
"""Asset classes registered by the Isaac CAP entry point."""

# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Syringe manipulands and fixtures for the shared CAP FR3 workcell."""

from isaaclab_arena.assets.nucleus import ARENA_NUCLEUS_DIR
from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose

# TODO(alexmillane) [cap-assets-permanent-location]: Replace temp_newton_envs with the permanent asset layout.
SYRINGE_ASSET_ROOT = (
    f"{ARENA_NUCLEUS_DIR}/Arena/assets/object_library/temp_newton_envs/cap_envs/syringe_disposal/assets"
)


@register_asset
class SyringeRedCap(LibraryObject):
    """Red-cap syringe."""

    name = "syringe"
    tags = ["object", "graspable"]
    usd_path = f"{SYRINGE_ASSET_ROOT}/vabar_tool_sort__syringe/vabar_tool_sort__syringe.usda"


@register_asset
class SyringeWhiteCap(LibraryObject):
    """White-cap syringe."""

    name = "syringe_blank"
    tags = ["object", "graspable"]
    usd_path = f"{SYRINGE_ASSET_ROOT}/vabar_tool_sort__syringe_blank/vabar_tool_sort__syringe_blank.usda"


@register_asset
class InstrumentTray(LibraryObject):
    """Instrument tray with the authored cavity colliders."""

    name = "instrument_tray"
    tags = ["object", "container"]
    usd_path = f"{SYRINGE_ASSET_ROOT}/vabar_tool_sort__instrument_tray/vabar_tool_sort__instrument_tray.usda"

    def __init__(self, initial_pose: Pose | None = None, **kwargs):
        super().__init__(initial_pose=initial_pose, collision_mode="mesh", **kwargs)


@register_asset
class SharpsContainer(InstrumentTray):
    """Sharps container with the authored aperture and interior."""

    name = "sharps_container"
    usd_path = f"{SYRINGE_ASSET_ROOT}/industrial__tool_sort_bin/bin2_syringe.usda"

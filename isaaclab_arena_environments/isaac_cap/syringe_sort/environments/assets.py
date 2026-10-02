# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Syringe manipulands and fixtures for the shared CAP FR3 workcell."""

from isaaclab.sim.spawners.from_files import spawn_from_usd
from isaaclab.sim.utils import clone

from isaaclab_arena.assets.nucleus import ARENA_NUCLEUS_DIR
from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose

# TODO(alexmillane) [cap-assets-permanent-location]: Replace temp_newton_envs with the permanent asset layout.
SYRINGE_ASSET_ROOT = (
    f"{ARENA_NUCLEUS_DIR}/Arena/assets/object_library/temp_newton_envs/cap_envs/syringe_disposal/assets"
)


@clone
def _spawn_instrument_tray(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Restore current CAP's mesh collider on the hosted tray asset."""
    from pxr import UsdPhysics

    prim = spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    body = prim.GetChild("Geometry").GetChild("instrument_tray_01_obj_00")
    mesh = body.GetChild("instrument_tray_01_mesh_00")
    cavity = body.GetChild("CavityCollision")
    assert mesh.IsValid() and cavity.IsValid(), "Hosted syringe tray structure changed"
    # The hosted copy adds primitive cavity faces absent from CAP's current asset.
    # Keep its visual/material references and match CAP's original mesh collision.
    cavity.SetActive(False)
    UsdPhysics.CollisionAPI(mesh).CreateCollisionEnabledAttr(True)
    return prim


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
    """Instrument tray with CAP's authored mesh collider."""

    name = "instrument_tray"
    tags = ["object", "container"]
    usd_path = f"{SYRINGE_ASSET_ROOT}/vabar_tool_sort__instrument_tray/vabar_tool_sort__instrument_tray.usda"
    spawn_cfg_addon = {"func": _spawn_instrument_tray}


@register_asset
class SharpsContainer(LibraryObject):
    """Sharps container with the authored aperture and interior."""

    name = "sharps_container"
    tags = ["object", "container"]
    usd_path = f"{SYRINGE_ASSET_ROOT}/industrial__tool_sort_bin/bin2_syringe.usda"

    def __init__(self, initial_pose: Pose | None = None, **kwargs):
        super().__init__(initial_pose=initial_pose, collision_mode="mesh", **kwargs)

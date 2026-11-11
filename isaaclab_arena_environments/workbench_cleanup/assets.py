# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Registered workbench assets from SimReady packages and prepared USD layers."""

import os

import isaaclab.sim as sim_utils

from isaaclab_arena.affordances.openable import Openable
from isaaclab_arena.affordances.pressable import Pressable
from isaaclab_arena.assets.background_library import LibraryBackground
from isaaclab_arena.assets.nucleus import ARENA_STAGING_NUCLEUS_DIR
from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.assets.simready_constants import SIMREADY_PHYSICS_VARIANTS
from isaaclab_arena.assets.sources.simready_central import SimReadyCentralUsdSource

# -----------------------------------------------------------------------------
# Direct SimReady Central assets
# -----------------------------------------------------------------------------


def _make_hand_tool_spawn_cfg_addon() -> dict[str, object]:
    """Create the common workbench hand-tool spawn overrides."""
    return {
        "variants": SIMREADY_PHYSICS_VARIANTS,
        "make_uninstanceable": True,
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(
            linear_damping=2.0,
            angular_damping=2.0,
            sleep_threshold=0.0,
        ),
        "collision_props": sim_utils.CollisionPropertiesCfg(
            contact_offset=0.001,
            rest_offset=0.0,
            mesh_collision_property=sim_utils.MeshCollisionPropertiesCfg(mesh_approximation_name="convexHull"),
        ),
    }


@register_asset
class WorkbenchCleanupWhiteBin(LibraryObject):
    """White plastic bin used by the workbench cleanup tasks."""

    registry_name = "workbench_cleanup_white_bin"
    tags = ["object", "container", "sim-ready", "workbench"]
    usd_source = SimReadyCentralUsdSource(
        "https://huggingface.co/datasets/PalatialSim/palatial-nv-asset-library/resolve/"
        "69a87329036cec0440403351c12230970c4e67d8/D00814/simready_usd/D00814.usd"
    )
    object_type = ObjectType.RIGID
    spawn_cfg_addon = {"variants": SIMREADY_PHYSICS_VARIANTS}


@register_asset
class WorkbenchCleanupSlottedScrewdriver(LibraryObject):
    """Slotted screwdriver with workbench collision and damping settings."""

    registry_name = "workbench_cleanup_slotted_screwdriver"
    tags = ["object", "graspable", "tool", "sim-ready", "workbench"]
    usd_source = SimReadyCentralUsdSource(
        "https://huggingface.co/datasets/nvidia/simready-assets/resolve/"
        "653d91e182011f357cf1644013cc94ddf194d349/handTool_screwdriver_slotted_55mm_a01/"
        "simready_usd/sm_handTool_screwdriver_slotted_55mm_a01_01.usd"
    )
    object_type = ObjectType.RIGID
    spawn_cfg_addon = _make_hand_tool_spawn_cfg_addon()


@register_asset
class WorkbenchCleanupSmallWrench(LibraryObject):
    """Combination wrench with workbench physics settings."""

    registry_name = "workbench_cleanup_small_wrench"
    tags = ["object", "graspable", "tool", "sim-ready", "workbench"]
    usd_source = SimReadyCentralUsdSource(
        "https://huggingface.co/datasets/nvidia/simready-assets/resolve/"
        "653d91e182011f357cf1644013cc94ddf194d349/handTool_wrench_combination_a01/"
        "simready_usd/sm_handTool_wrench_combination_a01_01.usd"
    )
    object_type = ObjectType.RIGID
    spawn_cfg_addon = _make_hand_tool_spawn_cfg_addon()


# -----------------------------------------------------------------------------
# Prepared assets
# -----------------------------------------------------------------------------

WORKBENCH_ASSET_ROOT = os.environ.get(
    "ARENA_WORKBENCH_ASSET_ROOT", f"{ARENA_STAGING_NUCLEUS_DIR}/Arena/assets/object_library/workbench"
).rstrip("/")
"""Uploaded bundle root; override with a local directory when testing rebuilt layers."""


@register_asset
class WorkbenchCleanupWorkbench(LibraryBackground):
    """Articulated workbench with independently contact-enabled shelf surfaces."""

    registry_name = "workbench_cleanup_workbench"
    tags = ["background", "sim-ready", "workbench"]
    usd_path = f"{WORKBENCH_ASSET_ROOT}/workbench.usda"
    object_min_z = 0.0
    spawn_cfg_addon = {"variants": SIMREADY_PHYSICS_VARIANTS}


@register_asset
class WorkbenchCleanupSocketSet(LibraryObject, Openable):
    """Fixed-base ratchet socket set with a passive openable lid."""

    registry_name = "workbench_cleanup_socket_set"
    tags = ["object", "openable", "sim-ready", "workbench"]
    usd_path = f"{WORKBENCH_ASSET_ROOT}/socket_set.usda"
    object_type = ObjectType.ARTICULATION
    spawn_cfg_addon = {"variants": SIMREADY_PHYSICS_VARIANTS}
    openable_joint_name = "case_lid"

    def __init__(self, **kwargs):
        initial_joint_positions = kwargs.pop("initial_joint_positions", None)
        kwargs.setdefault("openable_joint_name", self.openable_joint_name)
        super().__init__(**kwargs)
        if initial_joint_positions is not None:
            self.object_cfg.init_state.joint_pos = dict(initial_joint_positions)


@register_asset
class WorkbenchCleanupOscilloscope(LibraryObject, Pressable):
    """Fixed oscilloscope with one spring-return power button."""

    registry_name = "workbench_cleanup_oscilloscope"
    tags = ["object", "pressable", "sim-ready", "workbench"]
    usd_path = f"{WORKBENCH_ASSET_ROOT}/oscilloscope.usda"
    object_type = ObjectType.ARTICULATION
    spawn_cfg_addon = {"variants": SIMREADY_PHYSICS_VARIANTS}
    pressable_joint_name = "joint_button_01_joint_00"

    def __init__(self, **kwargs):
        kwargs.setdefault("pressable_joint_name", self.pressable_joint_name)
        super().__init__(**kwargs)

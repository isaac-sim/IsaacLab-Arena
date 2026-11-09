# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Registered workbench assets derived from revision-pinned SimReady packages."""

from isaaclab_arena.affordances.openable import Openable
from isaaclab_arena.affordances.pressable import Pressable
from isaaclab_arena.assets.background_library import LibraryBackground
from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.assets.sources.derived import DerivedUsdSource
from isaaclab_arena.assets.sources.simready import SIMREADY_PHYSICS_VARIANTS, SimReadyCentralPackageSource

_RECIPE_ROOT = "package://isaaclab_arena_environments/workbench_cleanup/assets/recipes"

_WORKBENCH = DerivedUsdSource(
    SimReadyCentralPackageSource(
        "hf://huggingface.co/datasets/nvidia/simready-assets/resolve/"
        "653d91e182011f357cf1644013cc94ddf194d349/furniture_workbench_standard_e01/"
        "simready_usd/sm_furniture_workbench_standard_e01_01.usd"
    ),
    f"{_RECIPE_ROOT}/workbench.json",
)
_WHITE_BIN = DerivedUsdSource(
    SimReadyCentralPackageSource(
        "hf://huggingface.co/datasets/PalatialSim/palatial-nv-asset-library/resolve/"
        "69a87329036cec0440403351c12230970c4e67d8/D00814/simready_usd/D00814.usd"
    ),
    f"{_RECIPE_ROOT}/white_bin.json",
)
_SOCKET_SET = DerivedUsdSource(
    SimReadyCentralPackageSource(
        "hf://huggingface.co/datasets/ForgeCAD/simready-assets/resolve/"
        "0573053fa2c9174ca2a4678a462ec4dfd3035d59/ratchet_socket_set/"
        "simready_usd/ratchet_socket_set.usda"
    ),
    f"{_RECIPE_ROOT}/socket_set.json",
)
_OSCILLOSCOPE = DerivedUsdSource(
    SimReadyCentralPackageSource(
        "hf://huggingface.co/datasets/nvidia/simready-assets/resolve/"
        "653d91e182011f357cf1644013cc94ddf194d349/testingTool_oscilloscope_a01/"
        "simready_usd/sm_testingTool_oscilloscope_a01_01.usd"
    ),
    f"{_RECIPE_ROOT}/oscilloscope.json",
)
_SLOTTED_SCREWDRIVER = DerivedUsdSource(
    SimReadyCentralPackageSource(
        "hf://huggingface.co/datasets/nvidia/simready-assets/resolve/"
        "653d91e182011f357cf1644013cc94ddf194d349/handTool_screwdriver_slotted_55mm_a01/"
        "simready_usd/sm_handTool_screwdriver_slotted_55mm_a01_01.usd"
    ),
    f"{_RECIPE_ROOT}/slotted_screwdriver.json",
)
_SMALL_WRENCH = DerivedUsdSource(
    SimReadyCentralPackageSource(
        "hf://huggingface.co/datasets/nvidia/simready-assets/resolve/"
        "653d91e182011f357cf1644013cc94ddf194d349/handTool_wrench_combination_a01/"
        "simready_usd/sm_handTool_wrench_combination_a01_01.usd"
    ),
    f"{_RECIPE_ROOT}/small_wrench.json",
)


@register_asset
class WorkbenchCleanupWorkbench(LibraryBackground):
    """Articulated workbench with independently contact-enabled shelf surfaces."""

    registry_name = "workbench_cleanup_workbench"
    tags = ["background", "sim-ready", "workbench"]
    usd_path = _WORKBENCH
    object_min_z = 0.0
    spawn_cfg_addon = {"variants": SIMREADY_PHYSICS_VARIANTS}


@register_asset
class WorkbenchCleanupWhiteBin(LibraryObject):
    """White plastic bin used by the workbench cleanup tasks."""

    registry_name = "workbench_cleanup_white_bin"
    tags = ["object", "container", "sim-ready", "workbench"]
    usd_path = _WHITE_BIN
    object_type = ObjectType.RIGID
    spawn_cfg_addon = {"variants": SIMREADY_PHYSICS_VARIANTS}


@register_asset
class WorkbenchCleanupSocketSet(LibraryObject, Openable):
    """Fixed-base ratchet socket set with a passive openable lid."""

    registry_name = "workbench_cleanup_socket_set"
    tags = ["object", "openable", "sim-ready", "workbench"]
    usd_path = _SOCKET_SET
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
    usd_path = _OSCILLOSCOPE
    object_type = ObjectType.ARTICULATION
    spawn_cfg_addon = {"variants": SIMREADY_PHYSICS_VARIANTS}
    pressable_joint_name = "joint_button_01_joint_00"

    def __init__(self, **kwargs):
        kwargs.setdefault("pressable_joint_name", self.pressable_joint_name)
        super().__init__(**kwargs)


@register_asset
class WorkbenchCleanupSlottedScrewdriver(LibraryObject):
    """Slotted screwdriver with workbench collision and damping settings."""

    registry_name = "workbench_cleanup_slotted_screwdriver"
    tags = ["object", "graspable", "tool", "sim-ready", "workbench"]
    usd_path = _SLOTTED_SCREWDRIVER
    object_type = ObjectType.RIGID
    spawn_cfg_addon = {"variants": SIMREADY_PHYSICS_VARIANTS}


@register_asset
class WorkbenchCleanupSmallWrench(LibraryObject):
    """Three-quarter-scale combination wrench with workbench physics settings."""

    registry_name = "workbench_cleanup_small_wrench"
    tags = ["object", "graspable", "tool", "sim-ready", "workbench"]
    usd_path = _SMALL_WRENCH
    object_type = ObjectType.RIGID
    spawn_cfg_addon = {"variants": SIMREADY_PHYSICS_VARIANTS}

# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Registered workbench assets processed from revision-pinned SimReady packages."""

import importlib.resources

from isaaclab_arena.affordances.openable import Openable
from isaaclab_arena.affordances.pressable import Pressable
from isaaclab_arena.assets.background_library import LibraryBackground
from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.assets.simready_constants import SIMREADY_PHYSICS_VARIANTS
from isaaclab_arena_environments.workbench_cleanup.asset_processing.usd_asset_processor import (
    UsdAssetProcessor,
    retrieve_pinned_usd_path,
)

_RECIPE_ROOT = "package://isaaclab_arena_environments/workbench_cleanup/asset_processing/recipes"
_WRAPPER_ROOT = "package://isaaclab_arena_environments/workbench_cleanup/asset_processing/wrappers"
_WRAPPER_RESOURCE_ROOT = "workbench_cleanup/asset_processing/wrappers"

_WORKBENCH = UsdAssetProcessor(
    "https://huggingface.co/datasets/nvidia/simready-assets/resolve/"
    "653d91e182011f357cf1644013cc94ddf194d349/furniture_workbench_standard_e01/"
    "simready_usd/sm_furniture_workbench_standard_e01_01.usd",
    f"{_WRAPPER_ROOT}/workbench.usda",
    structural_recipe=f"{_RECIPE_ROOT}/workbench.json",
)


def _retrieve_wrapper(wrapper_name: str) -> str:
    """Localize a packaged USDA wrapper and its remote dependencies."""
    resource = importlib.resources.files("isaaclab_arena_environments").joinpath(_WRAPPER_RESOURCE_ROOT, wrapper_name)
    with importlib.resources.as_file(resource) as wrapper_path:
        return retrieve_pinned_usd_path(str(wrapper_path))


class _ProcessedLibraryBackground(LibraryBackground):
    processor: UsdAssetProcessor

    def __init__(self, **kwargs):
        self.usd_path = self.processor.process()
        super().__init__(**kwargs)


class _RetrievedWrapperLibraryObject(LibraryObject):
    wrapper_name: str

    def __init__(self, **kwargs):
        self.usd_path = _retrieve_wrapper(self.wrapper_name)
        super().__init__(**kwargs)


@register_asset
class WorkbenchCleanupWorkbench(_ProcessedLibraryBackground):
    """Articulated workbench with independently contact-enabled shelf surfaces."""

    registry_name = "workbench_cleanup_workbench"
    tags = ["background", "sim-ready", "workbench"]
    processor = _WORKBENCH
    object_min_z = 0.0
    spawn_cfg_addon = {"variants": SIMREADY_PHYSICS_VARIANTS}


@register_asset
class WorkbenchCleanupWhiteBin(_RetrievedWrapperLibraryObject):
    """White plastic bin used by the workbench cleanup tasks."""

    registry_name = "workbench_cleanup_white_bin"
    tags = ["object", "container", "sim-ready", "workbench"]
    wrapper_name = "white_bin.usda"
    object_type = ObjectType.RIGID
    spawn_cfg_addon = {"variants": SIMREADY_PHYSICS_VARIANTS}


@register_asset
class WorkbenchCleanupSocketSet(_RetrievedWrapperLibraryObject, Openable):
    """Fixed-base ratchet socket set with a passive openable lid."""

    registry_name = "workbench_cleanup_socket_set"
    tags = ["object", "openable", "sim-ready", "workbench"]
    wrapper_name = "socket_set.usda"
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
class WorkbenchCleanupOscilloscope(_RetrievedWrapperLibraryObject, Pressable):
    """Fixed oscilloscope with one spring-return power button."""

    registry_name = "workbench_cleanup_oscilloscope"
    tags = ["object", "pressable", "sim-ready", "workbench"]
    wrapper_name = "oscilloscope.usda"
    object_type = ObjectType.ARTICULATION
    spawn_cfg_addon = {"variants": SIMREADY_PHYSICS_VARIANTS}
    pressable_joint_name = "joint_button_01_joint_00"

    def __init__(self, **kwargs):
        kwargs.setdefault("pressable_joint_name", self.pressable_joint_name)
        super().__init__(**kwargs)


@register_asset
class WorkbenchCleanupSlottedScrewdriver(_RetrievedWrapperLibraryObject):
    """Slotted screwdriver with workbench collision and damping settings."""

    registry_name = "workbench_cleanup_slotted_screwdriver"
    tags = ["object", "graspable", "tool", "sim-ready", "workbench"]
    wrapper_name = "slotted_screwdriver.usda"
    object_type = ObjectType.RIGID
    spawn_cfg_addon = {"variants": SIMREADY_PHYSICS_VARIANTS}


@register_asset
class WorkbenchCleanupSmallWrench(_RetrievedWrapperLibraryObject):
    """Three-quarter-scale combination wrench with workbench physics settings."""

    registry_name = "workbench_cleanup_small_wrench"
    tags = ["object", "graspable", "tool", "sim-ready", "workbench"]
    wrapper_name = "small_wrench.usda"
    object_type = ObjectType.RIGID
    spawn_cfg_addon = {"variants": SIMREADY_PHYSICS_VARIANTS}

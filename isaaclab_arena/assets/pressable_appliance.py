# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pressable appliances loaded from pinned asset packages."""

from pxr import UsdPhysics

from isaaclab_arena.affordances.pressable import Pressable
from isaaclab_arena.assets.lightwheel_utils import acquire_lightwheel_asset
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.assets.simready_central_asset_loader import (
    _download_central_asset_package_to_cache,
    is_central_asset_url,
)
from isaaclab_arena.assets.turnable_appliance import _write_appliance_wrapper


@register_asset
class PressableAppliance(Object, Pressable):
    """A countertop articulation exposing a prismatic button through Pressable."""

    registry_name = "pressable_appliance"
    tags = ["object", "pressable", "appliance"]
    object_type = ObjectType.ARTICULATION

    def __init__(
        self,
        pressable_joint_name: str,
        usd_path: str | None = None,
        lightwheel_file_version: str | None = None,
        pressedness_threshold: float = 0.5,
        initial_joint_positions: dict[str, float] | None = None,
        name: str | None = None,
        **kwargs,
    ):
        """Load an appliance whose button moves from its lower toward its upper joint limit.

        Args:
            pressable_joint_name: Prismatic joint controlling the button.
            usd_path: Local USD path or revision-pinned SimReady Central URL.
            lightwheel_file_version: Pinned Lightwheel USD version, instead of usd_path.
            pressedness_threshold: Fraction of joint travel required for success.
            initial_joint_positions: Initial joint positions in radians or meters.
            name: Scene object name.
            **kwargs: Additional Object configuration.
        """
        assert bool(usd_path) != bool(lightwheel_file_version), "Specify exactly one appliance source"
        if lightwheel_file_version:
            from lightwheel_sdk.loader import object_loader

            usd_path, _, _ = acquire_lightwheel_asset(
                object_loader,
                object_loader.acquire_by_file_version,
                description=f"Lightwheel appliance {lightwheel_file_version}",
                file_version_id=lightwheel_file_version,
            )
        elif is_central_asset_url(usd_path):
            usd_path = _download_central_asset_package_to_cache(usd_path)
        usd_path = _write_appliance_wrapper(usd_path, pressable_joint_name, joint_type=UsdPhysics.PrismaticJoint)
        spawn_cfg_addon = dict(kwargs.pop("spawn_cfg_addon", None) or {})
        spawn_cfg_addon.setdefault("fix_root_link", True)
        super().__init__(
            name=name or self.registry_name,
            usd_path=usd_path,
            object_type=ObjectType.ARTICULATION,
            pressable_joint_name=pressable_joint_name,
            pressedness_threshold=pressedness_threshold,
            spawn_cfg_addon=spawn_cfg_addon,
            **kwargs,
        )
        if initial_joint_positions is not None:
            self.object_cfg.init_state.joint_pos = dict(initial_joint_positions)

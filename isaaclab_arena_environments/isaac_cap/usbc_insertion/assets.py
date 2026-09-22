# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Registered S3-hosted assets used by the USB-C environment graphs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy

import isaaclab.sim as sim_utils
from isaaclab_newton.sim.schemas import NewtonCollisionCfg

from isaaclab_arena.assets.background import Background
from isaaclab_arena.assets.nucleus import ARENA_NUCLEUS_DIR
from isaaclab_arena.assets.object_library import DomeLight, LibraryObject
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.assets.registries import HDRImageRegistry
from isaaclab_arena.utils.pose import Pose, PoseRange

from .cables import UsbcConnectorCable

ASSET_ROOT = f"{ARENA_NUCLEUS_DIR}/Arena/assets/object_library/temp_newton_envs/usbc_insertion/assets"


class _UsbcAsset(LibraryObject):
    """Give each registered USB-C asset independent mutable simulator configuration."""

    tags = ["object", "usbc_insertion"]
    spawn_cfg_addon = {"copy_from_source": False}

    def __init__(
        self,
        *,
        initial_pose: Pose | PoseRange | Mapping[str, Sequence[float]] | None = None,
        **kwargs,
    ) -> None:
        """Accept a typed pose or a YAML mapping describing a pose or reset range.

        Args:
            initial_pose: Fixed pose or randomized reset range, typed or YAML-authored.
            **kwargs: Instance name, prim path, and other LibraryObject arguments.
        """
        if isinstance(initial_pose, Mapping):
            pose_type = PoseRange if "position_xyz_min" in initial_pose else Pose
            initial_pose = pose_type(**{key: tuple(value) for key, value in initial_pose.items()})
        self.spawn_cfg_addon = deepcopy(self.spawn_cfg_addon)
        self.asset_cfg_addon = deepcopy(self.asset_cfg_addon)
        super().__init__(initial_pose=initial_pose, **kwargs)


class _UsbcFixture(_UsbcAsset):
    """Keep a rigid USB-C fixture kinematic with ordinary initial-pose resets."""

    spawn_cfg_addon = {
        "copy_from_source": False,
        "rigid_props": sim_utils.RigidBodyBaseCfg(kinematic_enabled=True),
    }


class _UsbcConnector(_UsbcAsset):
    """Use the authored connector collision mesh with the task's contact margin."""

    spawn_cfg_addon = {
        "copy_from_source": False,
        "rigid_props": sim_utils.RigidBodyBaseCfg(kinematic_enabled=False),
        "collision_props": [NewtonCollisionCfg(contact_margin=0.0, contact_gap=1.0e-4)],
    }


@register_asset
class UsbcEasyPlug(_UsbcConnector):
    """CAP's scaled plug with the easy task's authored mesh contacts."""

    name = "usbc_insertion_easy_plug"
    usd_path = f"{ASSET_ROOT}/industrial__usbc_easy_plug/industrial__usbc_easy_plug.usda"
    spawn_cfg_addon = {key: value for key, value in _UsbcConnector.spawn_cfg_addon.items() if key != "collision_props"}


@register_asset
class UsbcMediumPlug(_UsbcConnector):
    """CAP's medium plug with matched bulkhead contact geometry."""

    name = "usbc_insertion_medium_plug"
    usd_path = f"{ASSET_ROOT}/industrial__usbc_easy_plug/medium_plug.usda"
    spawn_cfg_addon = UsbcEasyPlug.spawn_cfg_addon


@register_asset
class UsbcEasyPort(_UsbcFixture):
    """Fixed chamfered receptacle for the easy task."""

    name = "usbc_insertion_easy_port"
    usd_path = f"{ASSET_ROOT}/industrial__usbc_easy_port/industrial__usbc_easy_port.usda"


@register_asset
class UsbcBulkhead(_UsbcConnector):
    """Dynamic receiver used by the bimanual USB-C task."""

    name = "usbc_insertion_bulkhead"
    usd_path = f"{ASSET_ROOT}/vabar_usbc_insert__bulkhead/vabar_usbc_insert__bulkhead.usda"
    spawn_cfg_addon = UsbcEasyPlug.spawn_cfg_addon


@register_asset
class UsbcBench(_UsbcFixture):
    """Bench supporting the plug in both USB-C variants."""

    name = "usbc_insertion_bench"
    usd_path = f"{ASSET_ROOT}/industrial__yam_usbc_fixtures/bench.usda"


@register_asset
class UsbcCradleFront(_UsbcFixture):
    """Front support for the medium USB-C bulkhead."""

    name = "usbc_insertion_cradle_front"
    usd_path = f"{ASSET_ROOT}/industrial__yam_usbc_fixtures/cradle_front.usda"


@register_asset
class UsbcCradleRear(_UsbcFixture):
    """Rear support for the medium USB-C bulkhead."""

    name = "usbc_insertion_cradle_rear"
    usd_path = f"{ASSET_ROOT}/industrial__yam_usbc_fixtures/cradle_rear.usda"


class _UsbcTable(Background):
    """A workcell background with YAML-authored placement and nested physics reset."""

    tags = ["background", "usbc_insertion"]

    def __init__(
        self,
        *,
        instance_name: str | None = None,
        initial_pose: Pose | Mapping[str, Sequence[float]] | None = None,
        **kwargs,
    ) -> None:
        """Configure a registered workcell table background.

        Args:
            instance_name: Optional scene name overriding the registry name.
            initial_pose: Fixed table pose, typed or YAML-authored.
            **kwargs: Additional Background configuration, including the prim path.
        """
        if isinstance(initial_pose, Mapping):
            initial_pose = Pose(**{key: tuple(value) for key, value in initial_pose.items()})
        super().__init__(
            name=instance_name if instance_name is not None else self.name,
            usd_path=self.usd_path,
            object_min_z=0.0,
            initial_pose=initial_pose,
            tags=list(self.tags),
            spawn_cfg_addon={"copy_from_source": False},
            **kwargs,
        )


@register_asset
class UsbcYamWorkcellTable(_UsbcTable):
    """YAM table from the USB-C asset bundle."""

    name = "usbc_insertion_yam_table"
    usd_path = f"{ASSET_ROOT}/industrial__yam_workcell_table/industrial__yam_workcell_table.usda"


@register_asset
class UsbcHdrShadowReceiver(_UsbcAsset):
    """Invisible collision-free floor anchor for the USB-C workcell."""

    name = "usbc_insertion_hdr_shadow_receiver"
    tags = ["background", "usbc_insertion"]
    object_type = ObjectType.BASE
    usd_path = f"{ASSET_ROOT}/industrial__hdr_shadow_receiver/industrial__hdr_shadow_receiver.usda"
    spawn_cfg_addon = {"copy_from_source": False, "visible": False}
    asset_cfg_addon = {"collision_group": -1}


@register_asset
class UsbcDomeLight(DomeLight):
    """Dome light configured with an HDR registry name from environment YAML."""

    name = "usbc_insertion_dome_light"
    tags = ["light", "usbc_insertion"]

    def __init__(
        self,
        *,
        hdr_name: str = "empty_warehouse_robolab",
        intensity: float = 1500.0,
        color: Sequence[float] = (0.75, 0.75, 0.75),
        **kwargs,
    ) -> None:
        """Configure an HDR-backed light with YAML-friendly parameters.

        Args:
            hdr_name: Name of the environment map in HDRImageRegistry.
            intensity: Initial dome-light intensity.
            color: RGB light color.
            **kwargs: Instance name, prim path, and other DomeLight arguments.
        """
        super().__init__(hdr=HDRImageRegistry().get_hdr_by_name(hdr_name)(), **kwargs)
        self.set_intensity(intensity)
        self.set_color(tuple(color))


USBC_ASSET_CLASSES = (
    UsbcEasyPlug,
    UsbcMediumPlug,
    UsbcEasyPort,
    UsbcBulkhead,
    UsbcBench,
    UsbcCradleFront,
    UsbcCradleRear,
    UsbcYamWorkcellTable,
    UsbcHdrShadowReceiver,
    UsbcDomeLight,
    UsbcConnectorCable,
)

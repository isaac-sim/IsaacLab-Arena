# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from abc import ABC, abstractmethod

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg

# Re-export ObjectType from the lightweight module so existing
# `from isaaclab_arena.assets.object_base import ObjectType` consumers keep working,
# while pure-Python spec modules can import from `object_type` directly without
# pulling in isaaclab/omni/pxr at module-load time.
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.relations.placement_asset import PlaceableAsset
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.velocity import Velocity

__all__ = [
    "ObjectBase",
    "ObjectType",
]


class ObjectBase(PlaceableAsset, ABC):
    """Base class for Arena scene objects with a construction-time immutable prim path."""

    def __init__(
        self,
        name: str,
        prim_path: str | None = None,
        object_type: ObjectType = ObjectType.BASE,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        if prim_path is None:
            prim_path = "{ENV_REGEX_NS}/" + self.name
        self.prim_path = prim_path
        self.object_type = object_type
        self.initial_velocity: Velocity | None = None
        self.object_cfg = None

    def resolve_object_cfg(self, physics_preset: object | None = None):
        """Return the concrete Isaac Lab config for a selected physics preset."""
        return self.object_cfg

    def get_prim_path(self) -> str:
        return self.prim_path

    def get_object_cfg(self) -> tuple[str, object]:
        return self.name, self.object_cfg

    def get_event_cfg(self) -> tuple[str, EventTermCfg | None]:
        return self.name, self._pose_event_cfg

    @abstractmethod
    def get_object_pose(self, env: ManagerBasedEnv, is_relative: bool = True) -> torch.Tensor:
        """Return the object's per-environment pose as ``(x, y, z, qx, qy, qz, qw)``."""

    @abstractmethod
    def set_object_pose(self, env: ManagerBasedEnv, pose: Pose, env_ids: torch.Tensor | None = None) -> None:
        """Write the object's pose for the selected environments."""

    def get_contact_sensor_cfg(self, contact_against_object: ObjectBase | None = None) -> ContactSensorCfg:
        """Return a contact sensor config when this object representation supports one."""
        raise NotImplementedError(f"{type(self).__name__} does not support contact sensors")

    def _get_contact_sensor_prim_path(self, usd_path: str | None = None) -> str:
        """Return the root prim used for contact sensing by default."""
        return self.prim_path

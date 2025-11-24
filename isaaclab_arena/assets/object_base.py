# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg

from isaaclab_arena.assets.asset import Asset


class ObjectType(Enum):
    BASE = "base"
    RIGID = "rigid"
    ARTICULATION = "articulation"


class ObjectBase(Asset, ABC):
    """Parent class for (spawnable) Object and ObjectReference."""

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

    def set_prim_path(self, prim_path: str) -> None:
        self.prim_path = prim_path

    def get_prim_path(self) -> str:
        return self.prim_path

    def get_cfgs(self) -> dict[str, Any]:
        if self.object_type == ObjectType.RIGID:
            object_cfg = self._generate_rigid_cfg()
        elif self.object_type == ObjectType.ARTICULATION:
            object_cfg = self._generate_articulation_cfg()
        elif self.object_type == ObjectType.BASE:
            object_cfg = self._generate_base_cfg()
        else:
            raise ValueError(f"Invalid object type: {self.object_type}")
        return {
            self.name: object_cfg,
        }

    def get_object_pose(self, env: ManagerBasedEnv, is_relative: bool = True) -> torch.Tensor:
        """Get the pose of the object in the environment.

        Args:
            env: The environment.
            is_relative: Whether to return the pose in the relative frame of the environment.

        Returns:
            The pose of the object in each environment. The shape is (num_envs, 7).
            The order is (x, y, z, qw, qx, qy, qz).
        """
        # We require that the asset has been added to the scene under its name.
        assert self.name in env.scene.keys(), f"Asset {self.name} not found in scene"
        if (self.object_type == ObjectType.RIGID) or (self.object_type == ObjectType.ARTICULATION):
            object_pose = env.scene[self.name].data.root_pose_w.clone()
        elif self.object_type == ObjectType.BASE:
            object_pose = torch.cat(env.scene[self.name].get_world_poses(), dim=-1)
        else:
            raise ValueError(f"Function not implemented for object type: {self.object_type}")
        if is_relative:
            object_pose[:, :3] -= env.scene.env_origins
        return object_pose

    def get_contact_sensor_cfg(self, contact_against_prim_paths: list[str] | None = None) -> ContactSensorCfg:
        assert self.object_type == ObjectType.RIGID, "Contact sensor is only supported for rigid objects"
        if contact_against_prim_paths is None:
            contact_against_prim_paths = []
        return ContactSensorCfg(
            prim_path=self.prim_path,
            filter_prim_paths_expr=contact_against_prim_paths,
        )

    @abstractmethod
    def _generate_rigid_cfg(self) -> RigidObjectCfg:
        # Subclasses must implement this method
        pass

    @abstractmethod
    def _generate_articulation_cfg(self) -> ArticulationCfg:
        # Subclasses must implement this method
        pass

    @abstractmethod
    def _generate_base_cfg(self) -> AssetBaseCfg:
        # Subclasses must implement this method
        pass

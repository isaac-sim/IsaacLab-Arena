# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.assets.object_base import ObjectBase, ObjectType
from isaaclab.assets import RigidObjectCfg
import isaaclab.sim as sim_utils
from isaaclab_arena.assets.object import Object
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.assets.object_utils import detect_object_type

class RigidObjectSet(Object):
    """
    A set of rigid objects.
    """

    def __init__(self, name: str, objects: list[Object],
                 prim_path: str | None = "/World/envs/env_.*/Object",
                 scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
                 random_choice: bool = False,
                 initial_pose: Pose | None = None,
                 **kwargs):
        if not self._examine_objects_type_are_rigid(objects):
            raise ValueError(f"Object set {name} must contain at least 1 rigid object.")

        self.object_usd_paths = [object.usd_path for object in objects]
        self.random_choice = random_choice

        super().__init__(name=name,
                         object_type=ObjectType.RIGID,
                         usd_path = '',
                         prim_path=prim_path,
                         scale=scale,
                         initial_pose=initial_pose,
                         **kwargs)


    def _examine_objects_type_are_rigid(self, objects: list[ObjectBase]) -> bool:
        if objects is None or len(objects) == 0:
            return False
        for object in objects:
            if detect_object_type(usd_path=object.usd_path) != ObjectType.RIGID:
                return False
        return True

    def _generate_rigid_cfg(self) -> RigidObjectCfg:
        assert self.object_type == ObjectType.RIGID
        object_cfg = RigidObjectCfg(
                prim_path=self.prim_path,
                spawn=sim_utils.MultiUsdFileCfg(
                usd_path=self.object_usd_paths,
                random_choice=self.random_choice,
            ),
        )
        object_cfg = self._add_initial_pose_to_cfg(object_cfg)
        return object_cfg

    def _generate_articulation_cfg(self):
        raise NotImplementedError("Articulation configuration is not supported for object sets")

    def _generate_base_cfg(self):
        raise NotImplementedError("Base configuration is not supported for object sets")

    def _generate_spawner_cfg(self):
        raise NotImplementedError("Spawner configuration is not supported for object sets")
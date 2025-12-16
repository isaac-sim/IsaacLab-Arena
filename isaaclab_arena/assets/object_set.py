# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectBase, ObjectType
from isaaclab_arena.assets.object_utils import detect_object_type
from isaaclab_arena.utils.pose import Pose


class RigidObjectSet(Object):
    """
    A set of rigid objects.
    """

    def __init__(
        self,
        name: str,
        objects: list[Object],
        prim_path: str | None = None,
        scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
        random_choice: bool = False,
        initial_pose: Pose | None = None,
        **kwargs,
    ):
        """
        Args:
            name: The name of the object set.
            objects: The list of objects to be included in the object set.
            prim_path: The prim path of the object set. Note that for all environments, the object set
                prim path must be the same.
            scale: The scale of the object set. Note all objects can only have the same scale, if
                different scales are needed, considering scaling the object USD file.
            random_choice: Whether to randomly choose an object from the object set to spawn in
                each environment. If False, object is spawned based on the order of objects in the list.
            initial_pose: The initial pose of the object from this object set.
        """
        if not self._are_all_objects_type_rigid(objects):
            raise ValueError(f"Object set {name} must contain only rigid objects.")

        self.object_usd_paths = [object.usd_path for object in objects]
        self.random_choice = random_choice

        # Set default prim_path if not provided
        if prim_path is None:
            prim_path = f"{{ENV_REGEX_NS}}/{name}"

        super().__init__(
            name=name,
            object_type=ObjectType.RIGID,
            usd_path="",
            prim_path=prim_path,
            scale=scale,
            initial_pose=initial_pose,
            **kwargs,
        )

    def _are_all_objects_type_rigid(self, objects: list[ObjectBase]) -> bool:
        if objects is None or len(objects) == 0:
            raise ValueError(f"Object set {self.name} must contain at least 1 object.")
        return all(detect_object_type(usd_path=object.usd_path) == ObjectType.RIGID for object in objects)

    def _generate_rigid_cfg(self) -> RigidObjectCfg:
        assert self.object_type == ObjectType.RIGID
        object_cfg = RigidObjectCfg(
            prim_path=self.prim_path,
            spawn=sim_utils.MultiUsdFileCfg(
                usd_path=self.object_usd_paths,
                random_choice=self.random_choice,
                activate_contact_sensors=True,
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

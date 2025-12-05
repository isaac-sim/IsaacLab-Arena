# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

from isaaclab_arena.assets.object_base import ObjectBase, ObjectType
from isaaclab_arena.assets.object_utils import detect_object_type
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.usd_helpers import has_light, open_stage


class Object(ObjectBase):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    def __init__(
        self,
        name: str,
        prim_path: str | None = None,
        object_type: ObjectType | None = None,
        usd_path: str | None = None,
        scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
        initial_pose: Pose | None = None,
        **kwargs,
    ):
        if object_type is not ObjectType.SPAWNER:
            assert usd_path is not None
        # Detect object type if not provided
        if object_type is None:
            object_type = detect_object_type(usd_path=usd_path)
        super().__init__(name=name, prim_path=prim_path, object_type=object_type, **kwargs)
        self.usd_path = usd_path
        self.scale = scale
        self.initial_pose = initial_pose
        self.object_cfg = self._init_object_cfg()

    def set_initial_pose(self, pose: Pose) -> None:
        self.initial_pose = pose
        self.object_cfg = self._add_initial_pose_to_cfg(self.object_cfg)

    def get_initial_pose(self) -> Pose | None:
        return self.initial_pose

    def is_initial_pose_set(self) -> bool:
        return self.initial_pose is not None

    def _generate_rigid_cfg(self) -> RigidObjectCfg:
        assert self.object_type == ObjectType.RIGID
        object_cfg = RigidObjectCfg(
            prim_path=self.prim_path,
            spawn=UsdFileCfg(
                usd_path=self.usd_path,
                scale=self.scale,
                activate_contact_sensors=True,
            ),
        )
        object_cfg = self._add_initial_pose_to_cfg(object_cfg)
        return object_cfg

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        assert self.object_type == ObjectType.ARTICULATION
        object_cfg = ArticulationCfg(
            prim_path=self.prim_path,
            spawn=UsdFileCfg(
                usd_path=self.usd_path,
                scale=self.scale,
                activate_contact_sensors=True,
            ),
            actuators={},
        )
        object_cfg = self._add_initial_pose_to_cfg(object_cfg)
        return object_cfg

    def _generate_base_cfg(self) -> AssetBaseCfg:
        assert self.object_type == ObjectType.BASE
        with open_stage(self.usd_path) as stage:
            if has_light(stage):
                print("WARNING: Base object has lights, this may cause issues when using with multiple environments.")
        object_cfg = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/" + self.name,
            spawn=UsdFileCfg(usd_path=self.usd_path, scale=self.scale),
        )
        object_cfg = self._add_initial_pose_to_cfg(object_cfg)
        return object_cfg

    def _generate_spawner_cfg(self) -> AssetBaseCfg:
        assert self.object_type == ObjectType.SPAWNER
        object_cfg = AssetBaseCfg(
            prim_path=self.prim_path,
            spawn=self.spawner_cfg,
        )
        object_cfg = self._add_initial_pose_to_cfg(object_cfg)
        return object_cfg

    def _add_initial_pose_to_cfg(
        self, object_cfg: RigidObjectCfg | ArticulationCfg | AssetBaseCfg
    ) -> RigidObjectCfg | ArticulationCfg | AssetBaseCfg:
        # Optionally specify initial pose
        if self.initial_pose is not None:
            object_cfg.init_state.pos = self.initial_pose.position_xyz
            object_cfg.init_state.rot = self.initial_pose.rotation_wxyz
        return object_cfg

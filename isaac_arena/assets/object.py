# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

from isaac_arena.assets.object_base import ObjectBase, ObjectType
from isaac_arena.geometry.pose import Pose


class Object(ObjectBase):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    # Defined in Asset, restated here for clariry
    # tags: list[str] | None = None
    name: str | None = None
    object_type: ObjectType = ObjectType.RIGID
    usd_path: str | None = None
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __init__(
        self,
        initial_pose: Pose | None = None,
        **kwargs,
    ):
        assert self.name is not None and self.usd_path is not None
        super().__init__(name=self.name, object_type=self.object_type, **kwargs)
        self.initial_pose = initial_pose

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
        object_cfg = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/" + self.name,
            spawn=UsdFileCfg(usd_path=self.usd_path),
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

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


from pxr import Usd, UsdPhysics, UsdGeom, UsdSkel
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg

from isaac_arena.affordances.openable import Openable
from isaac_arena.assets.asset import Asset
from isaac_arena.assets.object_base import ObjectBase, ObjectType
from isaac_arena.utils.pose import Pose


class ObjectReference(ObjectBase):
    """An object which *refers* to an existing element in the scene"""

    def __init__(self, parent_asset: Asset, **kwargs):
        super().__init__(**kwargs)
        if parent_asset:
            self._check_path_in_parent_usd(parent_asset)
        self.parent_asset = parent_asset
        parent_stage = Usd.Stage.Open(parent_asset.usd_path)
        reference_prim = self._get_prim_by_name(parent_stage.GetPseudoRoot(), self.name)[0]
        if not reference_prim:
            raise ValueError(f"Reference prim {self.name} not found in parent USD file {parent_asset.usd_path}")
        reference_pos, reference_quat = self._get_prim_pos_rot_in_world(reference_prim)
        self.initial_pose = Pose(position_xyz=tuple(reference_pos), rotation_wxyz=tuple(reference_quat))

    def get_initial_pose(self) -> Pose:
        return self.initial_pose

    def get_contact_sensor_cfg(self, contact_against_prim_paths: list[str] | None = None) -> ContactSensorCfg:
        # NOTE(alexmillane): Right now this requires that the object
        # has the contact sensor enabled prior to using this reference.
        # At the moment, for the tests, I enabled the relevant APIs in the GUI.
        # TODO(alexmillane, 2025.09.08): Make the code automatically enable the
        # contact reporter API.
        # Just call out to the parent class method.
        return super().get_contact_sensor_cfg(contact_against_prim_paths)

    def _generate_rigid_cfg(self) -> RigidObjectCfg:
        assert self.object_type == ObjectType.RIGID
        object_cfg = RigidObjectCfg(
            prim_path=self.prim_path,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=self.initial_pose.position_xyz,
                rot=self.initial_pose.rotation_wxyz,
            ),
        )
        return object_cfg

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        assert self.object_type == ObjectType.ARTICULATION
        object_cfg = ArticulationCfg(
            prim_path=self.prim_path,
            actuators={},
            init_state=ArticulationCfg.InitialStateCfg(
                pos=self.initial_pose.position_xyz,
                rot=self.initial_pose.rotation_wxyz,
            ),
        )
        return object_cfg

    def _generate_base_cfg(self) -> AssetBaseCfg:
        assert self.object_type == ObjectType.BASE
        object_cfg = AssetBaseCfg(
            prim_path=self.prim_path,
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=self.initial_pose.position_xyz,
                rot=self.initial_pose.rotation_wxyz,
            ),
        )
        return object_cfg

    def _check_path_in_parent_usd(self, parent_asset: Asset) -> bool:
        # TODO(alexmillane, 2025.09.08): Implement this check!
        return True

    def _get_prim_pos_rot_in_world(self, prim):
        """Get prim position, rotation and scale in world coordinates"""
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            return None, None
        matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        try:
            pos, rot, _ = UsdSkel.DecomposeTransform(matrix)
            pos_list = list(pos)
            quat_list = [rot.GetReal(), rot.GetImaginary()[0], rot.GetImaginary()[1], rot.GetImaginary()[2]]  # wxyz
            return pos_list, quat_list
        except Exception as e:
            print(f"Error decomposing transform for {prim.GetName()}: {e}")
            return None, None

    def _get_prim_by_name(self, prim, name, only_xform=True):
        """Get prim by name"""
        result = []
        if prim.GetName().lower() == name.lower():
            if not only_xform or prim.GetTypeName() == "Xform":
                result.append(prim)
        for child in prim.GetAllChildren():
            result.extend(self._get_prim_by_name(child, name, only_xform))
        return result


class OpenableObjectReference(ObjectReference, Openable):
    """An object which *refers* to an existing element in the scene and is openable."""

    def __init__(self, openable_joint_name: str, openable_open_threshold: float = 0.5, **kwargs):
        super().__init__(
            openable_joint_name=openable_joint_name,
            openable_open_threshold=openable_open_threshold,
            object_type=ObjectType.ARTICULATION,
            **kwargs,
        )

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

from contextlib import contextmanager

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from pxr import Usd, UsdGeom, UsdSkel

from isaaclab_arena.affordances.openable import Openable
from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.object_base import ObjectBase, ObjectType
from isaaclab_arena.utils.pose import Pose


# Somewhat hacky way to open the stage and ensure it is closed after use.
@contextmanager
def open_stage(path):
    stage = Usd.Stage.Open(path)
    try:
        yield stage
    finally:
        # Drop the local reference; GC will reclaim once no prim/attr handles remain
        del stage


class ObjectReference(ObjectBase):
    """An object which *refers* to an existing element in the scene"""

    def __init__(self, parent_asset: Asset, **kwargs):
        super().__init__(**kwargs)
        # We open the stage as its the only way to get the initial pose of the reference prim.
        with open_stage(parent_asset.usd_path) as parent_stage:
            reference_prim = self._return_reference_prim_in_parent_usd(parent_asset, parent_stage)
            reference_pos, reference_quat = self._get_prim_pos_rot_in_world(reference_prim)
        self.parent_asset = parent_asset
        self.initial_pose = Pose(position_xyz=tuple(reference_pos), rotation_wxyz=tuple(reference_quat))

    def get_initial_pose(self) -> None:
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

    def _return_reference_prim_in_parent_usd(self, parent_asset: Asset, parent_stage: Usd.Stage) -> Usd.Prim:
        # TODO(Vik, 2025.10.17): Make this neater.
        # Currently, we take the last part of the prim path to find the prim in the scene.
        prim_name_in_scene = self.prim_path.split("/")[-1]
        reference_prims = []
        reference_prims = self._get_prim_by_name(parent_stage.GetPseudoRoot(), prim_name_in_scene)
        if len(reference_prims) == 0:
            raise ValueError(f"No prim found with name {prim_name_in_scene} in {parent_asset.usd_path}")
        # We return the first prim found.
        return reference_prims[0]

    def _get_prim_by_name(self, prim, name, only_xform=True):
        """Get prim by name"""
        result = []
        if prim.GetName().lower() == name.lower():
            if not only_xform or prim.GetTypeName() == "Xform":
                result.append(prim)
        for child in prim.GetAllChildren():
            result.extend(self._get_prim_by_name(child, name, only_xform))
        return result

    def _get_prim_pos_rot_in_world(self, prim: Usd.Prim) -> tuple[list[float], list[float]]:
        """Get prim position, rotation and scale in world coordinates"""
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            raise ValueError(f"Prim {prim.GetName()} is not a xformable")
        matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        try:
            pos, rot, _ = UsdSkel.DecomposeTransform(matrix)
            pos_list = list(pos)
            quat_list = [rot.GetReal(), rot.GetImaginary()[0], rot.GetImaginary()[1], rot.GetImaginary()[2]]  # wxyz
            return pos_list, quat_list
        except Exception as e:
            print(f"Error decomposing transform for {prim.GetName()}: {e}")
            raise ValueError(f"Error decomposing transform for {prim.GetName()}: {e}")


class OpenableObjectReference(ObjectReference, Openable):
    """An object which *refers* to an existing element in the scene and is openable."""

    def __init__(self, openable_joint_name: str, openable_open_threshold: float = 0.5, **kwargs):
        super().__init__(
            openable_joint_name=openable_joint_name,
            openable_open_threshold=openable_open_threshold,
            object_type=ObjectType.ARTICULATION,
            **kwargs,
        )

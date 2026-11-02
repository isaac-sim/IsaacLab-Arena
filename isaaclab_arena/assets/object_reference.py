# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import trimesh

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from pxr import Usd

from isaaclab_arena.affordances.openable import Openable
from isaaclab_arena.affordances.pressable import Pressable
from isaaclab_arena.affordances.turnable import Turnable
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectBase, RootedObjectBase
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.relations.relations import IsAnchor, RelationBase
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.usd.helpers import (
    NoCollisionMeshError,
    compute_world_aligned_bounding_box_relative_to_prim_origin,
    extract_trimesh_from_prim,
    open_stage,
)
from isaaclab_arena.utils.usd.pose import get_prim_pose_in_default_prim_frame


class ObjectReference(RootedObjectBase):
    """An object which *refers* to an existing element in the scene"""

    def __init__(self, parent_asset: Object, **kwargs):
        super().__init__(**kwargs)
        self.parent_asset = parent_asset
        self._parent_scale = parent_asset.scale
        # Resolve the path and pose together to avoid opening the parent USD stage multiple times.
        (
            self._prim_path_in_parent_usd,
            self.initial_pose_relative_to_parent,
        ) = self._get_referenced_prim_path_and_pose_relative_to_parent(parent_asset)
        self.object_cfg = self._init_object_cfg()
        self._pose_event_cfg = self._build_reset_event()
        self._bounding_box: AxisAlignedBoundingBox | None = None
        self._collision_mesh: trimesh.Trimesh | None = None
        # None is a valid cached result for meshless prims; this flag distinguishes that from not-yet-loaded.
        self._collision_mesh_loaded = False

    def get_initial_pose(self) -> Pose:
        """Return T_E_O for reference O, parent P and local environment frame E."""
        T_P_O = self.initial_pose_relative_to_parent
        T_E_P = self.get_parent_pose()
        T_E_O = T_E_P.multiply(T_P_O)
        return T_E_O

    def get_parent_pose(self) -> Pose:
        """Return the parent's fixed pose, using identity when no pose is configured."""
        pose = self.parent_asset.initial_pose
        assert pose is None or isinstance(pose, Pose), "ObjectReference requires a fixed parent pose"
        return pose if pose is not None else Pose.identity()

    @property
    def prim_path_in_parent_usd(self) -> str:
        """Return the referenced prim's absolute path in its parent USD stage."""
        return self._prim_path_in_parent_usd

    def add_relation(self, relation: RelationBase) -> None:
        """Add a relation to this object reference.

        ObjectReference only supports IsAnchor relations because the placement
        solver treats references as fixed points.

        Args:
            relation: Must be an IsAnchor relation.
        """
        assert isinstance(relation, IsAnchor), (
            f"ObjectReference only supports IsAnchor relations, got {type(relation).__name__}. "
            "The placement solver does not optimize ObjectReference positions."
        )
        self.relations.append(relation)

    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        """Get world-axis-aligned bounds measured from the referenced prim's origin.

        The coordinates use the parent asset's USD axes, with the origin shifted to
        the referenced prim's world position.

        The bounding box is computed lazily and cached for subsequent calls.
        """
        if self._bounding_box is None:
            with open_stage(self.parent_asset.usd_path) as parent_stage:
                prim_path_in_usd = self.isaaclab_prim_path_to_original_prim_path(
                    self.prim_path, self.parent_asset, parent_stage
                )
                raw_bbox = compute_world_aligned_bounding_box_relative_to_prim_origin(parent_stage, prim_path_in_usd)
                # Apply parent's scale (no centering - solver is origin-agnostic)
                self._bounding_box = raw_bbox.scaled(self._parent_scale)
        return self._bounding_box

    def get_bounding_box_rotation(self) -> tuple[float, float, float, float]:
        """Return the parent rotation; the prim's authored rotation is already included in its bounds."""
        return self.get_parent_pose().rotation_xyzw

    def get_collision_mesh(self) -> trimesh.Trimesh | None:
        """Return the referenced prim's collision mesh in its local frame, or None if unavailable."""
        if not self._collision_mesh_loaded:
            try:
                self._collision_mesh = self._extract_collision_mesh()
            except OSError as e:
                # Stage/file errors can be transient in Isaac Sim startup paths, so leave
                # _collision_mesh_loaded false and retry on the next call.
                print(f"Could not extract collision mesh for object reference '{self.name}': {e}")
                return None
            except NoCollisionMeshError as e:
                print(f"Could not extract collision mesh for object reference '{self.name}': {e}")
            self._collision_mesh_loaded = True
        return self._collision_mesh

    def _extract_collision_mesh(self) -> trimesh.Trimesh:
        """Extract the referenced prim mesh from the parent asset USD."""
        with open_stage(self.parent_asset.usd_path) as parent_stage:
            prim_path_in_usd = self.isaaclab_prim_path_to_original_prim_path(
                self.prim_path, self.parent_asset, parent_stage
            )
            if not parent_stage.GetPrimAtPath(prim_path_in_usd):
                raise ValueError(f"No prim found with path {prim_path_in_usd} in {self.parent_asset.usd_path}")
            return extract_trimesh_from_prim(parent_stage, prim_path_in_usd, self._parent_scale)

    def get_contact_sensor_cfg(self, contact_against_object: ObjectBase | None = None) -> ContactSensorCfg:
        # NOTE(alexmillane): Right now this requires that the object
        # has the contact sensor enabled prior to using this reference.
        # At the moment, for the tests, I enabled the relevant APIs in the GUI.
        # TODO(alexmillane, 2025.09.08): Make the code automatically enable the
        # contact reporter API.
        # NOTE(alexmillane, 2025.11.27): I've added a function for adding
        # the contact reporter API to a prim in a USD, perhaps that can be repurposed
        # and used here.
        # Just call out to the parent class method.
        return super().get_contact_sensor_cfg(contact_against_object)

    def _generate_rigid_cfg(self) -> RigidObjectCfg:
        assert self.object_type == ObjectType.RIGID
        initial_pose = self.get_initial_pose()
        object_cfg = RigidObjectCfg(
            prim_path=self.prim_path,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=initial_pose.position_xyz,
                rot=initial_pose.rotation_xyzw,
            ),
        )
        return object_cfg

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        assert self.object_type == ObjectType.ARTICULATION
        initial_pose = self.get_initial_pose()
        object_cfg = ArticulationCfg(
            prim_path=self.prim_path,
            actuators={},
            init_state=ArticulationCfg.InitialStateCfg(
                pos=initial_pose.position_xyz,
                rot=initial_pose.rotation_xyzw,
            ),
        )
        return object_cfg

    def _generate_base_cfg(self) -> AssetBaseCfg:
        assert self.object_type == ObjectType.BASE
        initial_pose = self.get_initial_pose()
        object_cfg = AssetBaseCfg(
            prim_path=self.prim_path,
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=initial_pose.position_xyz,
                rot=initial_pose.rotation_xyzw,
            ),
        )
        return object_cfg

    def _get_referenced_prim_path_and_pose_relative_to_parent(self, parent_asset: Object) -> tuple[str, Pose]:
        """Get the prim path and transform pose relative to the parent's default prim.

        The position is scaled by the parent's scale factor.
        """
        with open_stage(parent_asset.usd_path) as parent_stage:
            prim_path_in_usd = self.isaaclab_prim_path_to_original_prim_path(self.prim_path, parent_asset, parent_stage)
            prim = parent_stage.GetPrimAtPath(prim_path_in_usd)
            if not prim:
                raise ValueError(f"No prim found with path {prim_path_in_usd} in {parent_asset.usd_path}")
            prim_pose = get_prim_pose_in_default_prim_frame(prim, parent_stage)
            # Apply parent's scale to the position
            scaled_pos = (
                prim_pose.position_xyz[0] * self._parent_scale[0],
                prim_pose.position_xyz[1] * self._parent_scale[1],
                prim_pose.position_xyz[2] * self._parent_scale[2],
            )
            return prim_path_in_usd, Pose(position_xyz=scaled_pos, rotation_xyzw=prim_pose.rotation_xyzw)

    @staticmethod
    def isaaclab_prim_path_to_original_prim_path(
        isaaclab_prim_path: str, parent_asset: Object, stage: Usd.Stage
    ) -> str:
        """Map a runtime path beneath the parent asset to its source USD stage.

        Args:
            isaaclab_prim_path: The runtime prim path of the reference.
            parent_asset: Asset whose configured prim path prefixes the reference.
            stage: The parent asset's opened USD stage.

        Returns:
            The same relative path beneath the source stage's default prim.
        """
        default_prim = stage.GetDefaultPrim()
        assert default_prim.IsValid(), "Parent USD must have a default prim"
        parent_path = parent_asset.get_prim_path().rstrip("/")
        assert isaaclab_prim_path == parent_path or isaaclab_prim_path.startswith(
            parent_path + "/"
        ), f"Reference path '{isaaclab_prim_path}' must be beneath parent path '{parent_path}'"
        relative_path = isaaclab_prim_path.removeprefix(parent_path)
        return str(default_prim.GetPath()) + relative_path


class OpenableObjectReference(ObjectReference, Openable):
    """An object which *refers* to an existing element in the scene and is openable."""

    def __init__(self, openable_joint_name: str, openable_threshold: float = 0.5, **kwargs):
        super().__init__(
            openable_joint_name=openable_joint_name,
            openable_threshold=openable_threshold,
            object_type=ObjectType.ARTICULATION,
            **kwargs,
        )


class PressableObjectReference(ObjectReference, Pressable):
    """A referenced articulation exposing one prismatic joint as a button."""

    def __init__(self, pressable_joint_name: str, pressedness_threshold: float = 0.5, **kwargs):
        super().__init__(
            pressable_joint_name=pressable_joint_name,
            pressedness_threshold=pressedness_threshold,
            object_type=ObjectType.ARTICULATION,
            **kwargs,
        )


class TurnableObjectReference(ObjectReference, Turnable):
    """A referenced articulation exposing one revolute joint as a discrete control."""

    def __init__(
        self,
        turnable_joint_name: str,
        min_level_angle_deg: float,
        max_level_angle_deg: float,
        num_levels: int,
        **kwargs,
    ):
        super().__init__(
            turnable_joint_name=turnable_joint_name,
            min_level_angle_deg=min_level_angle_deg,
            max_level_angle_deg=max_level_angle_deg,
            num_levels=num_levels,
            object_type=ObjectType.ARTICULATION,
            **kwargs,
        )

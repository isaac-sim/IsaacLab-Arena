# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import trimesh

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from pxr import Usd

from isaaclab_arena.affordances.openable import Openable
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectBase, ObjectType
from isaaclab_arena.relations.relations import IsAnchor, RelationBase
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox, quaternion_to_90_deg_z_quarters
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.usd_helpers import (
    NoCollisionMeshError,
    compute_local_bounding_box_from_prim,
    extract_trimesh_from_prim,
    open_stage,
)
from isaaclab_arena.utils.usd_pose_helpers import get_prim_pose_in_default_prim_frame


class ObjectReference(ObjectBase):
    """An object which *refers* to an existing element in the scene"""

    def __init__(self, parent_asset: Object, **kwargs):
        super().__init__(**kwargs)
        self.parent_asset = parent_asset
        self._parent_scale = parent_asset.scale
        # Get the prim's transform pose (not geometry center - solver is origin-agnostic)
        self.initial_pose_relative_to_parent = self._get_referenced_prim_pose_relative_to_parent(parent_asset)
        self.object_cfg = self._init_object_cfg()
        self._bounding_box: AxisAlignedBoundingBox | None = None
        self._collision_mesh: trimesh.Trimesh | None = None
        # None is a valid cached result for meshless prims; this flag distinguishes that from not-yet-loaded.
        self._collision_mesh_loaded = False

    def get_initial_pose(self) -> Pose:
        if self.parent_asset.initial_pose is None:
            T_W_O = self.initial_pose_relative_to_parent
        else:
            T_P_O = self.initial_pose_relative_to_parent
            T_W_P = self.parent_asset.initial_pose
            T_W_O = T_W_P.multiply(T_P_O)
        return T_W_O

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
        """Get local bounding box of the referenced prim (relative to prim transform).

        The bounding box is relative to the prim's transform origin, consistent with
        how Object.get_bounding_box() returns bbox relative to USD origin.

        The bounding box is computed lazily and cached for subsequent calls.
        """
        if self._bounding_box is None:
            with open_stage(self.parent_asset.usd_path) as parent_stage:
                prim_path_in_usd = self.isaaclab_prim_path_to_original_prim_path(
                    self.prim_path, self.parent_asset, parent_stage
                )
                raw_bbox = compute_local_bounding_box_from_prim(parent_stage, prim_path_in_usd)
                # Apply parent's scale (no centering - solver is origin-agnostic)
                self._bounding_box = raw_bbox.scaled(self._parent_scale)
        return self._bounding_box

    def get_world_bounding_box(self) -> AxisAlignedBoundingBox:
        """Bounding box in world coordinates.

        get_bounding_box() is already axis-aligned in the parent's frame, so only the parent's
        placement rotation (identity or a 90° Z multiple) and the prim's world position are applied.
        """
        box = self.get_bounding_box()
        world_position = self.get_initial_pose().position_xyz
        parent_pose = self.parent_asset.initial_pose
        if parent_pose is None:
            return box.translated(world_position)
        quarters = quaternion_to_90_deg_z_quarters(parent_pose.rotation_xyzw)
        return box.rotated_90_around_z(quarters).translated(world_position)

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

    def _get_referenced_prim_pose_relative_to_parent(self, parent_asset: Object) -> Pose:
        """Get the prim's transform pose relative to the parent's default prim.

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
            return Pose(position_xyz=scaled_pos, rotation_xyzw=prim_pose.rotation_xyzw)

    @staticmethod
    def isaaclab_prim_path_to_original_prim_path(
        isaaclab_prim_path: str, parent_asset: Object, stage: Usd.Stage
    ) -> str:
        """Convert an IsaacLab prim path to the prim path in the original USD stage.

        Two steps to getting the original prim path from the IsaacLab prim path.

        # 1. Remove the ENV_REGEX_NS prefix
        # 2. Replace the asset name with the default prim path.

        Args:
            isaaclab_prim_path: The IsaacLab prim path.
            parent_asset: The asset the prim belongs to; its name is stripped from the path.
            stage: The parent asset's opened USD stage, used to resolve the default prim.

        Returns:
            The prim path in the original USD stage.
        """
        default_prim = stage.GetDefaultPrim()
        default_prim_path = default_prim.GetPath()
        assert default_prim_path is not None
        # Check that the path starts with the ENV_REGEX_NS prefix.
        assert isaaclab_prim_path.startswith("{ENV_REGEX_NS}/")
        original_prim_path = isaaclab_prim_path.removeprefix("{ENV_REGEX_NS}/")
        # Check that the path starts with the asset name.
        assert original_prim_path.startswith(parent_asset.name), (
            "Expected the prim path to start with the parent asset name {parent_asset.name}. Instead got"
            " {original_prim_path}"
        )
        original_prim_path = original_prim_path.removeprefix(parent_asset.name)
        # Append the default prim path.
        original_prim_path = str(default_prim_path) + original_prim_path
        return original_prim_path


class OpenableObjectReference(ObjectReference, Openable):
    """An object which *refers* to an existing element in the scene and is openable."""

    def __init__(self, openable_joint_name: str, openable_threshold: float = 0.5, **kwargs):
        super().__init__(
            openable_joint_name=openable_joint_name,
            openable_threshold=openable_threshold,
            object_type=ObjectType.ARTICULATION,
            **kwargs,
        )

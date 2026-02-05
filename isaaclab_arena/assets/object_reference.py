# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from pxr import Usd, UsdPhysics

from isaaclab_arena.affordances.openable import Openable
from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.object_base import ObjectBase, ObjectType
from isaaclab_arena.relations.relations import IsAnchor, RelationBase
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.usd_helpers import compute_local_bounding_box_from_prim, open_stage
from isaaclab_arena.utils.usd_pose_helpers import get_prim_pose_in_default_prim_frame


class ObjectReference(ObjectBase):
    """An object which *refers* to an existing element in the scene"""

    def __init__(self, parent_asset: Asset, **kwargs):
        super().__init__(**kwargs)
        self.parent_asset = parent_asset
        # Store parent's scale for bounding box calculations
        self._parent_scale = getattr(parent_asset, "scale", (1.0, 1.0, 1.0))
        # Get the prim's transform pose (not geometry center - solver is origin-agnostic)
        self.initial_pose_relative_to_parent = self._get_referenced_prim_pose_relative_to_parent(parent_asset)
        assert self.object_type != ObjectType.SPAWNER, "Object reference cannot be a spawner"
        self.object_cfg = self._init_object_cfg()
        self._bounding_box: AxisAlignedBoundingBox | None = None

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
        """Get bounding box in world coordinates (local bbox + world position)."""
        return self.get_bounding_box().translated(self.get_initial_pose().position_xyz)

    def get_contact_sensor_cfg(self, contact_against_prim_paths: list[str] | None = None) -> ContactSensorCfg:
        # NOTE(alexmillane): Right now this requires that the object
        # has the contact sensor enabled prior to using this reference.
        # At the moment, for the tests, I enabled the relevant APIs in the GUI.
        # TODO(alexmillane, 2025.09.08): Make the code automatically enable the
        # contact reporter API.
        # NOTE(alexmillane, 2025.11.27): I've added a function for adding
        # the contact reporter API to a prim in a USD, perhaps that can be repurposed
        # and used here.
        # Just call out to the parent class method.
        return super().get_contact_sensor_cfg(contact_against_prim_paths)

    def _generate_rigid_cfg(self) -> RigidObjectCfg:
        assert self.object_type == ObjectType.RIGID
        # Validate that the referenced prim has RigidBodyAPI applied
        with open_stage(self.parent_asset.usd_path) as parent_stage:
            prim_path_in_usd = self.isaaclab_prim_path_to_original_prim_path(
                self.prim_path, self.parent_asset, parent_stage
            )
            prim = parent_stage.GetPrimAtPath(prim_path_in_usd)
            if prim and not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError(
                    f"ObjectReference '{self.name}' has object_type=RIGID but the prim "
                    f"'{prim_path_in_usd}' does not have USD RigidBodyAPI applied. "
                    "Consider using object_type=BASE for static prims that are part of a background asset."
                )
        initial_pose = self.get_initial_pose()
        object_cfg = RigidObjectCfg(
            prim_path=self.prim_path,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=initial_pose.position_xyz,
                rot=initial_pose.rotation_wxyz,
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
                rot=initial_pose.rotation_wxyz,
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
                rot=initial_pose.rotation_wxyz,
            ),
        )
        return object_cfg

    def _get_referenced_prim_pose_relative_to_parent(self, parent_asset: Asset) -> Pose:
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
            return Pose(position_xyz=scaled_pos, rotation_wxyz=prim_pose.rotation_wxyz)

    def isaaclab_prim_path_to_original_prim_path(
        self, isaaclab_prim_path: str, parent_asset: Asset, stage: Usd.Stage
    ) -> str:
        """Convert an IsaacLab prim path to the prim path in the original USD stage.

        Two steps to getting the original prim path from the IsaacLab prim path.

        # 1. Remove the ENV_REGEX_NS prefix
        # 2. Replace the asset name with the default prim path.

        Args:
            isaaclab_prim_path: The IsaacLab prim path.

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
        assert original_prim_path.startswith(parent_asset.name)
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

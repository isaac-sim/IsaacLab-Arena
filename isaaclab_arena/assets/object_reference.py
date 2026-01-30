# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg

from isaaclab_arena.affordances.openable import Openable
from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.object_base import ObjectBase, ObjectType
from isaaclab_arena.relations.relations import IsAnchor, RelationBase
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.usd_helpers import open_stage, read_prim_bounding_box, read_prim_pose_in_default_prim_frame


class ObjectReference(ObjectBase):
    """An object which *refers* to an existing element in the scene"""

    def __init__(self, parent_asset: Asset, **kwargs):
        super().__init__(**kwargs)
        self.parent_asset = parent_asset
        # Store parent's scale for bounding box calculations
        self._parent_scale = getattr(parent_asset, "scale", (1.0, 1.0, 1.0))
        assert self.object_type != ObjectType.SPAWNER, "Object reference cannot be a spawner"
        self.object_cfg = self._init_object_cfg()
        self._bounding_box: AxisAlignedBoundingBox | None = None

    def _get_usd_prim_path(self) -> str:
        """Convert IsaacLab prim path to the original USD prim path.

        IsaacLab uses paths like "{ENV_REGEX_NS}/asset_name/prim/path" at runtime,
        but the USD file uses paths like "/default_prim/prim/path".
        """
        with open_stage(self.parent_asset.usd_path) as stage:
            default_prim = stage.GetDefaultPrim()
            default_prim_path = default_prim.GetPath()
            assert default_prim_path is not None

            # Remove the {ENV_REGEX_NS}/ prefix
            assert self.prim_path.startswith("{ENV_REGEX_NS}/")
            path = self.prim_path.removeprefix("{ENV_REGEX_NS}/")

            # Remove the asset name prefix
            assert path.startswith(self.parent_asset.name)
            path = path.removeprefix(self.parent_asset.name)

            # Prepend the default prim path
            return str(default_prim_path) + path

    def get_initial_pose(self) -> Pose:
        usd_prim_path = self._get_usd_prim_path()
        T_P_O = read_prim_pose_in_default_prim_frame(
            self.parent_asset.usd_path,
            usd_prim_path,
            self._parent_scale,
        )
        if self.parent_asset.initial_pose is None:
            return T_P_O
        T_W_P = self.parent_asset.initial_pose
        return T_W_P.multiply(T_P_O)

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
            usd_prim_path = self._get_usd_prim_path()
            self._bounding_box = read_prim_bounding_box(
                self.parent_asset.usd_path,
                usd_prim_path,
                self._parent_scale,
            )
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


class OpenableObjectReference(ObjectReference, Openable):
    """An object which *refers* to an existing element in the scene and is openable."""

    def __init__(self, openable_joint_name: str, openable_threshold: float = 0.5, **kwargs):
        super().__init__(
            openable_joint_name=openable_joint_name,
            openable_threshold=openable_threshold,
            object_type=ObjectType.ARTICULATION,
            **kwargs,
        )

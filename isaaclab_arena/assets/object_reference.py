# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import trimesh

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from isaaclab.utils.math import matrix_from_quat
from pxr import Usd, UsdGeom

from isaaclab_arena.affordances.openable import Openable
from isaaclab_arena.affordances.pressable import Pressable
from isaaclab_arena.affordances.turnable import Turnable
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectBase, RootedObjectBase
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.relations.relations import IsAnchor, RelationBase
from isaaclab_arena.terms.events import reset_articulation_pose_and_joints
from isaaclab_arena.utils.bounding_box import OrientedBoundingBox
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.usd_helpers import (
    NoCollisionMeshError,
    compute_local_bounding_box_from_prim,
    extract_trimesh_from_prim,
    open_stage,
)
from isaaclab_arena.utils.usd_pose_helpers import get_prim_pose_in_default_prim_frame


class ObjectReference(RootedObjectBase):
    """An object which *refers* to an existing element in the scene"""

    def __init__(self, parent_asset: Object, **kwargs):
        parent_scale = parent_asset.scale
        assert all(
            component > 0 for component in parent_scale
        ), f"ObjectReference parent scale must be positive, got {parent_scale}."
        super().__init__(**kwargs)
        self.parent_asset = parent_asset
        self._parent_scale = parent_scale
        # Resolve the path and pose together to avoid opening the parent USD stage multiple times.
        (
            self._prim_path_in_parent_usd,
            self.initial_pose_relative_to_parent,
        ) = self._get_referenced_prim_path_and_pose_relative_to_parent(parent_asset)
        self.object_cfg = self._init_object_cfg()
        self._pose_event_cfg = self._build_reset_event()
        self._bounding_box: OrientedBoundingBox | None = None
        self._collision_mesh: trimesh.Trimesh | None = None
        # None is a valid cached result for meshless prims; this flag distinguishes that from not-yet-loaded.
        self._collision_mesh_loaded = False

    def _build_reset_event(self):
        """Build a complete reset event for a referenced rigid body or articulation."""
        event_cfg = super()._build_reset_event()
        if event_cfg is not None and self.object_type == ObjectType.ARTICULATION:
            event_cfg.func = reset_articulation_pose_and_joints
        return event_cfg

    def get_initial_pose(self) -> Pose:
        if self.parent_asset.initial_pose is None:
            T_W_O = self.initial_pose_relative_to_parent
        else:
            T_P_O = self.initial_pose_relative_to_parent
            T_W_P = self.parent_asset.initial_pose
            T_W_O = T_W_P.multiply(T_P_O)
        return T_W_O

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

    def get_bounding_box(self) -> OrientedBoundingBox:
        """Get local bounding box of the referenced prim (relative to prim transform).

        Coordinates are expressed in the referenced prim's local frame, with
        the parent asset's spawn scale included.

        The bounding box is computed lazily and cached for subsequent calls.
        """
        if self._bounding_box is None:
            with open_stage(self.parent_asset.usd_path) as parent_stage:
                prim_path_in_usd = self.isaaclab_prim_path_to_original_prim_path(
                    self.prim_path, self.parent_asset, parent_stage
                )
                raw_bbox = compute_local_bounding_box_from_prim(parent_stage, prim_path_in_usd)
                scaled_corners = self._transform_raw_local_points(
                    raw_bbox.get_corners(), parent_stage, prim_path_in_usd
                )
                self._bounding_box = OrientedBoundingBox.from_min_max(
                    min_point=scaled_corners.amin(dim=1),
                    max_point=scaled_corners.amax(dim=1),
                )
        return self._bounding_box

    def get_world_bounding_box(self) -> OrientedBoundingBox:
        """Return the referenced prim's bounding box in world coordinates."""
        box = self.get_bounding_box()
        world_pose = self.get_initial_pose()
        return box.transformed(world_pose.position_xyz, world_pose.rotation_xyzw)

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
            mesh = extract_trimesh_from_prim(parent_stage, prim_path_in_usd, (1.0, 1.0, 1.0))
            vertices = torch.as_tensor(mesh.vertices, dtype=torch.float64)
            mesh.vertices = self._transform_raw_local_points(vertices, parent_stage, prim_path_in_usd).numpy()
            return mesh

    def _transform_raw_local_points(self, points: torch.Tensor, stage: Usd.Stage, prim_path: str) -> torch.Tensor:
        """Include authored and spawn scale in the reference's rigid local frame."""
        # P is the parent default-prim frame, O the referenced prim frame. The
        # full affine transform includes scale/shear omitted from the rigid pose.
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        T_P_O = (
            cache.GetLocalToWorldTransform(stage.GetPrimAtPath(prim_path))
            * cache.GetLocalToWorldTransform(stage.GetDefaultPrim()).GetInverse()
        )
        linear_P_O = points.new_tensor([list(T_P_O[row])[:3] for row in range(3)])
        rotation = matrix_from_quat(points.new_tensor(self.initial_pose_relative_to_parent.rotation_xyzw).unsqueeze(0))[
            0
        ]
        scale = points.new_tensor(self._parent_scale)
        return ((points @ linear_P_O) * scale) @ rotation

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

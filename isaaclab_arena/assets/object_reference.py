# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import trimesh
from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from isaaclab.utils.configclass import configclass
from pxr import Gf, Sdf, Usd, UsdPhysics

from isaaclab_arena.affordances.openable import Openable
from isaaclab_arena.affordances.pressable import Pressable
from isaaclab_arena.affordances.turnable import Turnable
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectBase, RootedObjectBase
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.relations.relations import IsAnchor, RelationBase
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.usd.articulation import (
    adjust_prismatic_joint_positions_for_spawn_scale,
    articulation_joint_prims,
    compute_posed_prim_world_deltas,
    resolve_joint_pos_patterns,
)
from isaaclab_arena.utils.usd.helpers import (
    NoCollisionMeshError,
    compute_world_aligned_bounding_box_relative_to_prim_origin,
    extract_trimesh_from_prim,
)
from isaaclab_arena.utils.usd.pose import get_prim_pose_in_default_prim_frame


@configclass
class ArticulationBodyReferenceCfg(AssetBaseCfg):
    """Configure a scene extra whose live pose is the link pose of an articulation body."""

    articulation_name: str = MISSING
    """Scene key of the articulation that owns the body."""

    body_name: str = MISSING
    """Name of the body within the articulation."""


class ObjectReference(RootedObjectBase):
    """An object which *refers* to an existing element in the scene"""

    def __init__(self, parent_asset: Object | ObjectReference, **kwargs):
        super().__init__(**kwargs)
        if isinstance(parent_asset, ObjectReference):
            assert parent_asset.object_type == ObjectType.ARTICULATION, (
                "Nested object references currently require an articulation reference parent, "
                f"got {parent_asset.object_type} for '{parent_asset.name}'."
            )
        self.parent_asset = parent_asset
        self._parent_scale = parent_asset.scale
        # Resolve the path, link motion and pose together to avoid opening the parent USD stage multiple times.
        # The link motion is None unless the reference is a BASE reference to a body of an articulated parent.
        (
            self._prim_path_in_parent_usd,
            self._initial_link_world_delta,
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
        if isinstance(self.parent_asset, ObjectReference):
            return self.parent_asset.get_initial_pose()
        pose = self.parent_asset.initial_pose
        assert pose is None or isinstance(pose, Pose), "ObjectReference requires a fixed parent pose"
        return pose if pose is not None else Pose.identity()

    @property
    def scale(self) -> tuple[float, float, float]:
        """Return the scale inherited from the source asset."""
        return self._parent_scale

    @property
    def root_asset(self) -> Object:
        """Return the spawned asset at the top of the reference chain."""
        if isinstance(self.parent_asset, ObjectReference):
            return self.parent_asset.root_asset
        return self.parent_asset

    @property
    def usd_path(self) -> str:
        """Return the source USD path inherited from the root asset."""
        return self.parent_asset.usd_path

    def open_spawned_usd_stage(self) -> Usd.Stage:
        """Open the root parent's spawned USD stage."""
        return self.parent_asset.open_spawned_usd_stage()

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
        the referenced prim's world position. A referenced articulation link is
        measured at the parent's initial joint positions.

        The bounding box is computed lazily and cached for subsequent calls.
        """
        if self._bounding_box is None:
            parent_stage = self.parent_asset.open_spawned_usd_stage()
            prim_path_in_stage = self.isaaclab_prim_path_to_original_prim_path(
                self.prim_path, self.parent_asset, parent_stage
            )
            raw_bbox = compute_world_aligned_bounding_box_relative_to_prim_origin(parent_stage, prim_path_in_stage)
            if self._initial_link_world_delta is not None:
                # The bounds are measured from the link origin, which carries the motion's translation.
                q_link = Gf.Matrix4d(self._initial_link_world_delta.tolist()).ExtractRotationQuat()
                q_link_xyzw = (*q_link.GetImaginary(), q_link.GetReal())
                raw_bbox = raw_bbox.rotated_by_quat(q_link_xyzw)
            # Apply parent's scale (no centering - solver is origin-agnostic)
            self._bounding_box = raw_bbox.scaled(self._parent_scale)
        return self._bounding_box

    def get_bounding_box_rotation(self) -> tuple[float, float, float, float]:
        """Return the root asset's rotation; authored prim rotations are already included in the bounds."""
        if isinstance(self.parent_asset, ObjectReference):
            return self.parent_asset.get_bounding_box_rotation()
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
        parent_stage = self.parent_asset.open_spawned_usd_stage()
        prim_path_in_stage = self.isaaclab_prim_path_to_original_prim_path(
            self.prim_path, self.parent_asset, parent_stage
        )
        if not parent_stage.GetPrimAtPath(prim_path_in_stage):
            raise ValueError(
                f"No prim found with path {prim_path_in_stage} in the stage spawned from {self.parent_asset.usd_path}"
            )
        return extract_trimesh_from_prim(parent_stage, prim_path_in_stage, self._parent_scale)

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
        init_state = AssetBaseCfg.InitialStateCfg(pos=initial_pose.position_xyz, rot=initial_pose.rotation_xyzw)
        if self._initial_link_world_delta is None:
            return AssetBaseCfg(prim_path=self.prim_path, init_state=init_state)
        return ArticulationBodyReferenceCfg(
            prim_path=self.prim_path,
            init_state=init_state,
            articulation_name=self.parent_asset.name,
            body_name=Sdf.Path(self._prim_path_in_parent_usd).name,
        )

    def _get_referenced_prim_path_and_pose_relative_to_parent(
        self, parent_asset: Object | ObjectReference
    ) -> tuple[str, np.ndarray | None, Pose]:
        """Get the prim path, initial link motion and pose relative to the parent's default prim.

        A referenced articulation link is posed at the parent's initial joint positions.
        The position is scaled by the parent's scale factor. The returned path is the prim's path in the parent's
        USD file.
        """
        # The spawned stage selects the parent's variants, behind which physics schemas such as a SimReady asset's
        # can sit.
        parent_stage = parent_asset.open_spawned_usd_stage()
        prim_path_in_stage = self.isaaclab_prim_path_to_original_prim_path(self.prim_path, parent_asset, parent_stage)
        prim_path_in_usd = _get_prim_path_in_referenced_usd(parent_stage, prim_path_in_stage)
        prim = parent_stage.GetPrimAtPath(prim_path_in_stage)
        if not prim:
            raise ValueError(f"No prim found with path {prim_path_in_usd} in {parent_asset.usd_path}")
        initial_link_world_delta = self._get_initial_link_world_delta(parent_asset, parent_stage, prim)
        T_D_O = get_prim_pose_in_default_prim_frame(prim, parent_stage, initial_link_world_delta)
        T_D_O = Pose(
            position_xyz=tuple(
                position * scale for position, scale in zip(T_D_O.position_xyz, self._parent_scale, strict=True)
            ),
            rotation_xyzw=T_D_O.rotation_xyzw,
        )
        if isinstance(parent_asset, ObjectReference):
            parent_prim_path = self.isaaclab_prim_path_to_original_prim_path(
                parent_asset.prim_path, parent_asset, parent_stage
            )
            parent_prim = parent_stage.GetPrimAtPath(parent_prim_path)
            T_D_P = get_prim_pose_in_default_prim_frame(parent_prim, parent_stage)
            T_D_P = Pose(
                position_xyz=tuple(
                    position * scale for position, scale in zip(T_D_P.position_xyz, self._parent_scale, strict=True)
                ),
                rotation_xyzw=T_D_P.rotation_xyzw,
            )
            T_D_O = T_D_O.relative_to(T_D_P)
        return (
            prim_path_in_usd,
            initial_link_world_delta,
            T_D_O,
        )

    def _get_initial_link_world_delta(
        self, parent_asset: Object | ObjectReference, parent_stage: Usd.Stage, prim: Usd.Prim
    ) -> np.ndarray | None:
        """Return the stage-world motion of a referenced link at the parent's initial joint positions.

        Args:
            parent_asset: Asset whose USD contains the referenced prim.
            parent_stage: The parent asset's opened USD stage.
            prim: The referenced prim.

        Returns:
            Row-vector 4x4 delta, as returned by compute_posed_prim_world_deltas, for a BASE reference to a rigid
            body of an articulated parent. The identity when no revolute or prismatic joint drives the body, or when
            every joint is at zero. None for every other reference.
        """
        if not (
            self.object_type == ObjectType.BASE
            and parent_asset.object_type == ObjectType.ARTICULATION
            and prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ):
            return None
        articulation_prim = parent_stage.GetDefaultPrim()
        if isinstance(parent_asset, ObjectReference):
            articulation_prim_path = self.isaaclab_prim_path_to_original_prim_path(
                parent_asset.prim_path, parent_asset, parent_stage
            )
            articulation_prim = parent_stage.GetPrimAtPath(articulation_prim_path)
        # Find the body's joint before reading joint names, so a link that no joint moves builds even when the
        # parent repeats a joint name. Joints may target a mesh beneath the body, so match each child target by its
        # rigid body.
        parent_joint_child_paths = []
        for joint_prim in Usd.PrimRange(articulation_prim):
            if not joint_prim.IsA(UsdPhysics.RevoluteJoint) and not joint_prim.IsA(UsdPhysics.PrismaticJoint):
                continue
            for child_path in UsdPhysics.Joint(joint_prim).GetBody1Rel().GetTargets():
                if _find_owning_rigid_body_prim(parent_stage.GetPrimAtPath(child_path)) == prim:
                    parent_joint_child_paths.append(child_path.pathString)
        assert len(parent_joint_child_paths) <= 1, f"Body {prim.GetPath()} is the child of several movable joints."
        if not parent_joint_child_paths:
            return np.eye(4)
        joint_prims = articulation_joint_prims(articulation_prim)
        # The joint positions are read once, here, so the parent's init_state.joint_pos must be set before the
        # reference is created.
        joint_pos = resolve_joint_pos_patterns(joint_prims, parent_asset.object_cfg.init_state.joint_pos)
        if not any(joint_pos.values()):
            # With every joint at zero, the authored pose is already right.
            return np.eye(4)
        usd_joint_pos = adjust_prismatic_joint_positions_for_spawn_scale(joint_prims, joint_pos, self._parent_scale)
        body_world_deltas = compute_posed_prim_world_deltas(
            parent_stage, articulation_prim.GetPath().pathString, usd_joint_pos
        )
        link_world_delta = body_world_deltas[parent_joint_child_paths[0]]
        # Posing treats joint targets as unscaled frames, so a target scaled relative to its body moves non-rigidly.
        assert np.allclose(
            link_world_delta[:3, :3] @ link_world_delta[:3, :3].T, np.eye(3), atol=1e-5
        ), f"Cannot pose body {prim.GetPath()}: a joint targets a prim scaled relative to its body."
        return link_world_delta

    @staticmethod
    def isaaclab_prim_path_to_original_prim_path(
        isaaclab_prim_path: str, parent_asset: Object | ObjectReference, stage: Usd.Stage
    ) -> str:
        """Map a runtime path beneath the parent asset to a stage opened from the parent's USD.

        Args:
            isaaclab_prim_path: The runtime prim path of the reference.
            parent_asset: Asset whose configured prim path prefixes the reference.
            stage: A stage opened from the parent asset's USD, such as its spawned stage.

        Returns:
            The same relative path beneath the stage's default prim.
        """
        default_prim = stage.GetDefaultPrim()
        assert default_prim.IsValid(), "Parent USD must have a default prim"
        parent_path = parent_asset.get_prim_path().rstrip("/")
        assert isaaclab_prim_path == parent_path or isaaclab_prim_path.startswith(
            parent_path + "/"
        ), f"Reference path '{isaaclab_prim_path}' must be beneath parent path '{parent_path}'"
        if isinstance(parent_asset, ObjectReference):
            return ObjectReference.isaaclab_prim_path_to_original_prim_path(
                isaaclab_prim_path, parent_asset.parent_asset, stage
            )
        relative_path = isaaclab_prim_path.removeprefix(parent_path)
        return str(default_prim.GetPath()) + relative_path


class ArticulationObjectReference(ObjectReference):
    """Refer to an existing articulation with optional initial joint positions."""

    def __init__(self, initial_joint_positions: dict[str, float] | None = None, **kwargs):
        """Set joint defaults in radians for revolute joints and metres for prismatic joints.

        Args:
            initial_joint_positions: Joint names or Isaac Lab regex patterns mapped to positions.
                None preserves Isaac Lab's defaults; an empty dict leaves all joints unspecified.
            **kwargs: ObjectReference arguments other than object_type, which is always ARTICULATION.
        """
        super().__init__(object_type=ObjectType.ARTICULATION, **kwargs)
        if initial_joint_positions is not None:
            assert isinstance(self.object_cfg, ArticulationCfg)
            self.object_cfg.init_state.joint_pos = dict(initial_joint_positions)


class OpenableObjectReference(ArticulationObjectReference, Openable):
    """An object which *refers* to an existing element in the scene and is openable."""

    def __init__(self, openable_joint_name: str, openable_threshold: float = 0.5, **kwargs):
        super().__init__(
            openable_joint_name=openable_joint_name,
            openable_threshold=openable_threshold,
            **kwargs,
        )


class PressableObjectReference(ArticulationObjectReference, Pressable):
    """A referenced articulation exposing one prismatic joint as a button."""

    def __init__(self, pressable_joint_name: str, pressedness_threshold: float = 0.5, **kwargs):
        super().__init__(
            pressable_joint_name=pressable_joint_name,
            pressedness_threshold=pressedness_threshold,
            **kwargs,
        )


class TurnableObjectReference(ArticulationObjectReference, Turnable):
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
            **kwargs,
        )


def _find_owning_rigid_body_prim(prim: Usd.Prim) -> Usd.Prim:
    """Return the nearest rigid body at or above prim, or an invalid prim when there is none."""
    while prim and not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        prim = prim.GetParent()
    return prim


def _get_prim_path_in_referenced_usd(stage: Usd.Stage, prim_path: str) -> str:
    """Return a prim's path in the USD file that the stage's default prim references.

    Args:
        stage: Stage whose default prim references one USD file, as returned by Object.open_spawned_usd_stage.
        prim_path: Path of a prim at or beneath the stage's default prim.

    Returns:
        The prim's path in the referenced USD file.
    """
    default_prim = stage.GetDefaultPrim()
    # The file's default prim can compose references of its own; only the stage's root layer adds the file's.
    [reference] = [
        arc
        for arc in Usd.PrimCompositionQuery.GetDirectReferences(default_prim).GetCompositionArcs()
        if arc.IsIntroducedInRootLayerStack()
    ]
    return Sdf.Path(prim_path).ReplacePrefix(default_prim.GetPath(), reference.GetTargetPrimPath()).pathString

# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Derived geometry for one native object spawn configuration."""

from __future__ import annotations

import numpy as np
import trimesh
from collections.abc import Collection
from copy import deepcopy

from isaaclab.sim import (
    CapsuleCfg,
    ConeCfg,
    CuboidCfg,
    CylinderCfg,
    MultiAssetSpawnerCfg,
    MultiUsdFileCfg,
    SphereCfg,
    UsdFileCfg,
)
from isaaclab.sim.spawners.shapes.shapes_cfg import ShapeCfg
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.sim.utils import standardize_xform_ops
from isaaclab.utils.assets import retrieve_file_path
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.usd.helpers import extract_trimesh_from_prim
from isaaclab_arena.utils.usd.rigid_bodies import apply_usd_variant_selections


class ObjectGeometry:
    """Cache bounds, contact paths, and meshes derived from one native spawn configuration."""

    def __init__(self, spawn_cfg: SpawnerCfg, object_type: ObjectType):
        assert not isinstance(
            spawn_cfg, (MultiAssetSpawnerCfg, MultiUsdFileCfg)
        ), "Singleton geometry requires one native spawn configuration"
        self.spawn_cfg = deepcopy(spawn_cfg)
        self.object_type = object_type
        self.bounding_box: AxisAlignedBoundingBox | None = None
        self._collision_meshes: dict[tuple[str, ...], trimesh.Trimesh | ValueError | None] = {}

    def matches(self, spawn_cfg: SpawnerCfg) -> bool:
        """Whether the cached geometry was derived from the current native configuration."""
        return self.spawn_cfg.to_dict() == spawn_cfg.to_dict()

    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        """Return scaled local bounds in the frame used to write this object's pose.

        Rigid objects use their rigid-body frame because Isaac Lab's root-pose writes
        target that body, which may be translated or rotated beneath the spawn root.
        Other object types use the spawn-root frame.
        """
        if self.bounding_box is None:
            if isinstance(self.spawn_cfg, UsdFileCfg):
                stage = self._open_scaled_usd_stage()
                bounds_prim = (
                    self._get_rigid_body(stage) if self.object_type == ObjectType.RIGID else stage.GetDefaultPrim()
                )
                bounds = UsdGeom.BBoxCache(
                    Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_]
                ).ComputeWorldBound(bounds_prim)
                if self.object_type == ObjectType.RIGID:
                    body_transform = Gf.Transform(
                        UsdGeom.Xformable(bounds_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    )
                    # B is the rigid body's pose frame; W is this temporary stage's world.
                    # Remove the pose only: physical scale stays in the geometry bounds.
                    T_W_B = Gf.Matrix4d(body_transform.GetRotation(), body_transform.GetTranslation())
                    bounds.Transform(T_W_B.GetInverse())
                bounds = bounds.ComputeAlignedRange()
                assert not bounds.IsEmpty(), f"No bounded geometry in {self.spawn_cfg.usd_path}"
                self.bounding_box = AxisAlignedBoundingBox(tuple(bounds.GetMin()), tuple(bounds.GetMax()))
            else:
                self.bounding_box = self._get_shape_bounding_box()
        return self.bounding_box

    def get_collision_mesh(self, excluded_prim_paths: Collection[str] = ()) -> trimesh.Trimesh | None:
        """Return a mesh in the same scaled pose frame as get_bounding_box().

        Args:
            excluded_prim_paths: Absolute paths in the original USD whose subtrees are omitted.

        Returns:
            An independent mesh, or None for non-USD spawners or fully excluded geometry.
        """
        if not isinstance(self.spawn_cfg, UsdFileCfg):
            assert not excluded_prim_paths, "USD prim exclusions require a USD asset."
            return None
        exclusions = tuple(sorted(excluded_prim_paths))
        if exclusions not in self._collision_meshes:
            try:
                self._collision_meshes[exclusions] = self._extract_collision_mesh(exclusions)
            except ValueError as error:
                self._collision_meshes[exclusions] = error
        mesh = self._collision_meshes[exclusions]
        if isinstance(mesh, ValueError):
            raise mesh
        return mesh.copy() if mesh is not None else None

    def _extract_collision_mesh(self, exclusions: tuple[str, ...]) -> trimesh.Trimesh | None:
        """Extract selected USD geometry after applying source-frame exclusions."""
        stage = self._open_scaled_usd_stage()
        geometry_root = self._get_rigid_body(stage) if self.object_type == ObjectType.RIGID else stage.GetDefaultPrim()
        geometry_root_path = geometry_root.GetPath()
        geometry_transform = UsdGeom.Xformable(geometry_root).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        if exclusions:
            source_stage = Usd.Stage.Open(retrieve_file_path(self.spawn_cfg.usd_path))
            source_root_path = source_stage.GetDefaultPrim().GetPath()
            for exclusion in exclusions:
                source_path = Sdf.Path(exclusion)
                assert source_path.IsAbsolutePath(), f"Excluded prim path must be absolute: {exclusion}"
                if source_path.HasPrefix(source_root_path):
                    target_path = source_path.ReplacePrefix(source_root_path, stage.GetDefaultPrim().GetPath())
                    target = stage.GetPrimAtPath(target_path)
                    if target:
                        # Instance proxies are read-only. Override their instance ancestors
                        # only in this temporary stage so the original asset stays instanceable.
                        while target.IsInstanceProxy():
                            instance_root = target.GetParent()
                            while instance_root.IsInstanceProxy():
                                instance_root = instance_root.GetParent()
                            instance_root.SetInstanceable(False)
                            target = stage.GetPrimAtPath(target_path)
                        target.SetActive(False)
            geometry_root = stage.GetPrimAtPath(geometry_root_path)
        if (
            not geometry_root
            or not geometry_root.IsActive()
            or (
                exclusions
                and not any(
                    prim.IsA(UsdGeom.Gprim) for prim in Usd.PrimRange(geometry_root, Usd.TraverseInstanceProxies())
                )
            )
        ):
            return None
        mesh = extract_trimesh_from_prim(stage, str(geometry_root.GetPath()))
        if self.object_type == ObjectType.RIGID:
            body_transform = Gf.Transform(geometry_transform)
            # Remove the body pose while retaining its accumulated physical scale.
            T_W_B = Gf.Matrix4d(body_transform.GetRotation(), body_transform.GetTranslation())
            geometry_transform *= T_W_B.GetInverse()
        mesh.apply_transform(np.asarray(geometry_transform).T)
        return mesh

    def get_contact_body_path(self) -> str:
        """Return the single rigid body's path suffix relative to the spawned asset root."""
        assert self.object_type == ObjectType.RIGID, "Contact bodies require a rigid object."
        if isinstance(self.spawn_cfg, UsdFileCfg):
            stage = self.open_usd_stage()
            return str(self._get_rigid_body(stage).GetPath()).removeprefix("/Asset")
        assert isinstance(
            self.spawn_cfg, ShapeCfg
        ), f"Contact-body discovery supports USD assets and native shape spawners; got {type(self.spawn_cfg).__name__}."
        return ""

    def open_usd_stage(self) -> Usd.Stage:
        """Reference the selected USD asset into an isolated stage without editing its source."""
        assert isinstance(self.spawn_cfg, UsdFileCfg), "A USD spawn configuration is required."
        stage = Usd.Stage.CreateInMemory()
        root = stage.DefinePrim("/Asset")
        root.GetReferences().AddReference(retrieve_file_path(self.spawn_cfg.usd_path))
        stage.SetDefaultPrim(root)
        selections = self.spawn_cfg.variants
        if selections is not None:
            if not isinstance(selections, dict):
                selections = selections.to_dict()
            apply_usd_variant_selections(stage, selections)
        return stage

    def _get_rigid_body(self, stage: Usd.Stage) -> Usd.Prim:
        """Find the only rigid body in the referenced source's default-prim subtree."""
        rigid_bodies = []
        for prim in Usd.PrimRange(stage.GetDefaultPrim(), Usd.TraverseInstanceProxies()):
            assert not prim.HasAPI(
                UsdPhysics.ArticulationRootAPI
            ), f"Rigid geometry cannot contain an articulation root: {prim.GetPath()}"
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_bodies.append(prim)
        assert len(rigid_bodies) == 1, (
            f"Rigid objects require exactly one rigid body in {self.spawn_cfg.usd_path}; "
            f"found {[str(body.GetPath()) for body in rigid_bodies]}."
        )
        return rigid_bodies[0]

    def _open_scaled_usd_stage(self) -> Usd.Stage:
        """Match native spawning at the origin with the configured scale and selected variants."""
        stage = self.open_usd_stage()
        standardize_xform_ops(
            stage.GetDefaultPrim(),
            translation=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            scale=self.spawn_cfg.scale,
        )
        return stage

    def _get_shape_bounding_box(self) -> AxisAlignedBoundingBox:
        """Compute bounds for the standard Isaac Lab primitive spawners."""
        spawn_cfg = self.spawn_cfg
        if isinstance(spawn_cfg, CuboidCfg):
            half_extents = [dimension / 2 for dimension in spawn_cfg.size]
        elif isinstance(spawn_cfg, SphereCfg):
            half_extents = [spawn_cfg.radius] * 3
        elif isinstance(spawn_cfg, (CylinderCfg, ConeCfg, CapsuleCfg)):
            half_extents = [spawn_cfg.radius] * 3
            axis = "XYZ".index(spawn_cfg.axis)
            half_extents[axis] = spawn_cfg.height / 2
            if isinstance(spawn_cfg, CapsuleCfg):
                half_extents[axis] += spawn_cfg.radius
        else:
            raise ValueError(f"Provide bounding_box for the custom spawner {type(spawn_cfg).__name__}.")
        return AxisAlignedBoundingBox(tuple(-extent for extent in half_extents), tuple(half_extents))

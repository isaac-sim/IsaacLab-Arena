# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Apply asset-relative physics configuration during USD spawning."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from isaaclab.sim import schemas
from isaaclab.sim.spawners.from_files import spawn_from_usd
from isaaclab.sim.utils import clone, use_stage
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

if TYPE_CHECKING:
    from .physics_config import PhysicsUsdFileCfg, PrimPhysicsCfg


def _relative_target(root: Usd.Prim, relative_path: str) -> Usd.Prim:
    """Resolve an exact path inside one asset and require an editable prim."""
    assert isinstance(relative_path, str) and relative_path, "Physics target must be a nonempty relative prim path."
    path = Sdf.Path(relative_path)
    assert (
        not path.IsAbsolutePath()
        and (path.IsPrimPath() or path == Sdf.Path.reflexiveRelativePath)
        and ".." not in relative_path.split("/")
        and "{" not in relative_path
    ), f"Physics target must be an asset-relative prim path: {relative_path!r}"
    prim = root.GetStage().GetPrimAtPath(path.MakeAbsolutePath(root.GetPath()))
    assert prim.IsValid(), f"Physics target does not exist: {root.GetPath()}/{relative_path}"
    assert (
        not prim.IsInstanceProxy()
    ), f"Physics target {prim.GetPath()} is an instance proxy; set make_uninstanceable=True to edit it."
    return prim


def _validate_override(
    root: Usd.Prim, prim: Usd.Prim, cfg: PrimPhysicsCfg, new_colliders: set[Sdf.Path]
) -> list[Sdf.Path]:
    """Validate one target and resolve its collision exclusions before authoring overrides."""
    from .physics_config import PrimPhysicsCfg

    assert isinstance(cfg, PrimPhysicsCfg), f"Physics override for {prim.GetPath()} must be PrimPhysicsCfg."
    if cfg.collision_props is not None:
        assert prim.IsA(UsdGeom.Gprim), f"Collision target must be a geometry prim: {prim.GetPath()}"
        assert all(isinstance(fragment, schemas.CollisionFragment) for fragment in cfg.collision_props)
    if cfg.physics_material is not None:
        assert (
            prim.HasAPI(UsdPhysics.CollisionAPI) or cfg.collision_props is not None
        ), f"Physics material target must be a collider: {prim.GetPath()}"
        material_path = prim.GetPath().AppendChild("ArenaPhysicsMaterial")
        assert not root.GetStage().GetPrimAtPath(
            material_path
        ), f"Physics material path already exists: {material_path}"
    if cfg.joint_drive_props is not None:
        assert prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(
            UsdPhysics.PrismaticJoint
        ), f"Joint-drive target must be a revolute or prismatic joint: {prim.GetPath()}"
        assert all(isinstance(fragment, schemas.JointDriveFragment) for fragment in cfg.joint_drive_props)
    if cfg.mujoco_equality is not None:
        assert any(
            schema in prim.GetAppliedSchemas()
            for schema in ("MjcEqualityJointAPI", "MjcEqualityConnectAPI", "MjcEqualityWeldAPI")
        ), f"Equality target has no authored MuJoCo equality schema: {prim.GetPath()}"
        for name, length in (("solref", 2), ("solimp", 5)):
            value = getattr(cfg.mujoco_equality, name)
            assert value is None or (
                len(value) == length and all(math.isfinite(x) for x in value)
            ), f"Equality {name} must contain {length} finite values: {prim.GetPath()}"
    excluded = [_relative_target(root, path) for path in cfg.filtered_pairs]
    assert prim not in excluded, f"Cannot exclude a collider from itself: {prim.GetPath()}"
    for target in [prim, *excluded] if excluded else []:
        assert (
            target.HasAPI(UsdPhysics.RigidBodyAPI)
            or target.HasAPI(UsdPhysics.CollisionAPI)
            or target.GetPath() in new_colliders
        ), f"Collision exclusion target must be a rigid body or collider: {target.GetPath()}"
    return [target.GetPath() for target in excluded]


def apply_prim_physics(root: Usd.Prim, overrides: dict[str, PrimPhysicsCfg]) -> None:
    """Author physics on selected prims without changing the source asset or backend builder.

    Args:
        root: Spawned asset root, whose stage receives the authored opinions.
        overrides: Exact asset-relative paths and their physics configuration.
    """
    targets = [(_relative_target(root, path), cfg) for path, cfg in overrides.items()]
    new_colliders = {prim.GetPath() for prim, cfg in targets if cfg.collision_props is not None}
    resolved = [(prim, cfg, _validate_override(root, prim, cfg, new_colliders)) for prim, cfg in targets]

    stage = root.GetStage()
    for prim, cfg, excluded in resolved:
        path = str(prim.GetPath())
        if cfg.collision_props is not None:
            UsdPhysics.CollisionAPI.Apply(prim)
            applied = schemas.apply_collision_properties(path, cfg.collision_props, stage)
            assert applied, f"Failed to apply collision properties to {path}"
        if cfg.physics_material is not None:
            # Keep materials beneath the collider so clone backends remap their binding paths.
            material_path = f"{path}/ArenaPhysicsMaterial"
            with use_stage(stage):
                cfg.physics_material.func(material_path, cfg.physics_material)
            binding = UsdShade.MaterialBindingAPI.Apply(prim)
            binding.Bind(
                UsdShade.Material(stage.GetPrimAtPath(material_path)),
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                materialPurpose="physics",
            )
        if cfg.joint_drive_props is not None:
            applied = schemas.apply_joint_drive_properties(path, cfg.joint_drive_props, stage)
            assert applied, f"Failed to apply joint-drive properties to {path}"
        if cfg.mujoco_equality is not None:
            for name in ("solref", "solimp"):
                value = getattr(cfg.mujoco_equality, name)
                if value is not None:
                    prim.CreateAttribute(f"mjc:{name}", Sdf.ValueTypeNames.DoubleArray).Set(Vt.DoubleArray(value))
        if excluded:
            relation = UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel()
            for target in excluded:
                relation.AddTarget(target)


@clone
def spawn_usd_with_physics(
    prim_path: str,
    cfg: PhysicsUsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn USD, apply selected physics properties, then clone the configured asset.

    Args:
        prim_path: Asset path, optionally with an environment expression in its parent path.
        cfg: USD spawn configuration with asset-relative physics overrides.
        translation: Root translation, following the ordinary USD spawner convention.
        orientation: Root quaternion in XYZW order.
        **kwargs: Additional arguments forwarded to the ordinary USD spawner.

    Returns:
        The first spawned asset root.
    """
    prim = spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    apply_prim_physics(prim, cfg.prim_physics)
    return prim

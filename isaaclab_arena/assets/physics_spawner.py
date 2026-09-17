# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Apply asset-relative physics configuration during USD spawning."""

from __future__ import annotations

import math
from dataclasses import fields
from typing import TYPE_CHECKING

from isaaclab.sim import UsdFileCfg, schemas
from isaaclab.sim.spawners.from_files import spawn_from_usd
from isaaclab.sim.utils import clone, use_stage
from isaaclab.utils.string import string_to_callable
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

if TYPE_CHECKING:
    from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg

    from .physics_config import MujocoEqualityPropertiesCfg, PrimPhysicsCfg


def with_prim_physics(cfg: UsdFileCfg, overrides: dict[str, PrimPhysicsCfg]) -> UsdFileCfg:
    """Return an independent USD spawn config that applies the given per-prim overrides.

    Args:
        cfg: Standard USD config whose ordinary spawn options are retained.
        overrides: Exact asset-relative paths and physics settings; replaces any previous mapping.

    Returns:
        A config using the physics spawn wrapper; assign it back to the asset's spawn field.
    """
    from .physics_config import _PhysicsUsdFileCfg

    assert type(cfg) in (UsdFileCfg, _PhysicsUsdFileCfg), "Per-prim physics requires a standard USD spawn config."
    spawn_func = string_to_callable(str(cfg.func)) if isinstance(cfg.func, str) else cfg.func
    assert spawn_func in (
        spawn_from_usd,
        spawn_usd_with_physics,
    ), "Custom spawn functions must call apply_prim_physics before cloning."
    # Preserve typed fields rather than converting nested configs to dictionaries.
    values = {field.name: getattr(cfg, field.name) for field in fields(cfg) if field.init}
    values.update(prim_physics=overrides, func=spawn_usd_with_physics)
    # Declared fields survive configclass.copy()/replace(); construction deep-copies mutable values.
    return _PhysicsUsdFileCfg(**values)


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
        _validate_equality(prim, cfg.mujoco_equality)
    return _resolve_filtered_pairs(root, prim, cfg.filtered_pairs, new_colliders)


def _validate_equality(prim: Usd.Prim, cfg: MujocoEqualityPropertiesCfg) -> None:
    """Require an existing equality schema and finite solver parameters of the expected lengths."""
    assert any(
        schema in prim.GetAppliedSchemas()
        for schema in ("MjcEqualityJointAPI", "MjcEqualityConnectAPI", "MjcEqualityWeldAPI")
    ), f"Equality target has no authored MuJoCo equality schema: {prim.GetPath()}"
    for name, length in (("solref", 2), ("solimp", 5)):
        value = getattr(cfg, name)
        assert value is None or (
            len(value) == length and all(math.isfinite(x) for x in value)
        ), f"Equality {name} must contain {length} finite values: {prim.GetPath()}"


def _resolve_filtered_pairs(
    root: Usd.Prim, prim: Usd.Prim, paths: list[str], new_colliders: set[Sdf.Path]
) -> list[Sdf.Path]:
    """Resolve collision exclusions, including colliders enabled by another override."""
    if not paths:
        return []
    excluded = [_relative_target(root, path) for path in paths]
    assert prim not in excluded, f"Cannot exclude a collider from itself: {prim.GetPath()}"
    for target in [prim, *excluded]:
        assert (
            target.HasAPI(UsdPhysics.RigidBodyAPI)
            or target.HasAPI(UsdPhysics.CollisionAPI)
            or target.GetPath() in new_colliders
        ), f"Collision exclusion target must be a rigid body or collider: {target.GetPath()}"
    return [target.GetPath() for target in excluded]


def _resolve_overrides(
    root: Usd.Prim, overrides: dict[str, PrimPhysicsCfg]
) -> list[tuple[Usd.Prim, PrimPhysicsCfg, list[Sdf.Path]]]:
    """Resolve and validate all targets before writing any per-prim properties."""
    from .physics_config import PrimPhysicsCfg

    targets = []
    new_colliders = set()
    for path, cfg in overrides.items():
        assert isinstance(cfg, PrimPhysicsCfg), f"Physics override for {path} must be PrimPhysicsCfg."
        prim = _relative_target(root, path)
        targets.append((prim, cfg))
        if cfg.collision_props is not None:
            new_colliders.add(prim.GetPath())

    # Collect prospective colliders first so exclusion validity does not depend on mapping order.
    resolved = []
    for prim, cfg in targets:
        excluded = _validate_override(root, prim, cfg, new_colliders)
        resolved.append((prim, cfg, excluded))
    return resolved


def _bind_material(prim: Usd.Prim, material_cfg: RigidBodyMaterialBaseCfg) -> None:
    """Create a local physics material and bind it to one collider."""
    stage = prim.GetStage()
    # Child material paths remap during cloning and avoid modifying a shared source material.
    material_path = f"{prim.GetPath()}/ArenaPhysicsMaterial"
    with use_stage(stage):
        material_cfg.func(material_path, material_cfg)
    binding = UsdShade.MaterialBindingAPI.Apply(prim)
    binding.Bind(
        UsdShade.Material(stage.GetPrimAtPath(material_path)),
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        materialPurpose="physics",
    )


def _apply_equality(prim: Usd.Prim, cfg: MujocoEqualityPropertiesCfg) -> None:
    """Author response parameters without changing the existing coupling relationship."""
    for name in ("solref", "solimp"):
        value = getattr(cfg, name)
        if value is not None:
            prim.CreateAttribute(f"mjc:{name}", Sdf.ValueTypeNames.DoubleArray).Set(Vt.DoubleArray(value))


def _apply_override(prim: Usd.Prim, cfg: PrimPhysicsCfg, excluded: list[Sdf.Path]) -> None:
    """Apply one previously validated collider or joint override."""
    stage = prim.GetStage()
    path = str(prim.GetPath())
    if cfg.collision_props is not None:
        UsdPhysics.CollisionAPI.Apply(prim)
        applied = schemas.apply_collision_properties(path, cfg.collision_props, stage)
        assert applied, f"Failed to apply collision properties to {path}"
    if cfg.physics_material is not None:
        _bind_material(prim, cfg.physics_material)
    if cfg.joint_drive_props is not None:
        applied = schemas.apply_joint_drive_properties(path, cfg.joint_drive_props, stage)
        assert applied, f"Failed to apply joint-drive properties to {path}"
    if cfg.mujoco_equality is not None:
        _apply_equality(prim, cfg.mujoco_equality)
    if excluded:
        relation = UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel()
        # Add relationships rather than replacing exclusions already authored in the asset.
        for target in excluded:
            relation.AddTarget(target)


def apply_prim_physics(root: Usd.Prim, overrides: dict[str, PrimPhysicsCfg]) -> None:
    """Author physics on selected prims without changing the source asset or backend builder.

    Args:
        root: Spawned asset root, whose stage receives the authored opinions.
        overrides: Exact asset-relative paths and their physics configuration.
    """
    # Validate the full mapping first so a bad later target does not leave earlier overrides applied.
    for prim, cfg, excluded in _resolve_overrides(root, overrides):
        _apply_override(prim, cfg, excluded)


@clone
def spawn_usd_with_physics(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn USD, apply selected physics properties, then clone the configured asset.

    Args:
        prim_path: Asset path, optionally with an environment expression in its parent path.
        cfg: USD config returned by with_prim_physics.
        translation: Root translation, following the ordinary USD spawner convention.
        orientation: Root quaternion in XYZW order.
        **kwargs: Additional arguments forwarded to the ordinary USD spawner.

    Returns:
        The first spawned asset root.
    """
    # The outer clone decorator resolves the pattern to one concrete prototype path first.
    # Loading that one path cannot clone siblings; they are copied only after our overrides.
    prim = spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    apply_prim_physics(prim, cfg.prim_physics)
    return prim

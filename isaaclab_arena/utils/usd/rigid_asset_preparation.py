# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Prepare USD layouts required by Isaac Lab's shared rigid-body view."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import fields

from isaaclab.sim import UsdFileCfg
from isaaclab.sim.schemas import SchemaFragment
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.sim.utils import standardize_xform_ops
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from isaaclab_arena.assets.asset_cache import get_arena_asset_cache_dir
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.object_variant import ObjectVariant
from isaaclab_arena.utils.usd.prim_paths import parse_relative_prim_path

_SOURCE_ROOT = Sdf.Path("/Asset")
_PREPARED_ROOT = Sdf.Path("/Prepared")
_PREPARED_SOURCE = _PREPARED_ROOT.AppendChild("source")
_PREPARED_BODY = _PREPARED_ROOT.AppendChild("rigid_body")


def prepare_rigid_object_variants(
    variants: Sequence[ObjectVariant],
) -> list[SpawnerCfg]:
    """Return independent spawn configs with a common relative rigid-body path.

    Isaac Lab discovers the rigid-body path from one environment and reuses it for
    every environment. Compatible sources need no asset edits, including when their
    scales differ. Incompatible USD hierarchies are normalized in a separate cache;
    the original sources and native per-variant spawn scales remain unchanged.

    Args:
        variants: Asset alternatives, each containing exactly one rigid body.

    Returns:
        One native spawn configuration per alternative, in the original order.
    """
    assert variants, "At least one rigid object variant is required."
    assert all(variant.object_type == ObjectType.RIGID for variant in variants), "Variants must be rigid objects."
    body_paths = [variant.get_contact_body_path() for variant in variants]
    if len(set(body_paths)) == 1:
        return [deepcopy(variant.spawn_cfg) for variant in variants]
    assert all(isinstance(variant.spawn_cfg, UsdFileCfg) for variant in variants), (
        "Different rigid-body layouts require USD asset preparation. "
        "Primitive and custom spawners must share the USD variants' relative body path."
    )
    return [_prepare_usd_variant(variant, body_path) for variant, body_path in zip(variants, body_paths)]


def _relocated_path(path: Sdf.Path, original_body: Sdf.Path) -> Sdf.Path:
    """Map a path from the original reference into the prepared asset."""
    if path.HasPrefix(original_body):
        return path.ReplacePrefix(original_body, _PREPARED_BODY)
    return path.ReplacePrefix(_SOURCE_ROOT, _PREPARED_SOURCE)


def _rewrite_targets(stage: Usd.Stage, original_body: Sdf.Path) -> None:
    """Keep material bindings and shader connections within the prepared reference."""
    for prim in stage.Traverse():
        for relationship in prim.GetRelationships():
            targets = relationship.GetTargets()
            if targets:
                relationship.SetTargets([_relocated_path(target, original_body) for target in targets])
        for attribute in prim.GetAttributes():
            connections = attribute.GetConnections()
            if connections:
                attribute.SetConnections([_relocated_path(path, original_body) for path in connections])


def _remap_spawn_targets(spawn_cfg: UsdFileCfg, stage: Usd.Stage, original_body: Sdf.Path) -> None:
    """Retain exact per-prim overrides and native physics pattern selections after relocation."""
    if getattr(spawn_cfg, "prim_physics", None):
        remapped_overrides = {}
        for relative_path, physics_cfg in spawn_cfg.prim_physics.items():
            original_path = parse_relative_prim_path(relative_path).MakeAbsolutePath(_SOURCE_ROOT)
            assert stage.GetPrimAtPath(original_path), f"Physics target does not exist: {relative_path}"
            new_path = _relocated_path(original_path, original_body).MakeRelativePath(_PREPARED_ROOT)
            remapped_overrides[str(new_path)] = physics_cfg
        spawn_cfg.prim_physics = remapped_overrides

    # Native schema fragments target patterns relative to the spawn root. Resolve those
    # patterns against the original hierarchy before replacing them with exact paths.
    for field in fields(spawn_cfg):
        if not field.name.endswith("_props"):
            continue
        value = getattr(spawn_cfg, field.name)
        if isinstance(value, SchemaFragment) or (
            isinstance(value, (list, tuple)) and all(isinstance(fragment, SchemaFragment) for fragment in value)
        ):
            if not value:
                continue
            target_pattern = "(/.*)?"
            body_schema = {"mass_props": UsdPhysics.MassAPI, "collision_props": UsdPhysics.CollisionAPI}.get(field.name)
            if body_schema is not None and not any(prim.HasAPI(body_schema) for prim in stage.Traverse()):
                # Native bare fragments create a missing body-family API on the source root.
                # Explicit mass targets expose that creation flag; collision targets do not.
                assert field.name == "mass_props", (
                    "USD layout preparation requires authored collision schemas when collision_props "
                    "contains native schema fragments. Add CollisionAPI to the source geometry first."
                )
                spawn_cfg.mass_props_create_if_missing = True
                target_pattern = ""
            value = {target_pattern: value}
        if not isinstance(value, dict):
            continue
        remapped_patterns = {}
        for pattern, fragments in value.items():
            matched_paths = []
            for prim in stage.Traverse():
                if re.fullmatch(str(_SOURCE_ROOT) + pattern, str(prim.GetPath())):
                    matched_paths.append(prim.GetPath())
            assert matched_paths, f"Physics pattern {field.name}[{pattern!r}] matches no source prims."
            for original_path in matched_paths:
                suffix = str(_relocated_path(original_path, original_body)).removeprefix(str(_PREPARED_ROOT))
                # Later patterns retain their override precedence when they hit the same prim.
                remapped_patterns.setdefault(suffix, []).extend(
                    fragments if isinstance(fragments, (list, tuple)) else [fragments]
                )
        setattr(spawn_cfg, field.name, remapped_patterns)

    material_path = spawn_cfg.visual_material_path
    if material_path and not Sdf.Path(material_path).IsAbsolutePath():
        original_path = Sdf.Path(material_path).MakeAbsolutePath(_SOURCE_ROOT)
        spawn_cfg.visual_material_path = str(
            _relocated_path(original_path, original_body).MakeRelativePath(_PREPARED_ROOT)
        )


def _prepare_usd_variant(variant: ObjectVariant, body_path: str) -> UsdFileCfg:
    """Move one composed USD body's subtree while preserving the complete referenced asset."""
    spawn_cfg = deepcopy(variant.spawn_cfg)
    stage = variant.open_usd_stage()
    original_body = Sdf.Path(str(_SOURCE_ROOT) + body_path)

    # Make composition editable in this throwaway stage, including referenced instance contents.
    while True:
        instances = [prim for prim in stage.Traverse() if prim.IsInstance()]
        if not instances:
            break
        for prim in instances:
            prim.SetInstanceable(False)
    _remap_spawn_targets(spawn_cfg, stage, original_body)

    root = stage.GetDefaultPrim()
    standardize_xform_ops(
        root,
        translation=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0, 1.0),
        # An explicit scale replaces the source root scale at native spawn time.
        # Keep its value on the spawn config rather than baking it into the cache.
        scale=(1.0, 1.0, 1.0) if spawn_cfg.scale is not None else None,
    )
    body = stage.GetPrimAtPath(original_body)
    ancestor = body
    while ancestor and ancestor != stage.GetPseudoRoot():
        if ancestor.IsA(UsdGeom.Xformable):
            for operation in UsdGeom.Xformable(ancestor).GetOrderedXformOps():
                assert (
                    not operation.GetNumTimeSamples()
                ), f"Rigid asset preparation requires static ancestor transforms: {ancestor.GetPath()}"
        ancestor = ancestor.GetParent()
    body_transform = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    inherited_materials = []
    for purpose in (
        UsdShade.Tokens.allPurpose,
        UsdShade.Tokens.preview,
        UsdShade.Tokens.full,
    ):
        material, relationship = UsdShade.MaterialBindingAPI(body).ComputeBoundMaterial(purpose)
        if material:
            binding_strength = UsdShade.MaterialBindingAPI.GetMaterialBindingStrength(relationship)
            inherited_materials.append((purpose, material.GetPath(), binding_strength))
    imageable = UsdGeom.Imageable(body)
    visibility = imageable.ComputeVisibility() if imageable else None
    purpose = imageable.ComputePurpose() if imageable else None

    # Flatten resolves referenced layers and makes texture paths absolute before moving the
    # cache to another directory. Keep the complete source subtree: materials may be siblings
    # of the body, and ancestors may carry geometry or referenced resources.
    layer = stage.Flatten(addSourceFileComment=False)
    prepared_stage = Usd.Stage.Open(layer)
    prepared_root = UsdGeom.Xform.Define(prepared_stage, _PREPARED_ROOT).GetPrim()
    assert Sdf.CopySpec(layer, _SOURCE_ROOT, layer, _PREPARED_SOURCE)
    copied_body = original_body.ReplacePrefix(_SOURCE_ROOT, _PREPARED_SOURCE)
    assert Sdf.CopySpec(layer, copied_body, layer, _PREPARED_BODY)
    prepared_stage.RemovePrim(copied_body)
    prepared_stage.RemovePrim(_SOURCE_ROOT)
    prepared_stage.SetDefaultPrim(prepared_root)
    _rewrite_targets(prepared_stage, original_body)

    prepared_body = prepared_stage.GetPrimAtPath(_PREPARED_BODY)
    UsdGeom.Xformable(prepared_body).MakeMatrixXform().Set(body_transform)
    if visibility is not None:
        UsdGeom.Imageable(prepared_body).CreateVisibilityAttr().Set(visibility)
        UsdGeom.Imageable(prepared_body).CreatePurposeAttr().Set(purpose)
    for purpose, material_path, binding_strength in inherited_materials:
        material = UsdShade.Material(prepared_stage.GetPrimAtPath(_relocated_path(material_path, original_body)))
        UsdShade.MaterialBindingAPI.Apply(prepared_body).Bind(
            material, bindingStrength=binding_strength, materialPurpose=purpose
        )

    # The content includes selected variants and referenced USD data. The source identifier
    # distinguishes same-named assets from different locations; scales stay native and share a cache.
    cache_identity = f"rigid-layout-v1\n{variant.spawn_cfg.usd_path}\n{layer.ExportToString()}"
    content_hash = hashlib.sha256(cache_identity.encode()).hexdigest()
    cache_path = get_arena_asset_cache_dir() / f"rigid_{content_hash}.usd"
    if not cache_path.exists():
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=cache_path.parent, suffix=".usd", delete=False) as temporary_file:
                temporary_path = temporary_file.name
            layer.Export(temporary_path)
            os.replace(temporary_path, cache_path)
        finally:
            if temporary_path is not None and os.path.exists(temporary_path):
                os.unlink(temporary_path)
    spawn_cfg.usd_path = str(cache_path)
    # Variant selections are resolved into the flattened asset; the new wrapper has no variant sets.
    spawn_cfg.variants = None
    return spawn_cfg

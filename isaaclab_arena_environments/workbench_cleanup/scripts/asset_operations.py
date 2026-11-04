# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""YAML-configured USD edits with no workbench-specific prim names."""

from __future__ import annotations

import fnmatch

from pxr import Sdf, Usd, UsdPhysics, UsdShade

from isaaclab_arena_environments.workbench_cleanup.scripts.fixed_articulations import create_fixed_articulations
from isaaclab_arena_environments.workbench_cleanup.scripts.mesh_partition import inspect_mesh_components, partition_mesh


def typed_value(config):
    """Convert a YAML value to its declared USD scalar, vector, matrix, quaternion or array type."""
    assert isinstance(config, dict) and set(config) == {"type", "value"}, "Typed value requires type and value"
    value_type = Sdf.ValueTypeNames.Find(config["type"])
    assert value_type, f"Unknown USD value type: {config['type']}"
    value = config["value"]
    python_class = value_type.type.pythonClass
    if python_class is not None:
        if config["type"].startswith("quat") and not value_type.isArray:
            value = python_class(*value)  # YAML order: w, x, y, z.
        else:
            value = python_class(value)
    elif config["type"] == "asset":
        value = Sdf.AssetPath(value)
    return value_type, value


def asset_path(stage: Usd.Stage, path: str) -> Sdf.Path:
    """Return a prim path beneath the asset's default prim."""
    root = stage.GetDefaultPrim().GetPath()
    if path == ".":
        return root
    result = Sdf.Path(path)
    if not result.IsAbsolutePath():
        result = root.AppendPath(result)
    assert result.IsPrimPath() and result.HasPrefix(root), f"Path must be inside the asset: {path}"
    return result


def select_prims(stage: Usd.Stage, selector: dict) -> list[Usd.Prim]:
    """Select existing prims by root-relative glob, schema and type; reject empty or misspelled selections."""
    assert set(selector) <= {"paths", "schema", "type", "exclude"}, f"Unknown selector fields: {selector}"
    paths = selector.get("paths", ["*"])
    assert isinstance(paths, list) and all(isinstance(path, str) for path in paths), "paths must be a list of globs"
    root = stage.GetDefaultPrim().GetPath()
    result = []
    for prim in stage.Traverse():
        relative = str(prim.GetPath().MakeRelativePath(root))
        if not any(fnmatch.fnmatchcase(relative, pattern) for pattern in paths):
            continue
        if any(fnmatch.fnmatchcase(relative, pattern) for pattern in selector.get("exclude", [])):
            continue
        if selector.get("schema") and selector["schema"] not in prim.GetAppliedSchemas():
            continue
        if selector.get("type") and prim.GetTypeName() != selector["type"]:
            continue
        result.append(prim)
    assert result, f"Selector matched no prims: {selector}"
    return result


def edit_prims(stage, *, select, attributes=None, apply_apis=(), remove_apis=(), type_name=None, active=None):
    """Edit typed attributes and applied schemas on explicitly selected prims."""
    prims = select_prims(stage, select)
    paths = [str(prim.GetPath()) for prim in prims]
    # Descendants must be edited before deactivating any selected ancestor.
    for prim in reversed(prims):
        for schema in remove_apis:
            prim.RemoveAppliedSchema(schema)
        for schema in apply_apis:
            prim.AddAppliedSchema(schema)
        if type_name is not None:
            prim.SetTypeName(type_name)
        for name, config in (attributes or {}).items():
            value_type, value = typed_value(config)
            prim.CreateAttribute(name, value_type, custom=False).Set(value)
        if active is not None:
            assert isinstance(active, bool), "active must be boolean"
            prim.SetActive(active)
    return {"prims": paths}


def partition(stage, *, prim_path, groups, remainder="Remaining", geometry_sha256=None):
    """Split complete mesh components with an optional source-geometry fingerprint guard."""
    before = inspect_mesh_components(stage, prim_path)
    if geometry_sha256 is not None:
        assert before["geometry_sha256"] == geometry_sha256, f"Mesh fingerprint changed: {prim_path}"
    return partition_mesh(stage, prim_path, groups, remainder)


def configure_articulation(stage, *, root=".", fixed_base_body=None, movable_joints=None, self_collisions=False):
    """Choose an articulation root, optionally weld other joints, and optionally anchor its base to the world."""
    root_prim = stage.GetPrimAtPath(asset_path(stage, root))
    assert root_prim, f"Missing articulation root: {root}"
    keep = None if movable_joints is None else {asset_path(stage, path) for path in movable_joints}
    if keep is not None:
        for path in keep:
            assert UsdPhysics.Joint(stage.GetPrimAtPath(path)), f"Missing joint: {path}"
    for prim in Usd.PrimRange(root_prim):
        prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if keep is not None and prim.IsA(UsdPhysics.Joint) and prim.GetPath() not in keep:
            prim.SetTypeName("PhysicsFixedJoint")
            for schema in prim.GetAppliedSchemas():
                if schema.startswith(("PhysicsDriveAPI:", "PhysxJointStateAPI:")):
                    prim.RemoveAppliedSchema(schema)
    UsdPhysics.ArticulationRootAPI.Apply(root_prim)
    root_prim.AddAppliedSchema("PhysxArticulationAPI")
    root_prim.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(self_collisions)
    if fixed_base_body is not None:
        body = stage.GetPrimAtPath(asset_path(stage, fixed_base_body))
        assert body and body.HasAPI(UsdPhysics.RigidBodyAPI), "Fixed base requires a rigid body"
        fixed_path = root_prim.GetPath().AppendChild("ArenaFixedBase")
        assert not stage.GetPrimAtPath(fixed_path), f"Fixed joint already exists: {fixed_path}"
        fixed = UsdPhysics.FixedJoint.Define(stage, fixed_path)
        fixed.CreateBody1Rel().SetTargets([body.GetPath()])
    return {"root": str(root_prim.GetPath())}


def preview_material(stage, *, prim_path, inputs, bind_to=()):
    """Create a UsdPreviewSurface material and optionally bind it over descendant bindings."""
    path = asset_path(stage, prim_path)
    assert not stage.GetPrimAtPath(path), f"Material already exists: {path}"
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path.AppendChild("Surface"))
    shader.CreateIdAttr("UsdPreviewSurface")
    for name, config in inputs.items():
        value_type, value = typed_value(config)
        shader.CreateInput(name, value_type).Set(value)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    for target in bind_to:
        prim = stage.GetPrimAtPath(asset_path(stage, target))
        assert prim, f"Missing material binding target: {target}"
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    return {"material": str(path)}


OPERATIONS = {
    "edit_prims": edit_prims,
    "partition_mesh": partition,
    "fixed_articulations": create_fixed_articulations,
    "configure_articulation": configure_articulation,
    "preview_material": preview_material,
}


def apply_operations(stage: Usd.Stage, operations: list[dict]) -> list[dict]:
    """Apply an ordered list of known operations; unknown fields fail instead of being ignored."""
    reports = []
    for index, operation in enumerate(operations):
        assert set(operation) == {"kind", "params"}, f"Operation {index} requires kind and params"
        kind = operation["kind"]
        assert kind in OPERATIONS, f"Unknown USD operation: {kind}"
        result = OPERATIONS[kind](stage, **operation["params"])
        reports.append({"kind": kind, "result": result})
    return reports

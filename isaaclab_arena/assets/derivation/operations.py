# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Reusable USD geometry, physics, joint, and material editing operations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def asset_prim(stage, path: str):
    """Return an existing prim addressed relative to the asset's default prim."""
    from pxr import Sdf

    root = stage.GetDefaultPrim().GetPath()
    target = root if path == "." else Sdf.Path(path)
    if not target.IsAbsolutePath():
        target = root.AppendPath(target)
    assert target.IsPrimPath() and target.HasPrefix(root), f"Prim must stay inside the asset: {path}"
    prim = stage.GetPrimAtPath(target)
    assert prim, f"Missing asset prim: {path}"
    return prim


def _selected_prims(stage, step):
    from pxr import Usd

    selected = []
    seen = set()
    excluded = {asset_prim(stage, path).GetPath() for path in step.get("exclude_paths", [])}
    for path in step.get("paths", ["."]):
        root = asset_prim(stage, path)
        candidates = list(Usd.PrimRange(root)) if step.get("descendants", False) else [root]
        for prim in candidates:
            if prim.GetPath() in seen or prim.GetPath() in excluded:
                continue
            seen.add(prim.GetPath())
            if "is_instanceable" in step and prim.IsInstanceable() != step["is_instanceable"]:
                continue
            if (api := step.get("has_api")) is not None and api not in prim.GetAppliedSchemas():
                continue
            if (schema := step.get("is_a")) is not None:
                schema_type = Usd.SchemaRegistry().GetTypeFromSchemaTypeName(schema)
                assert schema_type, f"Unknown USD schema: {schema}"
                if not prim.IsA(schema_type):
                    continue
            selected.append(prim)
    return selected


def _typed_value(config):
    from pxr import Sdf

    value_type = getattr(Sdf.ValueTypeNames, config["type"], None)
    assert value_type, f"Unknown USD value type: {config['type']}"
    value = config["value"]
    if isinstance(value, list):
        value_class = value_type.type.pythonClass
        assert value_class, f"USD value type does not accept an array: {config['type']}"
        value = value_class(value) if value_type.isArray else value_class(*value)
    return value_type, value


def _edit_prims(stage, step) -> None:
    for prim in _selected_prims(stage, step):
        if step.get("deinstance", False):
            prim.SetInstanceable(False)
        for schema in step.get("remove_api_schemas", []):
            prim.RemoveAppliedSchema(schema)
        for schema in step.get("apply_api_schemas", []):
            prim.AddAppliedSchema(schema)
        if "type_name" in step:
            prim.SetTypeName(step["type_name"])
        for name, config in step.get("attributes", {}).items():
            value_type, value = _typed_value(config)
            prim.CreateAttribute(name, value_type, custom=config.get("custom", True)).Set(value)
        if "active" in step:
            prim.SetActive(step["active"])


def _preview_surface(stage, step) -> None:
    from pxr import Sdf, UsdShade

    root = stage.GetDefaultPrim().GetPath()
    path = Sdf.Path(step["path"])
    if not path.IsAbsolutePath():
        path = root.AppendPath(path)
    assert path.IsPrimPath() and path.HasPrefix(root), "Material must stay inside the asset"
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path.AppendChild("Surface"))
    shader.CreateIdAttr("UsdPreviewSurface")
    for name, config in step["inputs"].items():
        value_type, value = _typed_value(config)
        shader.CreateInput(name, value_type).Set(value)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    for target in step.get("bind_to", []):
        binding = UsdShade.MaterialBindingAPI.Apply(asset_prim(stage, target))
        if step.get("stronger_than_descendants", False):
            binding.Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
        else:
            binding.Bind(material)


def _fixed_joint(stage, step) -> None:
    from pxr import Sdf, UsdPhysics

    root = stage.GetDefaultPrim().GetPath()
    path = Sdf.Path(step["path"])
    if not path.IsAbsolutePath():
        path = root.AppendPath(path)
    assert path.IsPrimPath() and path.HasPrefix(root), "Joint must stay inside the asset"
    joint = UsdPhysics.FixedJoint.Define(stage, path)
    if "body0" in step:
        joint.CreateBody0Rel().SetTargets([asset_prim(stage, step["body0"]).GetPath()])
    joint.CreateBody1Rel().SetTargets([asset_prim(stage, step["body1"]).GetPath()])


def _joint_drive(stage, step) -> None:
    from pxr import UsdPhysics

    assert step["instance"] in ("linear", "angular"), "Expected a linear or angular drive"
    prim = asset_prim(stage, step["path"])
    assert prim.IsA(UsdPhysics.Joint), "Drive target must be a joint"
    drive = UsdPhysics.DriveAPI.Apply(prim, step["instance"])
    attributes = {
        "type": drive.CreateTypeAttr,
        "target_position": drive.CreateTargetPositionAttr,
        "stiffness": drive.CreateStiffnessAttr,
        "damping": drive.CreateDampingAttr,
        "max_force": drive.CreateMaxForceAttr,
    }
    for name, value in step["settings"].items():
        assert name in attributes, f"Unsupported drive setting: {name}"
        attributes[name](value)


def apply_step(stage, step: Mapping[str, Any]) -> None:
    """Apply one recipe step to a USD stage without depending on an environment package.

    Args:
        stage: Editable stage whose default prim contains the asset.
        step: Operation name and operation-specific selectors and parameters.
    """
    from isaaclab_arena.utils.usd.fixed_articulations import create_fixed_articulations
    from isaaclab_arena.utils.usd.mesh_partition import inspect_mesh_components, partition_mesh

    assert isinstance(step, Mapping), "Derivation step must be a mapping"
    operation = step["operation"]
    if operation == "partition_mesh":
        if expected := step.get("expected_geometry_sha256"):
            report = inspect_mesh_components(stage, step["prim"])
            assert report["geometry_sha256"] == expected, f"Mesh geometry changed: {step['prim']}"
        partition_mesh(stage, step["prim"], step["groups"], remainder=step.get("remainder", "Remaining"))
    elif operation == "fixed_articulations":
        create_fixed_articulations(stage, step["groups"])
    elif operation == "edit_prims":
        _edit_prims(stage, step)
    elif operation == "preview_surface":
        _preview_surface(stage, step)
    elif operation == "fixed_joint":
        _fixed_joint(stage, step)
    elif operation == "joint_drive":
        _joint_drive(stage, step)
    else:
        raise AssertionError(f"Unknown asset derivation operation: {operation}")

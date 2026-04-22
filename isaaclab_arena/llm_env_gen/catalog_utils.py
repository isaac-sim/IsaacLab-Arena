# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for generating and loading object catalogs from USD files.

Adapted from robolab/assets/objects/_utils/common.py and
robolab/core/utils/usd_utils.py for use within Arena's scene_gen pipeline.

These functions extract metadata (dims, physics properties, semantic labels)
from individual USD files — no Isaac Sim runtime required (only ``pxr``).
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Any

from pxr import Usd, UsdGeom, UsdPhysics

# ---------------------------------------------------------------------------
# Arena-specific paths
# ---------------------------------------------------------------------------

ARENA_ROOT = Path(__file__).resolve().parents[2]  # IsaacLab-Arena repo root
OBJECT_CATALOG_PATH = ARENA_ROOT / "isaaclab_arena" / "scene_gen" / "object_catalog.json"

# ---------------------------------------------------------------------------
# USD file discovery
# ---------------------------------------------------------------------------


def find_usd_files(
    root: Path,
    recursive: bool = True,
    exclude_underscore_dirs: bool = True,
    exclude_materials: bool = True,
) -> list[Path]:
    """Find all USD files under *root*.

    Args:
        root: Directory (or single file) to search.
        recursive: Search subdirectories.
        exclude_underscore_dirs: Skip directories starting with ``_``.
        exclude_materials: Skip ``materials`` directories.

    Returns:
        Sorted list of USD file paths.
    """
    root = Path(root)
    if root.is_file():
        return [root] if root.suffix.lower() in {".usd", ".usda", ".usdc", ".usdz"} else []
    if not root.is_dir():
        return []

    usd_files: list[Path] = []
    extensions = {".usd", ".usda", ".usdc", ".usdz"}

    for ext in extensions:
        pattern = f"**/*{ext}" if recursive else f"*{ext}"
        for fp in root.glob(pattern):
            if not fp.is_file():
                continue
            try:
                rel_parts = fp.relative_to(root).parts[:-1]
            except ValueError:
                rel_parts = ()
            if exclude_underscore_dirs and any(p.startswith("_") for p in rel_parts):
                continue
            if exclude_materials and any(p.lower() == "materials" for p in rel_parts):
                continue
            usd_files.append(fp)

    return sorted(set(usd_files))


def iter_object_files(root: Path) -> list[Path]:
    """Return all USD files under *root* (convenience wrapper)."""
    return find_usd_files(root, recursive=True, exclude_underscore_dirs=True, exclude_materials=True)


# ---------------------------------------------------------------------------
# USD introspection helpers
# ---------------------------------------------------------------------------


def get_aabb(body_prim: Usd.Prim) -> tuple[np.ndarray, np.ndarray]:
    """Compute the local Axis-Aligned Bounding Box for a USD prim.

    Returns:
        ``(lower, upper)`` — each a ``(3,)`` numpy array.
    """
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
    bbox_cache.Clear()
    prim_bbox = bbox_cache.ComputeLocalBound(body_prim)
    prim_range = prim_bbox.ComputeAlignedRange()
    return np.array(prim_range.GetMin()), np.array(prim_range.GetMax())


def get_attribute_by_name(prim: Usd.Prim, name: str) -> str:
    """Search *prim* and its direct children for an attribute, metadata, or
    assetInfo/customData entry with the given *name*."""
    for p in [prim, *list(prim.GetChildren())]:
        for attr in p.GetAttributes():
            if attr.GetName() == name:
                val = attr.Get()
                if val:
                    return val
        doc = p.GetMetadata("description")
        if doc:
            return doc
        for accessor in ("GetAssetInfoByKey", "GetCustomDataByKey"):
            try:
                val = getattr(p, accessor)(name)
            except Exception:
                val = None
            if val:
                return val
    return ""


def get_prim_payload(prim: Usd.Prim, as_string: bool = True) -> list:
    """Get payload asset paths for a prim."""
    payload_list: list = []
    for prim_spec in prim.GetPrimStack():
        payloads = prim_spec.payloadList.prependedItems
        if as_string:
            payload_list.extend(payload.assetPath for payload in payloads)
        else:
            payload_list.extend(payloads)
    return payload_list


def _find_physics_material(parent_prim: Usd.Prim, stage: Usd.Stage) -> Usd.Prim | None:
    """Recursively search mesh children for a bound physics material."""
    for child in parent_prim.GetAllChildren():
        if child.IsA(UsdGeom.Mesh):
            for rel_name in (
                "material:binding:physics",
                "material:binding",
                "physxMaterial:physicsMaterial",
                "physics:material:binding",
            ):
                if child.HasRelationship(rel_name):
                    targets = child.GetRelationship(rel_name).GetTargets()
                    if targets:
                        mat = stage.GetPrimAtPath(targets[0].pathString)
                        if mat.IsValid() and mat.HasAPI(UsdPhysics.MaterialAPI):
                            return mat
        result = _find_physics_material(child, stage)
        if result:
            return result
    return None


def _get_friction(prim: Usd.Prim, stage: Usd.Stage) -> dict[str, Any] | None:
    """Extract physics material properties (friction, restitution, density)."""
    mat = _find_physics_material(prim, stage)
    if mat is None:
        return None
    api = UsdPhysics.MaterialAPI(mat)
    result: dict[str, Any] = {"prim_path": mat.GetPath().pathString}
    for attr_name, key in (
        ("GetStaticFrictionAttr", "static_friction"),
        ("GetDynamicFrictionAttr", "dynamic_friction"),
        ("GetRestitutionAttr", "restitution"),
        ("GetDensityAttr", "density"),
    ):
        attr = getattr(api, attr_name)()
        if attr and attr.IsValid():
            result[key] = attr.Get()
    return result


# ---------------------------------------------------------------------------
# Main catalog entry point
# ---------------------------------------------------------------------------


def get_usd_rigid_body_info(usd_path: str) -> dict[str, Any]:
    """Analyse a single-object USD file and return a metadata dict.

    Fields returned: ``name``, ``usd_path``, ``prim_path``, ``position``,
    ``quat_wxyz``, ``payload``, ``rigid_body``, ``static_body``, ``class``,
    ``description``, ``dims``, ``mass``, ``density``, ``dynamic_friction``,
    ``static_friction``, ``restitution``.
    """
    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        raise ValueError(f"Failed to open stage: {usd_path}")

    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        raise ValueError(f"No default prim found in: {usd_path}")

    lower, upper = get_aabb(default_prim)
    dims = upper - lower
    size = [float(dims[i]) for i in range(3)]

    description = get_attribute_by_name(default_prim, "description")
    class_name = get_attribute_by_name(default_prim, "class")

    rigid_body_api = UsdPhysics.RigidBodyAPI(default_prim)
    rigid_body = bool(rigid_body_api and rigid_body_api.GetRigidBodyEnabledAttr().Get())

    mass = density = None
    mass_api = UsdPhysics.MassAPI(default_prim)
    if mass_api:
        m_attr = mass_api.GetMassAttr()
        if m_attr and m_attr.IsValid():
            mass = m_attr.Get()
        d_attr = mass_api.GetDensityAttr()
        if d_attr and d_attr.IsValid():
            density = d_attr.Get()

    friction_info = _get_friction(default_prim, stage)
    dynamic_friction = friction_info.get("dynamic_friction") if friction_info else None
    static_friction = friction_info.get("static_friction") if friction_info else None
    restitution = friction_info.get("restitution") if friction_info else None
    if (density is None or density == 0) and friction_info:
        density = friction_info.get("density")

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    world_xform = xform_cache.GetLocalToWorldTransform(default_prim)
    pos = world_xform.ExtractTranslation()
    rot = world_xform.ExtractRotation().GetQuat()
    rot_quat = (rot.GetReal(), rot.GetImaginary()[0], rot.GetImaginary()[1], rot.GetImaginary()[2])

    return {
        "name": default_prim.GetName(),
        "usd_path": str(usd_path),
        "prim_path": str(default_prim.GetPath()),
        "position": (pos[0], pos[1], pos[2]),
        "quat_wxyz": rot_quat,
        "payload": get_prim_payload(default_prim, as_string=True),
        "rigid_body": rigid_body,
        "static_body": not rigid_body,
        "class": class_name,
        "description": description,
        "dims": size,
        "mass": mass,
        "density": density,
        "dynamic_friction": dynamic_friction,
        "static_friction": static_friction,
        "restitution": restitution,
    }


# ---------------------------------------------------------------------------
# Catalog I/O helpers
# ---------------------------------------------------------------------------


def load_catalog(catalog_path: Path = OBJECT_CATALOG_PATH) -> list[dict[str, Any]]:
    """Load the object catalog JSON file."""
    with open(catalog_path) as f:
        return json.load(f)


def get_dataset_from_path(usd_path: str) -> str:
    """Extract dataset name from a USD path (e.g. ``…/objects/vomp/obj.usd`` → ``vomp``)."""
    parts = str(usd_path).replace("\\", "/").split("/")
    if "objects" in parts:
        idx = parts.index("objects")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "unknown"


def print_object_info(object_info: dict[str, Any], usd_path: Path) -> None:
    """Pretty-print an object info dictionary."""
    print(f"Object: {usd_path.name}, usd_path: {usd_path}")
    for key, value in object_info.items():
        if isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], (int, float)):
            formatted = f"[{', '.join(f'{v:.4f}' if isinstance(v, float) else str(v) for v in value)}]"
        else:
            formatted = str(value)
        print(f"  {key:20s}: {formatted}")

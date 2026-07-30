# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Turn a multi-part USD asset into a single rigid body.

SimReady props give each part its own rigid body and hold the parts together with fixed joints:
a spray bottle is a bottle plus a cap. Isaac Lab expects exactly one rigid body per rigid object,
so it refuses to spawn such a prop even though nothing about it can move.

The fix is to put the rigid body on a parent prim instead of on each part, and switch off the
joints that are no longer needed.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from isaaclab.utils.assets import retrieve_file_path
from pxr import Usd, UsdGeom, UsdPhysics

from isaaclab_arena.assets.asset_cache import get_arena_asset_cache_dir
from isaaclab_arena.utils.usd.physics_structure import PhysicsStructure, get_physics_structure_from_usd
from isaaclab_arena.utils.usd.rigid_body_mass import (
    combine_rigid_body_masses,
    read_rigid_body_mass,
    write_rigid_body_mass,
)
from isaaclab_arena.utils.usd_helpers import apply_usd_variant_selections

_CACHE_DIR_NAME = "merged_rigid_body"
_ROOT_PRIM_PATH = "/Prop"
_MERGE_VERSION = 2
"""Bump this when the merge changes, so old cached files are not reused."""

_merged_paths: dict[tuple[str, tuple[tuple[str, str], ...]], str] = {}
"""What each source asset and variant selection resolved to, so a repeat asset costs nothing."""


def _cache_file_name(usd_path: str, variant_items: tuple[tuple[str, str], ...]) -> str:
    """Name the output after the source asset plus a hash of everything that affects it."""
    payload = json.dumps([usd_path, list(variant_items), _MERGE_VERSION])
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"{Path(usd_path).stem}_merged_{digest}.usd"


def _add_source_reference(
    stage: Usd.Stage, source_usd_path: str, variant_items: tuple[tuple[str, str], ...]
) -> Usd.Prim:
    """Point the stage's root prim at the source asset and select its variants."""
    root = stage.DefinePrim(_ROOT_PRIM_PATH, "Xform")
    root.GetReferences().AddReference(source_usd_path)
    stage.SetDefaultPrim(root)
    apply_usd_variant_selections(stage, dict(variant_items))
    return root


def _needs_merge(structure: PhysicsStructure) -> bool:
    """True if the asset has several bodies that fixed joints already hold together."""
    return structure.is_single_rigid_body and len(structure.rigid_body_paths) > 1


def _combined_rigid_body_mass(stage: Usd.Stage, root: Usd.Prim, structure: PhysicsStructure):
    """Add up what the parts weigh, or None if any part does not say."""
    transform_cache = UsdGeom.XformCache()
    parts = []
    for body_path in structure.rigid_body_paths:
        body_mass = read_rigid_body_mass(stage.GetPrimAtPath(body_path))
        if body_mass is None:
            return None
        into_root = transform_cache.ComputeRelativeTransform(stage.GetPrimAtPath(body_path), root)[0]
        parts.append((body_mass, into_root))
    return combine_rigid_body_masses(parts)


def _apply_merge(stage: Usd.Stage, root: Usd.Prim, structure: PhysicsStructure) -> None:
    """Move the rigid body from the parts up to the root prim and switch off the joints."""
    # Apply the RigidBodyAPI to the root prim.
    UsdPhysics.RigidBodyAPI.Apply(root)
    combined = _combined_rigid_body_mass(stage, root, structure)
    for body_path in structure.rigid_body_paths:
        part = stage.GetPrimAtPath(body_path)
        assert part, f"rigid body {body_path} is missing from the stage"
        assert part.RemoveAPI(UsdPhysics.RigidBodyAPI), f"could not remove RigidBodyAPI from {body_path}"
    # Fixed joints are no longer needed, so disable them.
    for joint in structure.joints:
        stage.GetPrimAtPath(joint.path).SetActive(False)
    # Write what the parts weigh together onto the root prim.
    if combined is not None:
        write_rigid_body_mass(root, combined)


def merge_to_one_rigid_body_if_needed(usd_path: str, variants: dict[str, str] | None = None) -> str:
    """Return a USD that spawns as one rigid body, merging the source asset's parts if multiple
    RigidBodyAPI are found connected by fixed joints.
    Args:
        usd_path: The source asset, local or remote.
        variants: USD variants to select before looking at the physics. SimReady props need
            ``{"Physics": "physics"}``, or they have no physics at all.

    Returns:
        Path to the merged asset, or usd_path if nothing needed merging.
    """
    variant_items = tuple(sorted((variants or {}).items()))
    key = (usd_path, variant_items)
    if key in _merged_paths:
        return _merged_paths[key]

    cache_dir = get_arena_asset_cache_dir().parent / "usd" / _CACHE_DIR_NAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / _cache_file_name(usd_path, variant_items)
    if not output_path.exists():
        source_usd_path = retrieve_file_path(usd_path)
        check_stage = Usd.Stage.CreateInMemory()
        _add_source_reference(check_stage, source_usd_path, variant_items)
        if not _needs_merge(get_physics_structure_from_usd(check_stage)):
            _merged_paths[key] = usd_path
            return usd_path

        # Write to a temporary file first, so a failure part way through leaves no broken cache file.
        with tempfile.NamedTemporaryFile(suffix=".usd", dir=cache_dir, delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        try:
            stage = Usd.Stage.CreateNew(str(temp_path))
            root = _add_source_reference(stage, source_usd_path, variant_items)
            _apply_merge(stage, root, get_physics_structure_from_usd(stage))
            result = get_physics_structure_from_usd(stage)
            assert result.rigid_body_paths == (
                _ROOT_PRIM_PATH,
            ), f"merging {usd_path} left these rigid bodies: {result.rigid_body_paths}"
            assert stage.GetRootLayer().Save(), f"could not save the merged asset for {usd_path}"
            os.replace(temp_path, output_path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    _merged_paths[key] = str(output_path)
    return str(output_path)

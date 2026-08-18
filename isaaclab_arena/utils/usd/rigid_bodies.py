# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils.assets import retrieve_file_path

if TYPE_CHECKING:
    from pxr import Usd


def get_all_rigid_body_prim_paths_from_stage(stage: Usd.Stage) -> list[str]:
    """
    Get the prim paths of all rigid bodies in a stage.

    Args:
        stage: The stage to analyze

    Returns:
        List of prim paths of all rigid bodies in the stage
    """
    # Avoid loading pxr at module scope before SimulationApp starts, which happens in unit tests.
    from pxr import UsdPhysics

    rigid_body_prim_paths = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_body_prim_paths.append(str(prim.GetPath()))
    return rigid_body_prim_paths


def get_all_rigid_body_prim_paths(usd_path: str) -> list[str]:
    """
    Get the prim paths of all rigid bodies in a USD file.

    Args:
        usd_path: Path to the USD file to analyze

    Returns:
        List of prim paths of all rigid bodies in the USD file
    """
    # Avoid loading pxr at module scope before SimulationApp starts, which happens in unit tests.
    from pxr import Usd

    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise ValueError(f"Error: Could not open USD file at {usd_path}")
    return get_all_rigid_body_prim_paths_from_stage(stage)


_RIGID_BODY_PATHS_BY_ASSET: dict[tuple[str, tuple[tuple[str, str], ...]], list[str]] = {}
"""Rigid body paths already read, keyed by asset path and variant selection."""


def read_asset_rigid_body_paths(usd_path: str, variants: dict[str, str] | None = None) -> list[str]:
    """Get the prim paths of the rigid bodies in an asset, fetching it first if it is remote.

    The asset is referenced into a throwaway stage instead of being opened, so selecting a variant
    leaves the asset itself untouched. Results are cached, because fetching a remote asset walks
    the whole remote tree.

    Args:
        usd_path: Path to the USD file to analyze, local or remote.
        variants: USD variants to select before looking for rigid bodies. SimReady props need
            ``{"Physics": "physics"}``, or they have no physics at all.

    Returns:
        Prim paths of the asset's rigid bodies, empty if it has none.
    """
    # Avoid loading pxr at module scope before SimulationApp starts, which happens in unit tests.
    from pxr import Usd

    variants = variants or {}
    key = (usd_path, tuple(sorted(variants.items())))
    if key not in _RIGID_BODY_PATHS_BY_ASSET:
        stage = Usd.Stage.CreateInMemory()
        root = stage.DefinePrim("/Asset", "Xform")
        root.GetReferences().AddReference(retrieve_file_path(usd_path))
        stage.SetDefaultPrim(root)
        if variants:
            apply_usd_variant_selections(stage, variants)
        _RIGID_BODY_PATHS_BY_ASSET[key] = get_all_rigid_body_prim_paths_from_stage(stage)
    return list(_RIGID_BODY_PATHS_BY_ASSET[key])


def apply_usd_variant_selections(stage: Usd.Stage, variants: dict[str, str]) -> None:
    """Select USD variants on the stage's default prim.

    Args:
        stage: Open USD stage.
        variants: Maps each variant set name to the variant to select from it.
    """
    root = stage.GetDefaultPrim()
    if not root:
        return
    variant_sets = root.GetVariantSets()
    for set_name, selection in variants.items():
        if set_name in variant_sets.GetNames():
            variant_set = variant_sets.GetVariantSet(set_name)
            if selection in variant_set.GetVariantNames():
                variant_set.SetVariantSelection(selection)


def _path_relative_to_usd_root(prim_path: str) -> str:
    """Strip the USD root prim name, returning a path suffix suitable for contact sensors."""
    assert prim_path[0] == "/", "We expect USD paths to start with a /"
    root_and_rest = prim_path.lstrip("/").split("/", 1)
    if len(root_and_rest) == 1:
        return ""
    return "/" + root_and_rest[1]


def find_shallowest_rigid_body_from_stage(stage: Usd.Stage, relative_to_root: bool = False) -> str | None:
    """
    Find the shallowest (closest to root) prim that is a rigid body.
    Also verifies that there is only one rigid body at that depth level.

    Args:
        stage: The stage to analyze
        relative_to_root: Whether to return the path relative to the root of the USD file

    Returns:
        Prim path for the shallowest rigid body. None if no rigid bodies are found.
        Empty string if the shallowest rigid body is the root prim, and
        relative_to_root is True.

    Raises:
        ValueError: If multiple rigid bodies exist at the shallowest level
    """
    rigid_body_prim_paths = get_all_rigid_body_prim_paths_from_stage(stage)

    if len(rigid_body_prim_paths) == 0:
        return None

    if len(rigid_body_prim_paths) == 1:
        shallowest_rigid_body = rigid_body_prim_paths[0]

    else:
        # Group the rigid bodies by depth
        rigid_bodies_by_depth = {}
        for prim_path in rigid_body_prim_paths:
            depth = prim_path.count("/") - 1
            if depth not in rigid_bodies_by_depth:
                rigid_bodies_by_depth[depth] = []
            rigid_bodies_by_depth[depth].append(prim_path)

        # Find the shallowest depth
        min_depth = min(rigid_bodies_by_depth.keys())
        shallowest_rigid_bodies = rigid_bodies_by_depth[min_depth]

        # Check if there's only one rigid body at the shallowest level
        if len(shallowest_rigid_bodies) > 1:
            raise ValueError(
                f"Found {len(shallowest_rigid_bodies)} rigid bodies at depth {min_depth}. "
                f"Expected only one. Rigid bodies at this level: {shallowest_rigid_bodies}"
            )
        shallowest_rigid_body = shallowest_rigid_bodies[0]

    if relative_to_root:
        shallowest_rigid_body = _path_relative_to_usd_root(shallowest_rigid_body)
    return shallowest_rigid_body


def find_shallowest_rigid_body(
    usd_path: str,
    relative_to_root: bool = False,
    variants: dict[str, str] | None = None,
) -> str | None:
    """
    Find the shallowest (closest to root) prim that is a rigid body.
    Also verifies that there is only one rigid body at that depth level.

    Args:
        usd_path: Path to the USD file to analyze
        relative_to_root: Whether to return the path relative to the root of the USD file
        variants: USD variants to select before searching. SimReady props need
            ``{"Physics": "physics"}``, or they have no physics at all.

    Returns:
        Prim path for the shallowest rigid body. None if no rigid bodies are found.
        Empty string if the shallowest rigid body is the root prim, and
        relative_to_root is True.

    Raises:
        ValueError: If multiple rigid bodies exist at the shallowest level
    """
    # Avoid loading pxr at module scope before SimulationApp starts, which happens in unit tests.
    from pxr import Usd

    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise ValueError(f"Error: Could not open USD file at {usd_path}")
    if variants:
        apply_usd_variant_selections(stage, variants)
    return find_shallowest_rigid_body_from_stage(stage, relative_to_root)

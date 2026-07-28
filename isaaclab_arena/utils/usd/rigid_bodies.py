# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from pxr import Usd, UsdPhysics


def get_all_rigid_body_prim_paths_from_stage(stage: Usd.Stage) -> list[str]:
    """
    Get the prim paths of all rigid bodies in a stage.

    Args:
        stage: The stage to analyze

    Returns:
        List of prim paths of all rigid bodies in the stage
    """
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
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise ValueError(f"Error: Could not open USD file at {usd_path}")
    return get_all_rigid_body_prim_paths_from_stage(stage)


def apply_usd_variant_selections(stage: Usd.Stage, variants: dict[str, str] | None) -> None:
    """Select USD variants on the stage's default prim.

    Variant sets and selections that the asset does not have are skipped.

    Args:
        stage: Open USD stage.
        variants: Variant set name to the variant to select.
    """
    if not variants:
        return
    root = stage.GetDefaultPrim()
    if not root:
        return
    variant_sets = root.GetVariantSets()
    for set_name, selection in variants.items():
        if set_name in variant_sets.GetNames():
            variant_set = variant_sets.GetVariantSet(set_name)
            if selection in variant_set.GetVariantNames():
                variant_set.SetVariantSelection(selection)


def _prefer_body_rigid_body(prim_paths: list[str]) -> str:
    """Prefer a prim whose leaf name contains ``body`` when several RBs tie on depth."""
    body_paths = [path for path in prim_paths if "body" in path.rsplit("/", 1)[-1].lower()]
    if body_paths:
        return sorted(body_paths)[0]
    return sorted(prim_paths)[0]


def _path_relative_to_usd_root(prim_path: str) -> str:
    """Strip the USD root prim name, returning a path suffix suitable for contact sensors."""
    assert prim_path[0] == "/", "We expect USD paths to start with a /"
    root_and_rest = prim_path.lstrip("/").split("/", 1)
    if len(root_and_rest) == 1:
        return ""
    return "/" + root_and_rest[1]


def find_shallowest_rigid_body_from_stage(
    stage: Usd.Stage,
    relative_to_root: bool = False,
    *,
    prefer_body_when_tied: bool = False,
) -> str | None:
    """
    Find the shallowest (closest to root) prim that is a rigid body.
    Also verifies that there is only one rigid body at that depth level.

    Args:
        stage: The stage to analyze
        relative_to_root: Whether to return the path relative to the root of the USD file
        prefer_body_when_tied: When several rigid bodies share the shallowest depth, prefer a
            prim whose leaf name contains ``body`` instead of raising.

    Returns:
        Prim path for the shallowest rigid body. None if no rigid bodies are found.
        Empty string if the shallowest rigid body is the root prim, and
        relative_to_root is True.

    Raises:
        ValueError: If multiple rigid bodies exist at the shallowest level and
            ``prefer_body_when_tied`` is False
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
            if prefer_body_when_tied:
                shallowest_rigid_body = _prefer_body_rigid_body(shallowest_rigid_bodies)
            else:
                raise ValueError(
                    f"Found {len(shallowest_rigid_bodies)} rigid bodies at depth {min_depth}. "
                    f"Expected only one. Rigid bodies at this level: {shallowest_rigid_bodies}"
                )
        else:
            shallowest_rigid_body = shallowest_rigid_bodies[0]

    if relative_to_root:
        shallowest_rigid_body = _path_relative_to_usd_root(shallowest_rigid_body)
    return shallowest_rigid_body


def find_shallowest_rigid_body(
    usd_path: str,
    relative_to_root: bool = False,
    *,
    variants: dict[str, str] | None = None,
    prefer_body_when_tied: bool = False,
) -> str | None:
    """
    Find the shallowest (closest to root) prim that is a rigid body.
    Also verifies that there is only one rigid body at that depth level.

    Args:
        usd_path: Path to the USD file to analyze
        relative_to_root: Whether to return the path relative to the root of the USD file
        variants: Optional USD variant selections applied before searching (e.g. SimReady
            ``{\"Physics\": \"physics\"}``).
        prefer_body_when_tied: When several rigid bodies share the shallowest depth, prefer a
            prim whose leaf name contains ``body`` instead of raising.

    Returns:
        Prim path for the shallowest rigid body. None if no rigid bodies are found.
        Empty string if the shallowest rigid body is the root prim, and
        relative_to_root is True.

    Raises:
        ValueError: If multiple rigid bodies exist at the shallowest level and
            ``prefer_body_when_tied`` is False
    """
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise ValueError(f"Error: Could not open USD file at {usd_path}")
    apply_usd_variant_selections(stage, variants)
    return find_shallowest_rigid_body_from_stage(
        stage,
        relative_to_root,
        prefer_body_when_tied=prefer_body_when_tied,
    )

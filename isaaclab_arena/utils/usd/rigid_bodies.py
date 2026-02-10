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
        assert shallowest_rigid_body[0] == "/", "We expect USD paths to start with a /"
        root_and_rest = shallowest_rigid_body.lstrip("/").split("/", 1)
        if len(root_and_rest) == 1:
            shallowest_rigid_body = ""
        else:
            shallowest_rigid_body = "/" + root_and_rest[1]
    return shallowest_rigid_body


def find_shallowest_rigid_body(usd_path: str, relative_to_root: bool = False) -> str | None:
    """
    Find the shallowest (closest to root) prim that is a rigid body.
    Also verifies that there is only one rigid body at that depth level.

    Args:
        usd_path: Path to the USD file to analyze
        relative_to_root: Whether to return the path relative to the root of the USD file

    Returns:
        Prim path for the shallowest rigid body. None if no rigid bodies are found.
        Empty string if the shallowest rigid body is the root prim, and
        relative_to_root is True.

    Raises:
        ValueError: If multiple rigid bodies exist at the shallowest level
    """
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise ValueError(f"Error: Could not open USD file at {usd_path}")
    return find_shallowest_rigid_body_from_stage(stage, relative_to_root)

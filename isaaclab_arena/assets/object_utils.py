# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from pxr import Usd

from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.utils.usd_helpers import get_prim_depth, is_articulation_root, is_rigid_body


def detect_object_type(usd_path: str | None = None, stage: Usd.Stage | None = None) -> ObjectType:
    """Detect the object type of the asset

    Goes through the USD tree and detects the object type. The detection is based
    on the presence of a RigidBodyAPI or ArticulationRootAPI at the shallowest depth
    in which one of these APIs is present.

    Note that if more than one API is present on that shallowest depth, we raise an error.

    Args:
        usd_path: The path to the USD file to inspect. Either this or stage must be provided.
        stage: The stage to inspect. Either this or usd_path must be provided.

    Returns:
        The object type of the asset.
    """
    assert usd_path is not None or stage is not None, "Either usd_path or stage must be provided"
    assert usd_path is None or stage is None, "Either usd_path or stage must be provided"
    if usd_path is not None:
        # Open a stage to inspect the USD.
        stage = Usd.Stage.Open(usd_path)
    # We do a Breadth First Search (BFS) through the prims, until we find either
    # a rigid body or an articulation root. At that point, we continue searching
    # the rest of the prims at that depth, to ensure that there's nothing else.
    # If we find more than one, we raise an error.
    open_prims = [stage.GetPseudoRoot()]
    found = False
    found_depth = -1
    interesting_prim = None
    while len(open_prims) > 0:
        # Update the DFS list
        prim = open_prims.pop(0)
        open_prims.extend(prim.GetChildren())
        # Check if we found an interesting prim on this level
        if is_articulation_root(prim) or is_rigid_body(prim):
            if found:
                raise ValueError(f"Found multiple rigid body or articulation roots at depth {get_prim_depth(prim)}")
            found_depth = get_prim_depth(prim)
            found = True
            interesting_prim = prim
        if found and get_prim_depth(prim) > found_depth:
            break
    if not found:
        return ObjectType.BASE
    if found and is_rigid_body(interesting_prim):
        return ObjectType.RIGID
    if found and is_articulation_root(interesting_prim):
        return ObjectType.ARTICULATION
    else:
        raise ValueError("This should not happen. There is an unknown USD type in the tree.")

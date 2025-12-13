# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager

from pxr import Usd, UsdLux, UsdPhysics


def get_all_prims(
    stage: Usd.Stage, prim: Usd.Prim | None = None, prims_list: list[Usd.Prim] | None = None
) -> list[Usd.Prim]:
    """Get all prims in the stage.

    Performs a Depth First Search (DFS) through the prims in a stage
    and returns all the prims.

    Args:
        stage: The stage to get the prims from.
        prim: The prim to start the search from. Defaults to the pseudo-root.
        prims_list: The list to store the prims in. Defaults to an empty list.

    Returns:
        A list of prims in the stage.
    """
    if prims_list is None:
        prims_list = []
    if prim is None:
        prim = stage.GetPseudoRoot()
    for child in prim.GetAllChildren():
        prims_list.append(child)
        get_all_prims(stage, child, prims_list)
    return prims_list


def has_light(stage: Usd.Stage) -> bool:
    """Check if the stage has a light"""
    LIGHT_TYPES = (
        UsdLux.SphereLight,
        UsdLux.RectLight,
        UsdLux.DomeLight,
        UsdLux.DistantLight,
        UsdLux.DiskLight,
    )
    has_light = False
    all_prims = get_all_prims(stage)
    for prim in all_prims:
        if any(prim.IsA(t) for t in LIGHT_TYPES):
            has_light = True
            break
    return has_light


def is_articulation_root(prim: Usd.Prim) -> bool:
    """Check if prim is articulation root"""
    return prim.HasAPI(UsdPhysics.ArticulationRootAPI)


def is_rigid_body(prim: Usd.Prim) -> bool:
    """Check if prim is rigidbody"""
    return prim.HasAPI(UsdPhysics.RigidBodyAPI)


def get_prim_depth(prim: Usd.Prim) -> int:
    """Get the depth of a prim"""
    return len(str(prim.GetPath()).split("/")) - 2


@contextmanager
def open_stage(path):
    """Open a stage and ensure it is closed after use."""
    stage = Usd.Stage.Open(path)
    try:
        yield stage
    finally:
        # Drop the local reference; Garbage Collection will reclaim once no prim/attr handles remain
        del stage


def get_asset_usd_path_from_prim_path(prim_path: str, stage: Usd.Stage) -> str | None:
    """Get the USD path from a prim path, that is referring to an asset."""
    # Note (xinjieyao, 2025.12.12): preferred way to get the composed asset path is to ask the Usd.Prim object itself,
    # which handles the entire composition stack. Here it achieved this goal thru root layer due to the USD API limitations.
    # It only finds references authored on the root layer.
    # If the asset was referenced in an intermediate sublayer, this method would fail to find the asset path.
    root_layer = stage.GetRootLayer()
    prim_spec = root_layer.GetPrimAtPath(prim_path)
    if not prim_spec:
        return None

    try:
        reference_list = prim_spec.referenceList.GetAddedOrExplicitItems()
    except Exception as e:
        print(f"Failed to get reference list for prim {prim_path}: {e}")
        return None
    if len(reference_list) > 0:
        for reference_spec in reference_list:
            if reference_spec.assetPath:
                return reference_spec.assetPath

    return None

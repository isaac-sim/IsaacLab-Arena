# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager

from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox


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
    all_prims = get_all_prims(stage)
    return any(any(prim.IsA(t) for t in LIGHT_TYPES) for prim in all_prims)


def usd_file_has_light(usd_path: str) -> bool:
    """Check if a USD file contains any lights.

    Args:
        usd_path: Path to the USD file.

    Returns:
        True if the USD file contains lights, False otherwise.
    """
    with open_stage(usd_path) as stage:
        return has_light(stage)


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


def compute_local_bounding_box_from_usd(
    usd_path: str,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> AxisAlignedBoundingBox:
    """Compute the local bounding box of a USD asset (relative to USD origin).

    Args:
        usd_path: Path to the USD file.
        scale: Scale to apply to the asset (x, y, z).

    Returns:
        AxisAlignedBoundingBox containing the min and max points relative to USD origin.
    """
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise ValueError(f"Failed to open USD file: {usd_path}")

    # Get the default prim (or pseudo root if no default prim)
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        default_prim = stage.GetPseudoRoot()

    # Compute the bounding box using USD's built-in functionality
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(default_prim)

    # Get the range (bounding box)
    bbox_range = bbox.ComputeAlignedBox()
    min_point = bbox_range.GetMin()
    max_point = bbox_range.GetMax()

    # Apply scale
    min_point = Gf.Vec3d(min_point[0] * scale[0], min_point[1] * scale[1], min_point[2] * scale[2])
    max_point = Gf.Vec3d(max_point[0] * scale[0], max_point[1] * scale[1], max_point[2] * scale[2])

    return AxisAlignedBoundingBox(
        min_point=(min_point[0], min_point[1], min_point[2]),
        max_point=(max_point[0], max_point[1], max_point[2]),
    )


def compute_local_bounding_box_from_prim(
    stage: Usd.Stage,
    prim_path: str,
) -> AxisAlignedBoundingBox:
    """Compute the local bounding box of a specific prim (relative to prim's transform origin).

    Args:
        stage: The USD stage containing the prim.
        prim_path: Path to the prim to compute the bounding box for.

    Returns:
        AxisAlignedBoundingBox containing the local min and max points relative to the
        prim's own origin.

    Raises:
        ValueError: If the prim is not found at the given path.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"No prim found at path {prim_path}")

    # Compute the world-space bounding box of the prim
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(prim)
    bbox_range = bbox.ComputeAlignedBox()

    # Get world-space min/max
    world_min = bbox_range.GetMin()
    world_max = bbox_range.GetMax()

    # Get the target prim's world position to compute local bounding box
    prim_xformable = UsdGeom.Xformable(prim)
    prim_world_transform = prim_xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    prim_world_pos = prim_world_transform.ExtractTranslation()

    # Compute local bounding box by subtracting the prim's own world position
    local_min = Gf.Vec3d(
        world_min[0] - prim_world_pos[0],
        world_min[1] - prim_world_pos[1],
        world_min[2] - prim_world_pos[2],
    )
    local_max = Gf.Vec3d(
        world_max[0] - prim_world_pos[0],
        world_max[1] - prim_world_pos[1],
        world_max[2] - prim_world_pos[2],
    )

    return AxisAlignedBoundingBox(
        min_point=(local_min[0], local_min[1], local_min[2]),
        max_point=(local_max[0], local_max[1], local_max[2]),
    )


def isaaclab_prim_path_to_original_prim_path(isaaclab_prim_path: str, parent_asset: Asset, stage: Usd.Stage) -> str:
    """Convert an IsaacLab prim path to the prim path in the original USD stage.

    Two steps to getting the original prim path from the IsaacLab prim path.

    # 1. Remove the ENV_REGEX_NS prefix
    # 2. Replace the asset name with the default prim path.

    Args:
        isaaclab_prim_path: The IsaacLab prim path.

    Returns:
        The prim path in the original USD stage.
    """
    default_prim = stage.GetDefaultPrim()
    default_prim_path = default_prim.GetPath()
    assert default_prim_path is not None
    # Check that the path starts with the ENV_REGEX_NS prefix.
    assert isaaclab_prim_path.startswith("{ENV_REGEX_NS}/")
    original_prim_path = isaaclab_prim_path.removeprefix("{ENV_REGEX_NS}/")
    # Check that the path starts with the asset name.
    assert original_prim_path.startswith(parent_asset.name)
    original_prim_path = original_prim_path.removeprefix(parent_asset.name)
    # Append the default prim path.
    original_prim_path = str(default_prim_path) + original_prim_path
    return original_prim_path

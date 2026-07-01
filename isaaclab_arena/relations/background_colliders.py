# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Discover background fixtures that a placement solver should avoid.

Two steps keep the solver cheap: build a placement region from the anchors an object is
placed on, then keep only background fixtures whose bounding box intersects that region
instead of feeding every prim in the scene into the all-pairs no-overlap term.
"""

from __future__ import annotations

import torch

from pxr import Usd, UsdGeom

from isaaclab_arena.assets.background import Background
from isaaclab_arena.assets.object_base import ObjectBase
from isaaclab_arena.assets.object_reference import ObjectReference
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox, quaternion_to_90_deg_z_quarters
from isaaclab_arena.utils.usd_helpers import open_stage


def build_placement_region(
    anchors: list[ObjectBase],
    objects: list[ObjectBase],
    clearance_m: float = 0.0,
) -> AxisAlignedBoundingBox:
    """Axis-aligned region above the anchor surfaces where placed objects can land.

    Spans the anchors' combined XY footprint and rises from the lowest anchor point to the
    highest anchor top plus the tallest object's height. Background geometry outside this
    region cannot collide with anything placed on the anchors.

    Args:
        anchors: Fixed surfaces objects are placed on (e.g. a counter).
        objects: Objects being placed; their heights set how far up the region extends.
        clearance_m: Extra margin added to the top of the region.
    """
    assert anchors, "build_placement_region requires at least one anchor"
    anchor_boxes = [anchor.get_world_bounding_box() for anchor in anchors]
    # AxisAlignedBoundingBox tensors carry a leading batch dim; anchors are single-env, so
    # index [0] to reduce to a plain (3,) corner and size[0, 2] to read the Z extent.
    min_point = torch.stack([box.min_point[0] for box in anchor_boxes]).min(dim=0).values
    max_point = torch.stack([box.max_point[0] for box in anchor_boxes]).max(dim=0).values

    max_object_height = max((obj.get_bounding_box().size[0, 2].item() for obj in objects), default=0.0)
    max_point = max_point.clone()
    max_point[2] = max_point[2] + max_object_height + clearance_m
    return AxisAlignedBoundingBox(min_point=min_point, max_point=max_point)


def find_background_colliders(
    background: Background,
    region: AxisAlignedBoundingBox,
    anchors: list[ObjectBase] | None = None,
    object_prim_paths: list[str] | None = None,
) -> list[ObjectReference]:
    """Object-level background fixtures whose bounding box intersects the placement region.

    Args:
        background: The background scene to pull collision fixtures from.
        region: Placement region (see build_placement_region); fixtures outside it are dropped.
        anchors: Placement surfaces to exclude — an anchor is not an obstacle for the objects
            placed on it, and including it would make every placement register as overlapping.
        object_prim_paths: Optional explicit fixture prim paths (e.g. from an upstream task).
            When omitted, fixtures are discovered structurally from the background USD.
    """
    exclude_prim_paths = {anchor.prim_path for anchor in (anchors or []) if isinstance(anchor, ObjectReference)}

    # Culling happens against the background USD directly (one stage open) so that an
    # ObjectReference — which reopens the USD — is only built for the few surviving fixtures.
    survivor_prim_paths = _cull_prim_paths_to_region(background, region, object_prim_paths, exclude_prim_paths)
    return [
        ObjectReference(name=_leaf_name(prim_path), prim_path=prim_path, parent_asset=background)
        for prim_path in survivor_prim_paths
    ]


def _cull_prim_paths_to_region(
    background: Background,
    region: AxisAlignedBoundingBox,
    object_prim_paths: list[str] | None,
    exclude_prim_paths: set[str],
) -> list[str]:
    """Return the fixture prim paths whose world bounding box intersects the region.

    Opens the background USD once. Candidates (every first-level Xformable prim, or the
    explicit object_prim_paths) are kept when their box overlaps the region, except a box
    that fully encloses the region: it would contain every candidate placement, so the
    solver gets no gradient to avoid it (e.g. a whole-room shell prim). Spatial culling,
    not prim names, decides what counts — a nearby wall or backsplash is a real constraint.
    """
    assert background.usd_path is not None, "background must have a usd_path to discover colliders"
    position, quarters, scale = _background_world_transform(background)

    survivors: list[str] = []
    with open_stage(background.usd_path) as stage:
        default_prim = stage.GetDefaultPrim()
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
        for prim_path, prim in _candidate_prims(background, default_prim, object_prim_paths):
            if prim_path in exclude_prim_paths:
                continue
            aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if aligned.IsEmpty():
                continue
            local_min, local_max = aligned.GetMin(), aligned.GetMax()
            # Scale, then rotate, then translate -- the same world box an ObjectReference to this
            # prim would produce, so culling matches what the solver sees.
            world_bbox = (
                AxisAlignedBoundingBox(
                    min_point=(local_min[0], local_min[1], local_min[2]),
                    max_point=(local_max[0], local_max[1], local_max[2]),
                )
                .scaled(scale)
                .rotated_90_around_z(quarters)
                .translated(position)
            )
            if world_bbox.overlaps(region).item() and not _encloses(world_bbox, region):
                survivors.append(prim_path)
    return survivors


def _candidate_prims(background: Background, default_prim, object_prim_paths: list[str] | None):
    """Yield (isaaclab prim path, prim) pairs to test against the placement region."""
    if object_prim_paths is not None:
        stage = default_prim.GetStage()
        for prim_path in object_prim_paths:
            # Reuse the reference's mapping so nested fixtures resolve identically.
            usd_path = ObjectReference.isaaclab_prim_path_to_original_prim_path(prim_path, background, stage)
            prim = stage.GetPrimAtPath(usd_path)
            assert prim, f"prim not found at '{usd_path}' for path '{prim_path}'"
            yield prim_path, prim
        return
    for child in default_prim.GetChildren():
        # Non-Xformable prims (materials, scopes) carry no geometry to collide with.
        if not child.IsA(UsdGeom.Xformable):
            continue
        yield f"{{ENV_REGEX_NS}}/{background.name}/{child.GetName()}", child


def _background_world_transform(
    background: Background,
) -> tuple[tuple[float, float, float], int, tuple[float, float, float]]:
    """Return the background's world (position, 90°-Z quarters, scale) used to place fixture bboxes."""
    scale = background.scale
    pose = background.initial_pose
    if pose is None:
        return (0.0, 0.0, 0.0), 0, scale
    return pose.position_xyz, quaternion_to_90_deg_z_quarters(pose.rotation_xyzw), scale


def _leaf_name(prim_path: str) -> str:
    """Return the last path segment of a prim path."""
    return prim_path.rstrip("/").rsplit("/", 1)[-1]


def _encloses(outer: AxisAlignedBoundingBox, inner: AxisAlignedBoundingBox) -> bool:
    """True if outer fully contains inner, so inner has no point outside outer."""
    return bool((outer.min_point[0] <= inner.min_point[0]).all()) and bool(
        (outer.max_point[0] >= inner.max_point[0]).all()
    )

# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Relation parents that are parts of placed assets, such as a link of an articulated object.

A part, for example an ``ObjectReference`` to a drawer or turntable link, has no relations of its own. When its
placement root takes part in placement, the part moves with that root: its bounds are fixed relative to the
root's origin, and the root's solved position places them.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING

from isaaclab_arena.relations.relations import On, RotateAroundSolution, get_relation
from isaaclab_arena.utils.bounding_box import quaternion_to_90_deg_z_quarters

if TYPE_CHECKING:
    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox


def get_placed_owner(asset: PlaceableAsset, placed_assets: Collection[PlaceableAsset]) -> PlaceableAsset | None:
    """Return the placed asset whose position places asset.

    Args:
        asset: A relation parent.
        placed_assets: Assets that take part in placement, such as the keys of a candidate's positions.

    Returns:
        The asset itself when it takes part in placement, else its placement root when that takes part, else None.
    """
    if asset in placed_assets:
        return asset
    root = asset.get_placement_root()
    if root is not asset and root in placed_assets:
        return root
    return None


def get_placed_supports(parent: PlaceableAsset, placed_assets: Collection[PlaceableAsset]) -> list[PlaceableAsset]:
    """Return the placed assets an On child of parent rests on: the parent and the placed root it is part of."""
    supports = []
    if parent in placed_assets:
        supports.append(parent)
    root = parent.get_placement_root()
    if root is not parent and root in placed_assets:
        supports.append(root)
    return supports


def get_owners_of_part_parents(placed_assets: Collection[PlaceableAsset]) -> list[PlaceableAsset]:
    """Return the placed assets that own a part used as an On parent, such as an articulation link reference."""
    owners = []
    for asset in placed_assets:
        for relation in asset.get_relations():
            if not isinstance(relation, On):
                continue
            owner = get_placed_owner(relation.parent, placed_assets)
            if owner is not None and owner is not relation.parent and owner not in owners:
                owners.append(owner)
    return owners


def get_fixed_heading_quarters(asset: PlaceableAsset) -> int:
    """Return an asset's fixed heading as a number of 90-degree turns about world Z.

    An anchor uses its pose rotation. A movable asset uses its RotateAroundSolution rotation, or none.
    """
    if asset.is_anchor:
        q_E_A = asset.get_bounding_box_rotation()
    else:
        marker = get_relation(asset, RotateAroundSolution)
        q_E_A = marker.get_rotation_xyzw() if marker is not None else (0.0, 0.0, 0.0, 1.0)
    try:
        return quaternion_to_90_deg_z_quarters(q_E_A)
    except ValueError as error:
        raise ValueError(f"'{asset.name}' carries placed parts, so it needs a quarter-turn heading: {error}") from error


def get_part_bounding_box_relative_to_owner(part: PlaceableAsset, owner: PlaceableAsset) -> AxisAlignedBoundingBox:
    """Return a part's bounds relative to its owner's origin, along environment axes.

    Args:
        part: Asset whose placement root is owner.
        owner: The placed root, with a fixed quarter-turn heading.
    """
    assert part.get_placement_root() is owner, f"'{part.name}' is not part of '{owner.name}'"
    bounds_R = part.get_bounding_box_relative_to_placement_root()
    return bounds_R.rotated_90_around_z(get_fixed_heading_quarters(owner))

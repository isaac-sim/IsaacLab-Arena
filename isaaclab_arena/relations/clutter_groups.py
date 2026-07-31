# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Resolution of clutter members into the piles they belong to."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.relations.relations import ClutteredOn

if TYPE_CHECKING:
    from isaaclab_arena.relations.placement_asset import PlaceableAsset


@dataclass(frozen=True)
class ClutterGroup:
    """One pile: the members that settle together on a shared support."""

    support: PlaceableAsset
    """The asset the pile comes to rest on."""

    name: str
    """Group name shared by every member."""

    members: tuple[PlaceableAsset, ...]
    """Members in declaration order, which is also the order they are dropped in."""

    relation: ClutteredOn
    """The relation of the first member, carrying the pile's shared parameters."""


def is_clutter_member(asset: PlaceableAsset) -> bool:
    """Whether an asset is placed by a clutter pour rather than by the solver."""
    return any(isinstance(relation, ClutteredOn) for relation in asset.get_relations())


def get_clutter_groups(assets: list[PlaceableAsset]) -> list[ClutterGroup]:
    """Group clutter members by the support and group name they declare.

    Members keep declaration order within a group, and groups keep the order of their
    first member, so a fixed asset list always yields the same pours in the same order.

    Args:
        assets: The assets participating in placement.
    """
    order: list[tuple[int, str]] = []
    members: dict[tuple[int, str], list[PlaceableAsset]] = {}
    relations: dict[tuple[int, str], ClutteredOn] = {}
    supports: dict[tuple[int, str], PlaceableAsset] = {}

    for asset in assets:
        for relation in asset.get_relations():
            if not isinstance(relation, ClutteredOn):
                continue
            # Identity, not equality: two distinct supports may compare equal but are separate piles.
            key = (id(relation.parent), relation.group)
            if key not in members:
                order.append(key)
                members[key] = []
                relations[key] = relation
                supports[key] = relation.parent
            members[key].append(asset)

    return [
        ClutterGroup(support=supports[key], name=key[1], members=tuple(members[key]), relation=relations[key])
        for key in order
    ]


def assert_group_parameters_agree(group: ClutterGroup) -> None:
    """Reject a group whose members disagree about the pile's shared parameters.

    Every member carries its own relation, but spread, gaps and clearance describe the
    pile as a whole, so conflicting values would silently resolve to whichever member
    happened to be first.
    """
    shared = group.relation
    for member in group.members:
        for relation in member.get_relations():
            if not isinstance(relation, ClutteredOn) or relation is shared:
                continue
            for field_name in ("spread", "gap_m", "clearance_m", "random_yaw"):
                declared = getattr(relation, field_name)
                expected = getattr(shared, field_name)
                assert declared == expected, (
                    f"Clutter group '{group.name}' on '{group.support.name}' has conflicting "
                    f"{field_name}: '{member.name}' declares {declared}, expected {expected}. "
                    "Group parameters describe the whole pile and must match across members."
                )

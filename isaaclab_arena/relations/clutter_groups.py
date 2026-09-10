# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Clutter group definitions and relation compatibility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.relations.relations import ClutteredOn, FaceTo, Relation, RequiresReachability, RotateAroundSolution

_CLUTTER_MEMBER_RELATIONS = (ClutteredOn, RequiresReachability, RotateAroundSolution)
"""Placement and validation markers supported on clutter members."""

if TYPE_CHECKING:
    from isaaclab_arena.relations.placement_asset import PlaceableAsset


@dataclass(frozen=True)
class ClutterGroup:
    """Objects sharing a support and group name. N is the number of members."""

    support: PlaceableAsset
    """The asset the pile comes to rest on."""

    name: str
    """Group name shared by every member."""

    members: tuple[PlaceableAsset, ...]
    """Members in asset order, shape (N,)."""

    relation: ClutteredOn
    """Shared pile parameters."""


def is_clutter_member(asset: PlaceableAsset) -> bool:
    """Whether an asset declares ClutteredOn."""
    return any(isinstance(relation, ClutteredOn) for relation in asset.get_relations())


def get_clutter_groups(assets: list[PlaceableAsset]) -> list[ClutterGroup]:
    """Return groups in first-occurrence order, preserving member order.

    Args:
        assets: Placement assets.
    """
    groups: dict[tuple[PlaceableAsset, str], tuple[ClutteredOn, list[PlaceableAsset]]] = {}
    for asset in assets:
        for relation in asset.get_relations():
            if isinstance(relation, ClutteredOn):
                _, members = groups.setdefault((relation.parent, relation.group), (relation, []))
                members.append(asset)
    return [
        ClutterGroup(relation.parent, relation.group, tuple(members), relation) for relation, members in groups.values()
    ]


def assert_group_parameters_agree(group: ClutterGroup) -> None:
    """Check that group members share spread and release order."""
    shared = group.relation
    for member in group.members:
        for relation in member.get_relations():
            if not isinstance(relation, ClutteredOn) or relation is shared:
                continue
            assert relation.spread == shared.spread, (
                f"Clutter group '{group.name}' on '{group.support.name}' has conflicting spread: "
                f"'{member.name}' declares {relation.spread}, expected {shared.spread}."
            )
            assert relation.drop_order == shared.drop_order, (
                f"Clutter group '{group.name}' on '{group.support.name}' has conflicting drop_order: "
                f"'{member.name}' declares {relation.drop_order.value}, expected {shared.drop_order.value}."
            )


def assert_relations_do_not_target_clutter(objects: list[PlaceableAsset]) -> None:
    """Check member relations and reject constraints targeting clutter members.

    Args:
        objects: Placement assets.
    """
    members = {asset for asset in objects if is_clutter_member(asset)}
    for asset in objects:
        for relation in asset.get_relations():
            if not isinstance(relation, (Relation, FaceTo, ClutteredOn)):
                continue
            parent = relation.parent
            assert parent not in members, (
                f"'{asset.name}' declares {type(relation).__name__} against '{parent.name}', which is "
                "a clutter member. Relate to the support instead."
            )
        if asset not in members:
            continue
        for relation in asset.get_relations():
            assert isinstance(relation, _CLUTTER_MEMBER_RELATIONS), (
                f"Clutter member '{asset.name}' also declares {type(relation).__name__}, which the "
                "pour cannot honour. Use geometric placement or remove the relation."
            )
        markers = [r for r in asset.get_relations() if isinstance(r, RotateAroundSolution)]
        assert (
            len(markers) <= 1
        ), f"Clutter member '{asset.name}' declares {len(markers)} RotateAroundSolution markers. Declare at most one."

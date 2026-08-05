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
    """Reject a group whose members disagree about the region they are poured into.

    A pile has one region, so ``spread`` has to be single-valued. Clearance, gaps and yaw are
    read from each member's own relation and may differ freely.
    """
    shared = group.relation
    for member in group.members:
        for relation in member.get_relations():
            if not isinstance(relation, ClutteredOn) or relation is shared:
                continue
            assert relation.spread == shared.spread, (
                f"Clutter group '{group.name}' on '{group.support.name}' has conflicting spread: "
                f"'{member.name}' declares {relation.spread}, expected {shared.spread}. A pile is "
                "poured into one region, so its members must agree on how much of the support it uses."
            )


def support_is_provably_immovable(support: PlaceableAsset) -> bool:
    """Return whether physics can be shown never to move this asset.

    True only for a support with no physics body at all, or a rigid body explicitly configured
    kinematic. A rigid body that is merely marked ``IsAnchor`` does not qualify: the marker fixes
    the asset for the relation solver, which is arithmetic, and says nothing about the simulation.
    """
    from isaaclab.assets import ArticulationCfg, RigidObjectCfg

    get_object_cfg = getattr(support, "get_object_cfg", None)
    if get_object_cfg is None:
        return False
    _, cfg = get_object_cfg()
    if isinstance(cfg, (RigidObjectCfg, ArticulationCfg)):
        rigid_props = getattr(getattr(cfg, "spawn", None), "rigid_props", None)
        return bool(getattr(rigid_props, "kinematic_enabled", False))
    # Anything else spawns without a dynamics body, so physics has nothing to push.
    return True


def assert_support_can_hold_a_pile(group: ClutterGroup) -> None:
    """Reject a pile whose support physics could move.

    Members are recorded relative to where their support stood while they settled, and a reset
    restores the support's authored pose, so a support the pile can shove leaves every later
    episode replaying resting poses that no longer match the surface under them. Settling against
    a moving support needs the support's own state captured and replayed, which anchors do not do.
    """
    assert support_is_provably_immovable(group.support), (
        f"Clutter group '{group.name}' rests on '{group.support.name}', which physics can move. "
        "A pile is captured relative to where its support stood while settling, so a support that "
        "shifts replays the pile against a surface that has moved out from under it. Use a static "
        "support, or set rigid_props.kinematic_enabled=True on this one."
    )

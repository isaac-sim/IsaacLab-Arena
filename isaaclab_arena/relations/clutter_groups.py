# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Resolution of clutter members into the piles they belong to."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.relations.relations import ClutteredOn, RotateAroundSolution

_RELATIONS_A_MEMBER_MAY_CARRY = (ClutteredOn, RotateAroundSolution)
"""Relations a poured member may declare.

``RotateAroundSolution`` survives because the pour reads it directly when composing a member's
drop rotation, rather than relying on the solver to apply it.
"""

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
            assert relation.drop_order == shared.drop_order, (
                f"Clutter group '{group.name}' on '{group.support.name}' has conflicting drop_order: "
                f"'{member.name}' declares {relation.drop_order.value}, expected {shared.drop_order.value}. "
                "A pile is released in one order, so its members must agree on which."
            )


def assert_relations_do_not_target_clutter(objects: list[PlaceableAsset]) -> None:
    """Reject relations the pour would leave unhonoured.

    A pour positions its members after the solver has run, so members are held out of the solve
    entirely. Nothing therefore constrains them, and nothing can be constrained against them:
    naming a member as a parent fails deep in the solver with a bare KeyError about a missing
    index, and a relation carried by a member is dropped without a word.

    Args:
        objects: The assets participating in placement.
    """
    members = {id(asset) for asset in objects if is_clutter_member(asset)}
    for asset in objects:
        for relation in asset.get_relations():
            parent = getattr(relation, "parent", None)
            assert parent is None or id(parent) not in members, (
                f"'{asset.name}' declares {type(relation).__name__} against '{parent.name}', which is "
                "a clutter member. Members are positioned by the pour after solving, so the solver "
                "holds no pose to constrain against. Relate to the support instead."
            )
        if id(asset) not in members:
            continue
        for relation in asset.get_relations():
            assert isinstance(relation, _RELATIONS_A_MEMBER_MAY_CARRY), (
                f"Clutter member '{asset.name}' also declares {type(relation).__name__}, which the "
                "pour cannot honour: members are held out of the solve, so the relation would be "
                "silently ignored. Pour the member, or place it with relations instead of pouring it."
            )
        # The rotation marker is read by picking the first one found, so a second is discarded
        # without complaint. One marker is one rotation.
        markers = [r for r in asset.get_relations() if isinstance(r, RotateAroundSolution)]
        assert len(markers) <= 1, (
            f"Clutter member '{asset.name}' declares {len(markers)} RotateAroundSolution markers. "
            "Only the first would be applied and the rest dropped in silence, so declare one."
        )


def support_is_provably_immovable(support: PlaceableAsset) -> bool:
    """Return whether physics can be shown never to move this asset.

    True only for a support with no physics body at all, or a rigid body explicitly configured
    kinematic. A rigid body that is merely marked ``IsAnchor`` does not qualify: the marker fixes
    the asset for the relation solver, which is arithmetic, and says nothing about the simulation.
    """
    from isaaclab.assets import ArticulationCfg

    get_object_cfg = getattr(support, "get_object_cfg", None)
    if get_object_cfg is None:
        return False
    _, cfg = get_object_cfg()
    # Pinning an articulation's root says nothing about its links and joints, which carry the
    # pile and can move under it.
    if isinstance(cfg, ArticulationCfg):
        return False
    # Judge the spawner, not the config class. A base config is not evidence of no dynamics:
    # Arena's backgrounds are ObjectType.BASE yet spawn rigid bodies through spawn_cfg_addon,
    # so the kinematic flag is the only claim here that physics actually honours.
    rigid_props = getattr(getattr(cfg, "spawn", None), "rigid_props", None)
    return bool(getattr(rigid_props, "kinematic_enabled", False))


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

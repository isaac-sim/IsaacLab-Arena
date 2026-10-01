# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Discover passive scene assets that should participate in placement collision."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import TYPE_CHECKING

from isaaclab_arena.assets.background import Background
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_reference import ObjectReference
from isaaclab_arena.relations.background_collision_object import make_fixed_collision_objects
from isaaclab_arena.relations.collision_mode import CollisionMode, get_object_collision_mode
from isaaclab_arena.relations.collision_object import CollisionObject
from isaaclab_arena.utils.pose import Pose

if TYPE_CHECKING:
    from isaaclab_arena.assets.asset import Asset
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.relations.placement_asset import PlaceableAsset


def get_placement_collision_objects(
    placement_assets: list[PlaceableAsset],
    scene_assets: Iterable[Asset | RigidObjectSet],
    default_collision_mode: CollisionMode,
) -> list[CollisionObject]:
    """Discover obstacles using the scene's collision modes and anchored support exclusions.

    Include room meshes when the solver, a placement asset, or a background requests
    MESH. Exclude anchored references from their parent mesh because placement already
    checks those supports separately.
    """
    scene_assets = list(scene_assets)
    include_background = (
        default_collision_mode == CollisionMode.MESH
        or any(
            get_object_collision_mode(asset, default_collision_mode) == CollisionMode.MESH for asset in placement_assets
        )
        or any(
            isinstance(asset, Background)
            and get_object_collision_mode(asset, default_collision_mode) == CollisionMode.MESH
            for asset in scene_assets
        )
    )
    passive_assets = discover_passive_assets(scene_assets, include_background=include_background)
    passive_asset_set = set(passive_assets)
    # A parent's collision geometry already contains its passive references.
    collision_objects = [
        asset
        for asset in passive_assets
        if not isinstance(asset, ObjectReference) or asset.parent_asset not in passive_asset_set
    ]
    if not include_background:
        return collision_objects

    excluded_prim_paths_by_object: defaultdict[CollisionObject, set[str]] = defaultdict(set)
    for asset in placement_assets:
        if isinstance(asset, ObjectReference) and asset.is_anchor and asset.parent_asset in passive_asset_set:
            excluded_prim_paths_by_object[asset.parent_asset].add(asset.prim_path_in_parent_usd)
    return make_fixed_collision_objects(collision_objects, excluded_prim_paths_by_object=excluded_prim_paths_by_object)


def discover_passive_assets(
    assets: Iterable[Asset | RigidObjectSet],
    include_background: bool = False,
) -> list[Object | ObjectReference]:
    """Return original relation-free scene assets with fixed placement geometry.

    Objects need a fixed Pose. Included backgrounds may omit their pose to use identity.
    Discovery does not aggregate geometry or remove references covered by a parent.

    Args:
        assets: Scene assets to scan for relation-free fixed objects.
        include_background: Include Background assets, treating an unset pose as identity.
            Returned assets retain their identities, including references to returned parents.
    """
    passive_assets: list[Object | ObjectReference] = []
    for asset in assets:
        if not isinstance(asset, (Object, ObjectReference)):
            continue
        if isinstance(asset, Background) and not include_background:
            continue
        if asset.get_relations():
            continue
        # Without a USD path no bounding box can be computed for collision.
        if isinstance(asset, Object) and asset.usd_path is None:
            print(f"Skipping '{asset.name}' as a collision obstacle: missing USD path.")
            continue
        if isinstance(asset, ObjectReference) and asset.parent_asset.usd_path is None:
            print(
                f"Skipping object reference '{asset.name}' as a collision obstacle: "
                f"parent asset '{asset.parent_asset.name}' is missing a USD path."
            )
            continue
        initial_pose = asset.get_initial_pose()
        if isinstance(asset, Background) and include_background:
            assert initial_pose is None or isinstance(initial_pose, Pose), (
                f"Whole-scene Background asset '{asset.name}' must have a fixed Pose or no initial_pose "
                f"for passive collision geometry, got {type(initial_pose).__name__}."
            )
            passive_assets.append(asset)
            continue
        if not isinstance(initial_pose, Pose):
            pose_kind = "None" if initial_pose is None else type(initial_pose).__name__
            print(f"Skipping '{asset.name}' as a collision obstacle: needs a fixed pose but has {pose_kind}.")
            continue
        passive_assets.append(asset)

    return passive_assets

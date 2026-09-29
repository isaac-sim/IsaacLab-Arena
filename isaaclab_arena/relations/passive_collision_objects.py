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
    exclusions = [asset for asset in placement_assets if asset.is_anchor and isinstance(asset, ObjectReference)]
    return get_passive_collision_objects(
        scene_assets, include_background=include_background, background_mesh_exclusions=exclusions
    )


def get_passive_collision_objects(
    assets: Iterable[Asset | RigidObjectSet],
    include_background: bool = False,
    background_mesh_exclusions: Iterable[ObjectReference] = (),
) -> list[CollisionObject]:
    """Return relation-free scene assets that qualify as passive collision obstacles.

    PoseRange, PosePerEnv, and unset poses are skipped because passive collision obstacles
    must have a fixed world transform during placement.

    Args:
        assets: Scene assets to scan for relation-free fixed objects.
        include_background: If True, include Background assets and aggregate all
            mesh-capable objects into a single FixedCollisionObject.
        background_mesh_exclusions: Object references whose USD subtrees are omitted
            from an aggregated parent Background mesh.
    """
    collision_objects: list[CollisionObject] = []
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
                f"for aggregate mesh collision, got {type(initial_pose).__name__}."
            )
            collision_objects.append(asset)
            continue
        if not isinstance(initial_pose, Pose):
            pose_kind = "None" if initial_pose is None else type(initial_pose).__name__
            print(f"Skipping '{asset.name}' as a collision obstacle: needs a fixed pose but has {pose_kind}.")
            continue
        collision_objects.append(asset)

    collision_object_set = set(collision_objects)
    # If a parent object is already an obstacle, its references are covered by the parent mesh/bbox.
    collision_objects = [
        asset
        for asset in collision_objects
        if not isinstance(asset, ObjectReference) or asset.parent_asset not in collision_object_set
    ]

    if include_background:
        excluded_prim_paths_by_object: defaultdict[CollisionObject, set[str]] = defaultdict(set)
        for reference in background_mesh_exclusions:
            if reference.parent_asset in collision_object_set:
                excluded_prim_paths_by_object[reference.parent_asset].add(reference.prim_path_in_parent_usd)
        return make_fixed_collision_objects(
            collision_objects,
            excluded_prim_paths_by_object=excluded_prim_paths_by_object,
        )
    return collision_objects

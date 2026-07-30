# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Discover passive scene assets that should participate in placement collision."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from isaaclab_arena.assets.background import Background
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_reference import ObjectReference
from isaaclab_arena.relations.background_collision_object import make_fixed_collision_objects
from isaaclab_arena.relations.collision_object import CollisionObject
from isaaclab_arena.utils.pose import Pose

if TYPE_CHECKING:
    from isaaclab_arena.assets.asset import Asset
    from isaaclab_arena.assets.object_set import RigidObjectSet


def get_passive_collision_objects(
    assets: Iterable[Asset | RigidObjectSet], include_background: bool = False
) -> list[CollisionObject]:
    """Return relation-free scene assets that qualify as passive collision obstacles.

    PoseRange, PosePerEnv, and unset poses are skipped because passive collision obstacles
    must have a fixed world transform during placement.

    Args:
        assets: Scene assets to scan for relation-free fixed objects.
        include_background: If True, include Background assets and aggregate all
            mesh-capable objects into a single FixedCollisionObject.
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
        return make_fixed_collision_objects(collision_objects)
    return list(collision_objects)

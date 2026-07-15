# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

from isaaclab_arena.relations.collision_mode import CollisionMode, get_object_collision_mode
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_events import (
    get_pose_from_layout,
    place_entities_from_layouts,
    solve_and_place_objects,
)
from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
from isaaclab_arena.relations.relations import get_anchor_objects
from isaaclab_arena.utils.pose import PosePerEnv

if TYPE_CHECKING:
    from isaaclab.managers import EventTermCfg

    from isaaclab_arena.assets.asset import Asset
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.relations.collision_object import CollisionObject
    from isaaclab_arena.relations.placement_entity import PlacementEntity
    from isaaclab_arena.relations.placement_result import PlacementResult


def _get_passive_collision_objects(
    assets: Iterable[Asset | RigidObjectSet],
    include_background: bool = False,
) -> list[CollisionObject]:
    """Load passive collision discovery only when relation placement needs it."""
    from isaaclab_arena.relations.passive_collision_objects import get_passive_collision_objects

    return get_passive_collision_objects(assets, include_background=include_background)


def solve_and_apply_relation_placement(
    objects: list[PlacementEntity],
    num_envs: int,
    placer_params: ObjectPlacerParams | None = None,
    collision_objects: list[CollisionObject] | None = None,
    scene_assets: Iterable[Asset | RigidObjectSet] | None = None,
    scene_entity_names: Mapping[str, str] | None = None,
) -> EventTermCfg | None:
    """Solve relation placement and apply the result to object reset/static state.

    Args:
        objects: Entities with spatial predicates that should be relation-solved.
        num_envs: Number of environments to prepare placements for.
        placer_params: Optional placement parameters. A shallow copy is used so
            this function can force pooled placement without mutating the caller's instance.
        collision_objects: Fixed obstacles avoided during placement but never optimized
            or relation-constrained.
        scene_assets: Optional scene assets to scan for passive collision objects
            when collision_objects is not supplied.
        scene_entity_names: Isaac Lab scene name for each placement entity.

    Returns:
        Reset event config to attach to the environment when placement should be
        resolved on reset. Returns ``None`` when no reset event is needed.
    """
    objects = list(objects)
    if not objects:
        print("No objects with relations found in scene. Skipping relation solving.")
        return None
    entity_names = {obj.name for obj in objects}
    assert len(entity_names) == len(objects), "Placement entity names must be unique"
    if scene_entity_names is None:
        scene_entity_names = {obj.name: obj.name for obj in objects}
    assert set(scene_entity_names) == entity_names, "scene_entity_names must contain every placement entity name"

    if placer_params is None:
        placer_params = ObjectPlacerParams()
    else:
        placer_params = copy.copy(placer_params)
    placer_params.apply_positions_to_objects = False
    if collision_objects is None and scene_assets is not None:
        scene_assets = list(scene_assets)
        collision_objects = _get_passive_collision_objects(
            scene_assets,
            include_background=_should_include_background_mesh(
                objects, scene_assets, placer_params.solver_params.collision_mode
            ),
        )
    placement_pool = PooledObjectPlacer(
        objects=objects,
        placer_params=placer_params,
        pool_size=num_envs * placer_params.min_unique_layouts_per_env,
        num_envs=num_envs,
        collision_objects=collision_objects,
    )

    if placement_pool.had_fallbacks:
        print(
            "Warning: Relation placement pool accepted best-loss fallback layouts "
            "that failed strict placement validation."
        )

    return _apply_relation_placement_result(
        objects=objects,
        placer_params=placer_params,
        placement_pool=placement_pool,
        num_envs=num_envs,
        scene_entity_names=scene_entity_names,
    )


def _should_include_background_mesh(
    objects: list[PlacementEntity],
    scene_assets: Iterable[Asset | RigidObjectSet],
    default_collision_mode: CollisionMode,
) -> bool:
    """Return True when the default mode or any object/Background override resolves to MESH."""
    from isaaclab_arena.assets.background import Background

    if default_collision_mode == CollisionMode.MESH:
        return True
    if any(get_object_collision_mode(obj, default_collision_mode) == CollisionMode.MESH for obj in objects):
        return True
    return any(
        isinstance(asset, Background) and get_object_collision_mode(asset, default_collision_mode) == CollisionMode.MESH
        for asset in scene_assets
    )


def _apply_relation_placement_result(
    objects: list[PlacementEntity],
    placer_params: ObjectPlacerParams,
    placement_pool: PooledObjectPlacer,
    num_envs: int,
    scene_entity_names: Mapping[str, str],
) -> EventTermCfg | None:
    """Apply selected layouts to object spawn state and build reset event config."""
    anchor_objects_set = set(get_anchor_objects(objects))
    # Prevent external pose-reset events from conflicting with relation-solved objects.
    _validate_no_conflicting_pose_reset_events(objects, anchor_objects_set)

    # Anchor objects do not move, so no need to apply reset event.
    if anchor_objects_set == set(objects):
        return None

    if placer_params.resolve_on_reset:
        return _apply_dynamic_spawn_pose(
            objects=objects,
            placement_pool=placement_pool,
            anchor_objects_set=anchor_objects_set,
            scene_entity_names=scene_entity_names,
        )

    if all(obj.supports_per_env_initial_pose() for obj in objects):
        _apply_static_initial_poses(
            objects=objects,
            placement_pool=placement_pool,
            anchor_objects_set=anchor_objects_set,
            num_envs=num_envs,
        )
        return None
    return _apply_static_spawn_pose(
        objects=objects,
        placement_pool=placement_pool,
        anchor_objects_set=anchor_objects_set,
        num_envs=num_envs,
        scene_entity_names=scene_entity_names,
    )


def _apply_dynamic_spawn_pose(
    objects: list[PlacementEntity],
    placement_pool: PooledObjectPlacer,
    anchor_objects_set: set[PlacementEntity],
    scene_entity_names: Mapping[str, str],
) -> EventTermCfg:
    """Set initial spawn pose from one layout and return the reset placement event."""
    from isaaclab.managers import EventTermCfg

    # For env-indexed pools this seeds from env 0; the first reset overwrites with per-env layouts.
    layout = placement_pool.sample_with_replacement(1)[0]
    _set_placement_initial_poses(objects, anchor_objects_set, layout)

    return EventTermCfg(
        func=solve_and_place_objects,
        mode="reset",
        params={
            "objects": objects,
            "placement_pool": placement_pool,
            "scene_entity_names": scene_entity_names,
        },
    )


def _apply_static_spawn_pose(
    objects: list[PlacementEntity],
    placement_pool: PooledObjectPlacer,
    anchor_objects_set: set[PlacementEntity],
    num_envs: int,
    scene_entity_names: Mapping[str, str],
) -> EventTermCfg:
    """Return a reset event that restores one fixed layout per environment."""
    from isaaclab.managers import EventTermCfg

    layouts = placement_pool.sample_with_replacement(num_envs)
    _set_placement_initial_poses(objects, anchor_objects_set, layouts[0])
    return EventTermCfg(
        func=place_entities_from_layouts,
        mode="reset",
        params={
            "objects": objects,
            "layouts": layouts,
            "scene_entity_names": scene_entity_names,
        },
    )


def _set_placement_initial_poses(
    objects: list[PlacementEntity],
    anchor_objects_set: set[PlacementEntity],
    layout: PlacementResult,
) -> None:
    """Seed non-anchor entities from one placement layout."""
    for obj in objects:
        if obj in anchor_objects_set:
            continue
        pose = get_pose_from_layout(obj, layout)
        obj.set_placement_initial_pose(pose)


def _apply_static_initial_poses(
    objects: list[PlacementEntity],
    placement_pool: PooledObjectPlacer,
    anchor_objects_set: set[PlacementEntity],
    num_envs: int,
) -> None:
    """Apply fixed per-environment poses for ``resolve_on_reset=False``."""
    layouts = placement_pool.sample_with_replacement(num_envs)
    for obj in objects:
        if obj in anchor_objects_set:
            continue
        poses = [get_pose_from_layout(obj, layouts[env_idx]) for env_idx in range(num_envs)]
        obj.set_initial_pose(PosePerEnv(poses=poses))


def _validate_no_conflicting_pose_reset_events(
    objects: list[PlacementEntity],
    anchor_objects_set: set[PlacementEntity],
) -> None:
    """Reject conflicting explicit pose-reset events on relation-solved objects."""
    for obj in objects:
        assert not (obj not in anchor_objects_set and obj.has_pose_reset_event()), (
            f"Non-anchor object '{obj.name}' has an explicit pose-reset event. "
            "Relational solving should not be combined with explicit setting of "
            "poses on non-anchor objects."
        )

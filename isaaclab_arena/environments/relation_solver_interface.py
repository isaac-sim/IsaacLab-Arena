# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import TYPE_CHECKING

from isaaclab_arena.relations.collision_mode import CollisionMode, get_object_collision_mode
from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_events import (
    get_pose_from_layout,
    place_assets_from_layouts,
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
    from isaaclab_arena.relations.placement_asset import PlacementAsset
    from isaaclab_arena.relations.placement_result import PlacementResult


def _get_passive_collision_objects(
    assets: Iterable[Asset | RigidObjectSet],
    include_background: bool = False,
) -> list[CollisionObject]:
    """Load passive collision discovery only when relation placement needs it."""
    from isaaclab_arena.relations.passive_collision_objects import get_passive_collision_objects

    return get_passive_collision_objects(assets, include_background=include_background)


def solve_and_apply_relation_placement(
    assets: list[PlacementAsset],
    num_envs: int,
    placer_params: ObjectPlacerParams | None = None,
    collision_objects: list[CollisionObject] | None = None,
    scene_assets: Iterable[Asset | RigidObjectSet] | None = None,
) -> EventTermCfg | None:
    """Solve relation placement and apply the result to asset reset/static state.

    Args:
        assets: Assets with spatial predicates that should be relation-solved.
        num_envs: Number of environments to prepare placements for.
        placer_params: Optional placement parameters. A shallow copy is used so
            this function can force pooled placement without mutating the caller's instance.
        collision_objects: Fixed obstacles avoided during placement but never optimized
            or relation-constrained.
        scene_assets: Optional scene assets to scan for passive collision objects
            when collision_objects is not supplied.

    Returns:
        Reset event config to attach to the environment when placement should be
        resolved on reset. Returns ``None`` when no reset event is needed.
    """
    assets = list(assets)
    if not assets:
        print("No assets with relations found in scene. Skipping relation solving.")
        return None
    asset_names = {asset.name for asset in assets}
    assert len(asset_names) == len(assets), "Placement asset names must be unique"
    scene_keys = [asset.get_scene_name() for asset in assets]
    assert len(set(scene_keys)) == len(scene_keys), "Placement assets map to duplicate scene keys"

    if placer_params is None:
        placer_params = ObjectPlacerParams()
    else:
        placer_params = copy.copy(placer_params)
    placer_params.apply_positions_to_objects = False
    # Note(xinjieyao, 2026-07-23): The build-time IK-reachability check reads the embodiment only while its validator is built (during the
    # pool construction below). Copy the config so the live embodiment can be dropped afterwards without
    # mutating the caller.
    placer_params.reachability_config = copy.copy(placer_params.reachability_config)
    if collision_objects is None and scene_assets is not None:
        scene_assets = list(scene_assets)
        collision_objects = _get_passive_collision_objects(
            scene_assets,
            include_background=_should_include_background_mesh(
                assets, scene_assets, placer_params.solver_params.collision_mode
            ),
        )
    placement_pool = PooledObjectPlacer(
        objects=assets,
        placer_params=placer_params,
        pool_size=num_envs * placer_params.min_unique_layouts_per_env,
        num_envs=num_envs,
        collision_objects=collision_objects,
    )
    # Validators are built once above and reused for every refill, so the embodiment is done being read; drop
    # it before the reset-event params below capture (and deep-copy/validate) the pool.
    placer_params.reachability_config.embodiment = None

    if placement_pool.had_fallbacks:
        print(
            "Warning: Relation placement pool accepted best-loss fallback layouts "
            "that failed strict placement validation."
        )

    return _apply_relation_placement_result(
        assets=assets,
        placer_params=placer_params,
        placement_pool=placement_pool,
        num_envs=num_envs,
    )


def _should_include_background_mesh(
    assets: list[PlacementAsset],
    scene_assets: Iterable[Asset | RigidObjectSet],
    default_collision_mode: CollisionMode,
) -> bool:
    """Return True when the default mode or any relevant asset override resolves to MESH."""
    from isaaclab_arena.assets.background import Background

    if default_collision_mode == CollisionMode.MESH:
        return True
    if any(get_object_collision_mode(asset, default_collision_mode) == CollisionMode.MESH for asset in assets):
        return True
    return any(
        isinstance(asset, Background) and get_object_collision_mode(asset, default_collision_mode) == CollisionMode.MESH
        for asset in scene_assets
    )


def _apply_relation_placement_result(
    assets: list[PlacementAsset],
    placer_params: ObjectPlacerParams,
    placement_pool: PooledObjectPlacer,
    num_envs: int,
) -> EventTermCfg | None:
    """Apply selected layouts to asset spawn state and build reset event config."""
    anchor_assets = set(get_anchor_objects(assets))
    # Prevent external pose-reset events from conflicting with relation-solved assets.
    _validate_no_conflicting_pose_reset_events(assets, anchor_assets)

    # Anchor assets do not move, so no need to apply reset event.
    if anchor_assets == set(assets):
        return None

    if placer_params.resolve_on_reset:
        return _apply_dynamic_spawn_pose(
            assets=assets,
            placement_pool=placement_pool,
            anchor_assets=anchor_assets,
        )

    # Objects can store PosePerEnv, so their reset events restore fixed per-env
    # poses. Scenes containing any asset without per-env pose support use one
    # coordinated event to restore layouts[env_id].
    if all(asset.supports_per_env_initial_pose() for asset in assets):
        _apply_static_initial_poses(
            assets=assets,
            placement_pool=placement_pool,
            anchor_assets=anchor_assets,
            num_envs=num_envs,
        )
        return None
    return _apply_static_spawn_pose(
        assets=assets,
        placement_pool=placement_pool,
        anchor_assets=anchor_assets,
        num_envs=num_envs,
    )


def _apply_dynamic_spawn_pose(
    assets: list[PlacementAsset],
    placement_pool: PooledObjectPlacer,
    anchor_assets: set[PlacementAsset],
) -> EventTermCfg:
    """Set initial spawn pose from one layout and return the reset placement event."""
    from isaaclab.managers import EventTermCfg

    # For env-indexed pools this seeds from env 0; the first reset overwrites with per-env layouts.
    layout = placement_pool.sample_with_replacement(1)[0]
    _set_placement_initial_poses(assets, anchor_assets, layout)

    return EventTermCfg(
        func=solve_and_place_objects,
        mode="reset",
        params={
            "assets": assets,
            "placement_pool": placement_pool,
        },
    )


def _apply_static_spawn_pose(
    assets: list[PlacementAsset],
    placement_pool: PooledObjectPlacer,
    anchor_assets: set[PlacementAsset],
    num_envs: int,
) -> EventTermCfg:
    """Return a coordinated reset event that restores one fixed layout per environment."""
    from isaaclab.managers import EventTermCfg

    layouts = placement_pool.sample_with_replacement(num_envs)
    _set_placement_initial_poses(assets, anchor_assets, layouts[0])
    return EventTermCfg(
        func=place_assets_from_layouts,
        mode="reset",
        params={
            "assets": assets,
            "layouts": layouts,
        },
    )


def _set_placement_initial_poses(
    assets: list[PlacementAsset],
    anchor_assets: set[PlacementAsset],
    layout: PlacementResult,
) -> None:
    """Seed the spawn pose while preserving coordinated reset ownership."""
    for asset in assets:
        if asset in anchor_assets:
            continue
        pose = get_pose_from_layout(asset, layout)
        asset.set_placement_initial_pose(pose)


def _apply_static_initial_poses(
    assets: list[PlacementAsset],
    placement_pool: PooledObjectPlacer,
    anchor_assets: set[PlacementAsset],
    num_envs: int,
) -> None:
    """Apply fixed per-environment poses for ``resolve_on_reset=False``."""
    layouts = placement_pool.sample_with_replacement(num_envs)
    for asset in assets:
        if asset in anchor_assets:
            continue
        poses = [get_pose_from_layout(asset, layouts[env_idx]) for env_idx in range(num_envs)]
        asset.set_initial_pose(PosePerEnv(poses=poses))


def _validate_no_conflicting_pose_reset_events(
    assets: list[PlacementAsset],
    anchor_assets: set[PlacementAsset],
) -> None:
    """Reject conflicting explicit pose-reset events on relation-solved assets."""
    for asset in assets:
        assert not (asset not in anchor_assets and asset.has_pose_reset_event()), (
            f"Non-anchor asset '{asset.name}' has an explicit pose-reset event. "
            "Relational solving should not be combined with explicit setting of "
            "poses on non-anchor assets."
        )

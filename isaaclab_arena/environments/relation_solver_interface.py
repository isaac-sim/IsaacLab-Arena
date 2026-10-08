# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_asset import get_scene_root_owners
from isaaclab_arena.relations.placement_events import PlacementPoolHandle, get_pose_from_layout, solve_and_place_objects
from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
from isaaclab_arena.relations.relations import get_anchor_objects
from isaaclab_arena.utils.pose import Pose, PosePerEnv

if TYPE_CHECKING:
    from isaaclab.managers import EventTermCfg

    from isaaclab_arena.assets.asset import Asset
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.relations.collision_object import CollisionObject
    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.relations.placement_result import PlacementResult


def create_relation_placement_variation(
    assets: list[PlaceableAsset],
    num_envs: int,
    placer_params: ObjectPlacerParams | None = None,
    collision_objects: list[CollisionObject] | None = None,
    scene_assets: Iterable[Asset | RigidObjectSet] | None = None,
    asset_identities: dict[str, str] | None = None,
    replay_assets: list[PlaceableAsset] | None = None,
):
    """Build relation placement as one coordinated scene-level variation."""
    from isaaclab_arena.variations.relation_placement_variation import PlacementPoolSampler, RelationPlacementVariation

    prepared = _build_relation_placement_pool(
        assets=assets,
        num_envs=num_envs,
        placer_params=placer_params,
        collision_objects=collision_objects,
        scene_assets=scene_assets,
    )
    if prepared is None:
        return None
    if replay_assets is not None:
        get_scene_root_owners(replay_assets)
    resolved_params, placement_pool = prepared
    anchor_assets = set(get_anchor_objects(assets))
    _validate_no_conflicting_pose_reset_events(assets, anchor_assets)
    if anchor_assets == set(assets):
        return None
    fixed_results = None
    if resolved_params.resolve_on_reset:
        [construction_layout] = placement_pool.sample_with_replacement(1)
        _seed_spawn_config_from_layout(assets, anchor_assets, construction_layout)
    else:
        layouts = placement_pool.sample_with_replacement(num_envs)
        fixed_results = {env_id: layout for env_id, layout in enumerate(layouts)}
        _apply_static_initial_poses(
            assets=assets,
            placement_pool=placement_pool,
            anchor_assets=anchor_assets,
            num_envs=num_envs,
            layouts=layouts,
        )
        for asset in assets:
            if asset in anchor_assets:
                continue
            assert asset.has_pose_reset_event(), (
                f"Static relation placement stored a per-env pose for non-anchor asset '{asset.name}', but it "
                "owns no reset event, so its solved layout would be silently discarded on every reset."
            )
    sampler = PlacementPoolSampler(
        assets=assets,
        placement_pool=placement_pool,
        fixed_results=fixed_results,
        asset_identities=asset_identities,
        replay_assets=replay_assets,
    )
    return RelationPlacementVariation(sampler, write_live_samples=resolved_params.resolve_on_reset)


def create_relation_placement_replay_variation(
    assets: list[PlaceableAsset],
    replay_assets: list[PlaceableAsset],
    samples: list[dict[str, Any]],
    num_envs: int,
    asset_identities: dict[str, str] | None = None,
):
    """Build solver-free relation placement backed by episode-condition samples."""
    from isaaclab_arena.variations.relation_placement_variation import PlacementPoolSampler, RelationPlacementVariation

    sampler = PlacementPoolSampler(
        assets=assets,
        placement_pool=None,
        asset_identities=asset_identities,
        replay_assets=replay_assets,
    )
    variation = RelationPlacementVariation(sampler, write_live_samples=False)
    variation.validate_replay_samples(samples)
    _seed_spawn_config_from_replay(samples, replay_assets, num_envs)
    return variation


def solve_and_apply_relation_placement(
    assets: list[PlaceableAsset],
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
    prepared = _build_relation_placement_pool(
        assets=assets,
        num_envs=num_envs,
        placer_params=placer_params,
        collision_objects=collision_objects,
        scene_assets=scene_assets,
    )
    if prepared is None:
        return None
    placer_params, placement_pool = prepared

    return _apply_relation_placement_result(
        assets=assets,
        placer_params=placer_params,
        placement_pool=placement_pool,
        num_envs=num_envs,
    )


def _build_relation_placement_pool(
    assets: list[PlaceableAsset],
    num_envs: int,
    placer_params: ObjectPlacerParams | None,
    collision_objects: list[CollisionObject] | None,
    scene_assets: Iterable[Asset | RigidObjectSet] | None,
) -> tuple[ObjectPlacerParams, PooledObjectPlacer] | None:
    """Validate relation assets and build their reusable placement pool."""
    if not assets:
        print("No assets with relations found in scene. Skipping relation solving.")
        return None
    asset_names = {asset.name for asset in assets}
    assert len(asset_names) == len(assets), "Placement asset names must be unique"
    scene_keys = [asset.get_scene_key() for asset in assets]
    assert len(set(scene_keys)) == len(scene_keys), "Placement assets map to duplicate scene keys"

    placer_params = ObjectPlacerParams() if placer_params is None else copy.copy(placer_params)
    placer_params.apply_positions_to_objects = False
    placer_params.reachability_config = copy.copy(placer_params.reachability_config)
    if collision_objects is None and scene_assets is not None:
        from isaaclab_arena.relations.passive_collision_objects import get_placement_collision_objects

        collision_objects = get_placement_collision_objects(
            assets, scene_assets, placer_params.solver_params.collision_mode
        )
    placement_pool = PooledObjectPlacer(
        objects=assets,
        placer_params=placer_params,
        pool_size=num_envs * placer_params.min_unique_layouts_per_env,
        num_envs=num_envs,
        collision_objects=collision_objects,
    )
    placer_params.reachability_config.embodiment = None
    if placement_pool.had_fallbacks:
        print(
            "Warning: Relation placement pool accepted best-loss fallback layouts "
            "that failed strict placement validation."
        )
    return placer_params, placement_pool


def _apply_relation_placement_result(
    assets: list[PlaceableAsset],
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

    # Every placement asset (objects and embodiments) stores its solved pose as a PosePerEnv and
    # owns a per-asset reset event, so static layouts need no coordinated place-from-layouts event.
    _apply_static_initial_poses(
        assets=assets,
        placement_pool=placement_pool,
        anchor_assets=anchor_assets,
        num_envs=num_envs,
    )
    for asset in assets:
        if asset in anchor_assets:
            continue
        assert asset.has_pose_reset_event(), (
            f"Static relation placement stored a per-env pose for non-anchor asset '{asset.name}', but it "
            "owns no reset event, so its solved layout would be silently discarded on every reset."
        )
    return None


def _apply_dynamic_spawn_pose(
    assets: list[PlaceableAsset],
    placement_pool: PooledObjectPlacer,
    anchor_assets: set[PlaceableAsset],
) -> EventTermCfg:
    """Set initial spawn pose from one layout and return the reset placement event."""
    from isaaclab.managers import EventTermCfg

    # Scene assets need a valid construction pose before reset events can run.
    # This non-consuming env-0 sample is bootstrap-only; reset draws independently per env.
    [construction_layout] = placement_pool.sample_with_replacement(1)
    _seed_spawn_config_from_layout(assets, anchor_assets, construction_layout)

    return EventTermCfg(
        func=solve_and_place_objects,
        mode="reset",
        params={
            "placement_pool": PlacementPoolHandle(placement_pool),
        },
    )


def _seed_spawn_config_from_layout(
    assets: list[PlaceableAsset],
    anchor_assets: set[PlaceableAsset],
    layout: PlacementResult,
) -> None:
    """Write one solved layout into the single-pose scene configuration."""
    for asset in assets:
        if asset in anchor_assets:
            continue
        pose = get_pose_from_layout(asset, layout)
        asset.set_initial_pose(pose, create_reset_event=False)


def _seed_spawn_config_from_replay(
    samples: list[dict[str, Any]],
    replay_assets: list[PlaceableAsset],
    num_envs: int,
) -> None:
    """Seed construction roots from replay rows without replacing reset events."""
    owners = get_scene_root_owners(replay_assets)
    poses_by_asset: dict[PlaceableAsset, dict[str, PosePerEnv]] = {}
    for scene_key in samples[0]["poses"]:
        per_env_poses = []
        for env_id in range(num_envs):
            pose = Pose.from_dict(samples[env_id % len(samples)]["poses"][scene_key])
            assert pose is not None
            per_env_poses.append(pose)
        poses_by_asset.setdefault(owners[scene_key], {})[scene_key] = PosePerEnv(per_env_poses)
    for asset, poses in poses_by_asset.items():
        asset.set_initial_scene_root_poses(poses)


def _apply_static_initial_poses(
    assets: list[PlaceableAsset],
    placement_pool: PooledObjectPlacer,
    anchor_assets: set[PlaceableAsset],
    num_envs: int,
    layouts: list[PlacementResult] | None = None,
) -> None:
    """Apply fixed per-environment poses for ``resolve_on_reset=False``."""
    layouts = placement_pool.sample_with_replacement(num_envs) if layouts is None else layouts
    for asset in assets:
        if asset in anchor_assets:
            continue
        poses = [get_pose_from_layout(asset, layouts[env_idx]) for env_idx in range(num_envs)]
        asset.set_initial_pose(PosePerEnv(poses=poses))


def _validate_no_conflicting_pose_reset_events(
    assets: list[PlaceableAsset],
    anchor_assets: set[PlaceableAsset],
) -> None:
    """Reject conflicting explicit pose-reset events on relation-solved assets."""
    for asset in assets:
        assert not (asset not in anchor_assets and asset.has_pose_reset_event()), (
            f"Non-anchor asset '{asset.name}' has an explicit pose-reset event. "
            "Relational solving should not be combined with explicit setting of "
            "poses on non-anchor assets."
        )

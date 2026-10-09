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
from isaaclab_arena.relations.placement_events import get_scene_root_poses_from_layout
from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
from isaaclab_arena.relations.relations import get_anchor_objects
from isaaclab_arena.utils.pose import Pose, PosePerEnv

if TYPE_CHECKING:
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
    replay_assets: list[PlaceableAsset] | None = None,
    *,
    live_placement_enabled: bool = True,
    replay_may_be_configured: bool = False,
):
    """Declare relation placement without solving, sampling, or mutating spawn poses."""
    from isaaclab_arena.variations.relation_placement_variation import PlacementPoolSampler, RelationPlacementVariation

    anchor_assets = set(get_anchor_objects(assets))
    can_prepare_live = live_placement_enabled and bool(assets) and anchor_assets != set(assets)
    if not can_prepare_live and not replay_may_be_configured:
        return None

    replay_assets = assets if replay_assets is None else replay_assets
    sampler = PlacementPoolSampler(
        assets=assets,
        placement_pool=None,
        replay_assets=replay_assets,
    )

    def prepare(variation: RelationPlacementVariation) -> None:
        replay_samples = variation.recorded_replay_samples
        if replay_samples is not None:
            _seed_spawn_config_from_replay(replay_samples, replay_assets, num_envs)
            return

        assert can_prepare_live, "Relation placement has neither recorded samples nor live solving enabled"
        prepared = _build_relation_placement_pool(
            assets=assets,
            num_envs=num_envs,
            placer_params=placer_params,
            collision_objects=collision_objects,
            scene_assets=scene_assets,
        )
        assert prepared is not None, "Live relation placement preparation produced no placement pool"
        resolved_params, placement_pool = prepared
        _validate_no_conflicting_pose_reset_events(assets, anchor_assets)
        fixed_results = None
        if resolved_params.resolve_on_reset:
            [construction_layout] = placement_pool.sample_with_replacement(1)
            _seed_spawn_config_from_layouts(
                assets,
                anchor_assets,
                [construction_layout] * num_envs,
            )
        else:
            layouts = placement_pool.sample_with_replacement(num_envs)
            fixed_results = {env_id: layout for env_id, layout in enumerate(layouts)}
            _seed_spawn_config_from_layouts(assets, anchor_assets, layouts)
        variation.configure_prepared_live_state(placement_pool, fixed_results)

    return RelationPlacementVariation(
        sampler,
        live_placement_enabled=can_prepare_live,
        prepare_at_build_time=prepare,
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


def _seed_spawn_config_from_layouts(
    assets: list[PlaceableAsset],
    anchor_assets: set[PlaceableAsset],
    layouts: list[PlacementResult],
) -> None:
    """Seed every scene root from environment-indexed solved layouts without asset reset events."""
    poses_by_asset: dict[PlaceableAsset, dict[str, list[Pose]]] = {}
    for layout in layouts:
        root_poses_by_asset = get_scene_root_poses_from_layout(assets, layout, anchor_assets)
        for asset, root_poses in root_poses_by_asset.items():
            poses_per_root = poses_by_asset.setdefault(asset, {key: [] for key in root_poses})
            for key, pose in root_poses.items():
                poses_per_root[key].append(pose)
    for asset, poses_per_root in poses_by_asset.items():
        asset.set_initial_scene_root_poses({key: PosePerEnv(poses=poses) for key, poses in poses_per_root.items()})


def _seed_spawn_config_from_replay(
    samples: list[dict[str, Any]],
    replay_assets: list[PlaceableAsset],
    num_envs: int,
) -> None:
    """Seed construction roots from replay rows without creating asset reset events."""
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

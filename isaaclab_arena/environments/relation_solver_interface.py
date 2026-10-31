# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import TYPE_CHECKING

from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_events import PlacementSamplerHandle, solve_and_place_objects
from isaaclab_arena.relations.placement_poses import get_scene_root_poses_from_layout
from isaaclab_arena.relations.placement_sampler import (
    PlacementSample,
    PlacementSampler,
    RecordedPlacementSampler,
    deserialize_placement_samples,
    placement_samples_to_pose_columns,
    validate_placement_sample_assets,
    validate_root_reset_for_live_placement,
    validate_root_reset_for_placement_replay,
)
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
    from isaaclab_arena.variations.variation_replay_scheduler import VariationReplayScheduler


def prepare_live_relation_placement(
    assets: list[PlaceableAsset],
    num_envs: int,
    placer_params: ObjectPlacerParams | None = None,
    collision_objects: list[CollisionObject] | None = None,
    scene_assets: Iterable[Asset | RigidObjectSet] | None = None,
) -> PlacementSampler | None:
    """Prepare and return a live relation placement sampler.

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
        Live placement sampler, or ``None`` when there is no placement.
    """
    if not assets:
        print("No assets with relations found in scene. Skipping relation solving.")
        return None
    asset_names = {asset.name for asset in assets}
    assert len(asset_names) == len(assets), "Placement asset names must be unique"
    scene_keys = [asset.get_scene_key() for asset in assets]
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
        # Import after SimulationApp starts to avoid preloading USD before Kit.
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
    # Validators are built once above and reused for every refill, so the embodiment is done being read; drop
    # it before the reset-event params below capture (and deep-copy/validate) the pool.
    placer_params.reachability_config.embodiment = None

    if placement_pool.had_fallbacks:
        print(
            "Warning: Relation placement pool accepted best-loss fallback layouts "
            "that failed strict placement validation."
        )

    anchor_assets = set(get_anchor_objects(assets))
    # Prevent external pose-reset events from conflicting with relation-solved assets.
    validate_root_reset_for_live_placement(assets)

    # Anchor assets do not move, so no need to apply reset event.
    if anchor_assets == set(assets):
        return None

    sampler = PlacementSampler(
        assets=assets,
        placement_pool=placement_pool,
        write_assets=[asset for asset in assets if asset not in anchor_assets],
    )
    layouts = sampler.prepare_live(num_envs, placer_params.resolve_on_reset)
    _seed_spawn_config_from_layouts(assets, anchor_assets, layouts)
    return sampler


def prepare_recorded_relation_placement(
    replay_assets: list[PlaceableAsset],
    scheduler: VariationReplayScheduler,
    num_envs: int,
) -> RecordedPlacementSampler:
    """Validate and prepare recorded relation placement without constructing a solver pool.

    Args:
        replay_assets: Scene assets available to own recorded pose roots.
        scheduler: Shared scheduler assigning recorded episodes to environments.
        num_envs: Number of environments whose construction poses should be seeded.

    Returns:
        Recorded placement sampler configured for replay.
    """
    raw_samples = [
        episode_record.placement_sample
        for episode_record in scheduler.variation_record.episode_records
        if episode_record.placement_sample is not None
    ]
    recorded_samples = deserialize_placement_samples(raw_samples)
    selected_assets = validate_placement_sample_assets(recorded_samples, replay_assets)
    validate_root_reset_for_placement_replay(selected_assets)
    for asset in selected_assets:
        asset.clear_pose_reset_event()
    sampler = RecordedPlacementSampler(
        recorded_samples=recorded_samples,
        scheduler=scheduler,
        write_assets=selected_assets,
    )
    _seed_spawn_config_from_replay(recorded_samples, selected_assets, num_envs)
    return sampler


def make_placement_event(sampler: PlacementSampler | RecordedPlacementSampler) -> EventTermCfg:
    """Return the coordinated builder-owned placement reset event."""
    from isaaclab.managers import EventTermCfg

    return EventTermCfg(
        func=solve_and_place_objects,
        mode="reset",
        params={"placement_sampler": PlacementSamplerHandle(sampler)},
    )


def _seed_spawn_config_from_layouts(
    assets: list[PlaceableAsset],
    anchor_assets: set[PlaceableAsset],
    layouts: list[PlacementResult],
) -> None:
    """Seed every scene root from environment-indexed solved layouts."""
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
    samples: list[PlacementSample],
    write_assets: list[PlaceableAsset],
    num_envs: int,
) -> None:
    """Seed construction roots from recorded rows."""
    pose_columns = placement_samples_to_pose_columns(samples)
    for asset in write_assets:
        poses = {}
        for scene_key in asset.get_scene_root_keys():
            source_poses = pose_columns[scene_key]
            poses[scene_key] = PosePerEnv([source_poses[env_id % len(samples)] for env_id in range(num_envs)])
        asset.set_initial_scene_root_poses(poses)

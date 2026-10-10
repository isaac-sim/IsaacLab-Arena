# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_events import PlacementPoolHandle, solve_and_place_objects
from isaaclab_arena.relations.placement_poses import get_scene_root_poses_from_layout
from isaaclab_arena.relations.placement_sampler import (
    PlacementSample,
    PlacementSampler,
    placement_samples_to_pose_columns,
    validate_placement_sample_assets,
    validate_root_reset_for_placement_replay,
)
from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
from isaaclab_arena.relations.relations import get_anchor_objects
from isaaclab_arena.utils.pose import Pose, PosePerEnv

if TYPE_CHECKING:
    import torch

    from isaaclab.managers import EventTermCfg

    from isaaclab_arena.assets.asset import Asset
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.relations.collision_object import CollisionObject
    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.relations.placement_result import PlacementResult


def solve_and_apply_relation_placement(
    assets: list[PlaceableAsset],
    num_envs: int,
    placer_params: ObjectPlacerParams | None = None,
    collision_objects: list[CollisionObject] | None = None,
    scene_assets: Iterable[Asset | RigidObjectSet] | None = None,
    recorded_samples: list[PlacementSample] | None = None,
    replay_assets: list[PlaceableAsset] | None = None,
    replay_sampler: Callable[[int, torch.Tensor | None], list[PlacementSample] | None] | None = None,
) -> EventTermCfg | None:
    """Prepare live or recorded relation placement and return its reset event.

    Args:
        assets: Assets with spatial predicates that should be relation-solved.
        num_envs: Number of environments to prepare placements for.
        placer_params: Optional placement parameters. A shallow copy is used so
            this function can force pooled placement without mutating the caller's instance.
        collision_objects: Fixed obstacles avoided during placement but never optimized
            or relation-constrained.
        scene_assets: Optional scene assets to scan for passive collision objects
            when collision_objects is not supplied.
        recorded_samples: Complete recorded placement rows to validate and use.
        replay_assets: Assets whose scene roots may receive recorded poses.
        replay_sampler: Episode-scheduled callable that supplies recorded rows.

    Returns:
        Coordinated placement reset event, or ``None`` when there is no placement.
    """
    if not assets and recorded_samples is None:
        print("No assets with relations found in scene. Skipping relation solving.")
        return None
    asset_names = {asset.name for asset in assets}
    assert len(asset_names) == len(assets), "Placement asset names must be unique"
    from isaaclab_arena.relations.placement_asset import get_scene_root_owners

    if placer_params is None:
        placer_params = ObjectPlacerParams()
    else:
        placer_params = copy.copy(placer_params)
    if recorded_samples is not None:
        assert replay_sampler is not None, "Recorded placement replay requires an episode scheduler"
        sampler = PlacementSampler(
            assets=assets,
            placement_pool=None,
            write_assets=replay_assets or assets,
        )
        selected_assets = validate_placement_sample_assets(recorded_samples, sampler.write_assets)
        validate_root_reset_for_placement_replay(selected_assets)
        sampler.write_assets = selected_assets
        sampler.set_replay_sampler(replay_sampler)
        _seed_spawn_config_from_replay(recorded_samples, selected_assets, num_envs)
        return _make_placement_event(sampler)

    get_scene_root_owners(assets)
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

    return _apply_relation_placement_result(
        assets=assets,
        placer_params=placer_params,
        placement_pool=placement_pool,
        num_envs=num_envs,
    )


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

    sampler = PlacementSampler(
        assets=assets,
        placement_pool=placement_pool,
        write_assets=[asset for asset in assets if asset not in anchor_assets],
    )
    layouts = sampler.prepare_live(num_envs, placer_params.resolve_on_reset)
    _seed_spawn_config_from_layouts(assets, anchor_assets, layouts)
    return _make_placement_event(sampler)


def _make_placement_event(sampler: PlacementSampler) -> EventTermCfg:
    """Return the coordinated builder-owned placement reset event."""
    from isaaclab.managers import EventTermCfg

    return EventTermCfg(
        func=solve_and_place_objects,
        mode="reset",
        params={"placement_pool": PlacementPoolHandle(sampler)},
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

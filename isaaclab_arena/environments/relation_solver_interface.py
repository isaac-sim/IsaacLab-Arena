# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import TYPE_CHECKING

from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
from isaaclab_arena.relations.relations import get_anchor_objects

if TYPE_CHECKING:
    from isaaclab_arena.assets.asset import Asset
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.relations.collision_object import CollisionObject
    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.variations.relation_placement_variation import RelationPlacementVariation
    from isaaclab_arena.variations.variation_base import VariationBase


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
) -> RelationPlacementVariation | None:
    """Declare relation placement without solving, sampling, or mutating spawn poses.

    Args:
        assets: Assets participating in live relation solving.
        num_envs: Number of parallel simulation environments.
        placer_params: Optional live relation-solver parameters.
        collision_objects: Optional fixed obstacles used by the live solver.
        scene_assets: Complete scene assets used to derive fixed obstacles when
            ``collision_objects`` is omitted.
        replay_assets: Assets whose scene roots may receive recorded poses.
        live_placement_enabled: Whether a live pool may supply placement samples.
        replay_may_be_configured: Whether to declare provisionally so recorded
            placement rows can be discovered during replay binding.

    Returns:
        The declared placement variation, or ``None`` when neither live nor
        recorded placement can supply samples.
    """
    from isaaclab_arena.variations.relation_placement_variation import PlacementPoolSampler, RelationPlacementVariation

    anchor_assets = set(get_anchor_objects(assets))
    can_prepare_live = live_placement_enabled and bool(assets) and anchor_assets != set(assets)
    if not can_prepare_live and not replay_may_be_configured:
        return None

    replay_assets = assets if replay_assets is None else replay_assets
    placement_pool = None
    if can_prepare_live:
        placement_pool = _build_relation_placement_pool(
            assets=assets,
            num_envs=num_envs,
            placer_params=placer_params,
            collision_objects=collision_objects,
            scene_assets=scene_assets,
        )
    sampler = PlacementPoolSampler(
        assets=assets,
        placement_pool=placement_pool,
        replay_assets=replay_assets,
    )

    return RelationPlacementVariation(
        sampler,
        num_envs=num_envs,
    )


def finalize_relation_placement_variations(variations: list[VariationBase]) -> list[VariationBase]:
    """Remove a provisional placement declaration that has no live or replay sample source."""
    from isaaclab_arena.variations.relation_placement_variation import RelationPlacementVariation

    return [
        variation
        for variation in variations
        if not isinstance(variation, RelationPlacementVariation) or variation.can_supply_samples
    ]


def _build_relation_placement_pool(
    assets: list[PlaceableAsset],
    num_envs: int,
    placer_params: ObjectPlacerParams | None,
    collision_objects: list[CollisionObject] | None,
    scene_assets: Iterable[Asset | RigidObjectSet] | None,
) -> PooledObjectPlacer:
    """Validate relation assets and build their reusable placement pool."""
    assert assets, "Relation placement requires at least one asset"
    asset_names = {asset.name for asset in assets}
    assert len(asset_names) == len(assets), "Placement asset names must be unique"
    scene_keys = [asset.get_scene_key() for asset in assets]
    assert len(set(scene_keys)) == len(scene_keys), "Placement assets map to duplicate scene keys"

    placer_params = ObjectPlacerParams() if placer_params is None else copy.copy(placer_params)
    placer_params.apply_positions_to_objects = False
    # The pool releases its build-time embodiment reference after preparation;
    # isolate that mutation from the caller's reachability configuration.
    placer_params.reachability_config = copy.copy(placer_params.reachability_config)
    if collision_objects is None and scene_assets is not None:
        # Keep USD-dependent collision discovery out of module import time,
        # before Isaac Sim's application context is available.
        from isaaclab_arena.relations.passive_collision_objects import get_placement_collision_objects

        collision_objects = get_placement_collision_objects(
            assets, scene_assets, placer_params.solver_params.collision_mode
        )
    return PooledObjectPlacer(
        objects=assets,
        placer_params=placer_params,
        pool_size=num_envs * placer_params.min_unique_layouts_per_env,
        num_envs=num_envs,
        collision_objects=collision_objects,
        defer_initial_fill=True,
    )

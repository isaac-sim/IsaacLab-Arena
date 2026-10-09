# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Resolve variation choices before placement and native scene composition."""

from collections.abc import Iterable

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_reference import ObjectReference
from isaaclab_arena.utils.usd.rigid_asset_preparation import prepare_rigid_object_variants
from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariation
from isaaclab_arena.variations.variation_base import BuildTimeVariationBase


def _get_asset_selection(asset: Asset) -> AssetSelectionVariation | None:
    return next(
        (variation for variation in asset.get_variations() if isinstance(variation, AssetSelectionVariation)), None
    )


def validate_asset_selections(assets: Iterable[Asset], *, has_recorded_placement: bool = False) -> None:
    """Reject stale definitions and unsupported selection combinations after overrides."""
    for asset in assets:
        if isinstance(asset, ObjectReference):
            selection = _get_asset_selection(asset.parent_asset)
            assert selection is None or not selection.enabled, (
                f"ObjectReference '{asset.name}' cannot refer into '{asset.parent_asset.name}' with enabled asset"
                " selection."
            )
        if not isinstance(asset, Object):
            continue
        selection = _get_asset_selection(asset)
        if selection is None:
            continue
        assert (
            selection.selected_candidate_names is None
        ), f"Object '{asset.name}' asset selection is already resolved; create a fresh environment for another build."
        if not selection.enabled:
            continue
        assert (
            not has_recorded_placement
        ), "Asset selection placement replay requires a build manifest and is not supported yet."
        for variation in asset.get_variations():
            assert not (
                variation is not selection and variation.enabled and isinstance(variation, BuildTimeVariationBase)
            ), f"Asset selection on '{asset.name}' cannot yet be combined with another build-time variation."


def resolve_object_assets(assets: Iterable[Asset], num_envs: int) -> None:
    """Resolve selected configurations and require every object to have an asset before placement."""
    for asset in assets:
        if not isinstance(asset, Object):
            continue
        selection = _get_asset_selection(asset)
        if selection is not None and selection.enabled:
            selected_names = selection.selected_candidate_names
            assert selected_names is not None, f"Object '{asset.name}' has no sampled asset selection."
            expected_count = num_envs if selection.cfg.sample_per_environment else 1
            assert len(selected_names) == expected_count, f"Object '{asset.name}' has an invalid asset selection count."
            candidate_indices = {name: index for index, name in enumerate(selection.candidate_names)}
            indices = tuple(candidate_indices[name] for name in selected_names)
            if not selection.cfg.sample_per_environment:
                indices *= num_envs
            spawn_configs = prepare_rigid_object_variants(selection.get_candidate_spawn_configs())
            asset._resolve_assets(spawn_configs, indices)
            selection.record_resolved_assets()
        asset.get_object_cfg()


def validate_resolved_asset_selections(assets: Iterable[Asset]) -> None:
    """Require selected assets and their settings to remain consistent through scene construction."""
    for asset in assets:
        selection = _get_asset_selection(asset)
        if selection is None:
            continue
        if selection.enabled or selection.selected_candidate_names is not None:
            selection.validate_resolved_selection()
            assert (
                isinstance(asset, Object) and asset.asset_indices_by_env is not None
            ), f"Object '{asset.name}' asset selection must be resolved before scene export."

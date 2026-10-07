# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bind recorded episode conditions to enabled variation samplers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_arena.variations.condition_scheduler import ConditionScheduler
from isaaclab_arena.variations.episode_conditions import (
    RebuildConditions,
    load_episode_conditions_overlay,
    validate_overlay_variation_keys,
)
from isaaclab_arena.variations.variation_base import BuildTimeVariationBase, RunTimeVariationBase

if TYPE_CHECKING:
    import torch

    from isaaclab_arena.variations.variation_base import VariationBase


def configure_condition_replay(
    path: str,
    variations: dict[str, list[VariationBase]],
) -> ConditionScheduler:
    """Load episode conditions, bind replay samplers, and return their scheduler.

    Args:
        path: Episode-result JSONL to replay.
        variations: Variations available in the environment.

    Returns:
        Scheduler configured to assign and replay the loaded conditions.
    """
    enabled = _enabled_variations_by_key(variations)
    conditions = load_episode_conditions_overlay(
        path,
        build_time_variation_keys={
            key for key, variation in enabled.items() if isinstance(variation, BuildTimeVariationBase)
        },
    )
    _validate_condition_replay_variations(enabled, conditions)
    scheduler = ConditionScheduler(conditions)
    _bind_condition_replay_samplers(enabled, conditions, scheduler)
    return scheduler


def _validate_condition_replay_variations(
    enabled: dict[str, VariationBase],
    conditions: RebuildConditions,
) -> None:
    """Validate condition keys, lifecycle phases, and per-row presence."""
    validate_overlay_variation_keys(conditions, set(enabled))
    for variation_key, variation in enabled.items():
        runtime_presence = [variation_key in episode.runtime_variations for episode in conditions.episodes]
        appears_at_runtime = any(runtime_presence)
        assert not appears_at_runtime or all(
            runtime_presence
        ), f"Runtime variation {variation_key!r} must be present in every source condition or none."
        if isinstance(variation, BuildTimeVariationBase):
            assert (
                not appears_at_runtime
            ), f"Build-time variation {variation_key!r} cannot appear in runtime_variations."
        elif isinstance(variation, RunTimeVariationBase):
            assert (
                variation_key not in conditions.build_time_variations
            ), f"Run-time variation {variation_key!r} cannot appear in build_time_variations."
            if appears_at_runtime:
                variation.validate_replay_samples(
                    [episode.runtime_variations[variation_key] for episode in conditions.episodes]
                )


def _bind_condition_replay_samplers(
    enabled: dict[str, VariationBase],
    conditions: RebuildConditions,
    scheduler: ConditionScheduler,
) -> None:
    """Replay recorded values while leaving absent enabled variations live-sampled."""
    for variation_key, variation in enabled.items():
        if isinstance(variation, BuildTimeVariationBase):
            if variation_key not in conditions.build_time_variations:
                variation.set_replay_sampler(None)
                continue

            def build_time_replay_sampler(
                _num_samples: int,
                env_ids: torch.Tensor | None,
                *,
                variation_key: str = variation_key,
            ) -> list:
                assert env_ids is None, f"Build-time variation {variation_key!r} received per-env ids."
                return [conditions.build_time_variations[variation_key]]

            variation.set_replay_sampler(build_time_replay_sampler)
            continue

        is_recorded = all(variation_key in episode.runtime_variations for episode in conditions.episodes)
        if not is_recorded:
            variation.set_replay_sampler(None)
            continue

        def runtime_replay_sampler(
            _num_samples: int,
            env_ids: torch.Tensor | None,
            *,
            variation_key: str = variation_key,
        ) -> list:
            assert env_ids is not None, f"Run-time variation {variation_key!r} requires per-env ids."
            return scheduler.runtime_sample_for(variation_key, env_ids.tolist())

        variation.set_replay_sampler(runtime_replay_sampler)


def _enabled_variations_by_key(
    variations: dict[str, list[VariationBase]],
) -> dict[str, VariationBase]:
    enabled: dict[str, VariationBase] = {}
    for asset_name, asset_variations in variations.items():
        for variation in asset_variations:
            if not variation.enabled:
                continue
            key = f"{asset_name}.{variation.name}"
            assert key not in enabled, f"Duplicate enabled variation record key: {key!r}"
            enabled[key] = variation
    return enabled

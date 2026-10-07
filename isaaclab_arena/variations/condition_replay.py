# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bind recorded episode conditions to enabled variation samplers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.variations.recorded_variation_samples import (
    RebuildVariationRecord,
    load_rebuild_variation_record,
    validate_recorded_variation_sample_keys,
)
from isaaclab_arena.variations.variation_base import BuildTimeVariationBase, RunTimeVariationBase

if TYPE_CHECKING:
    import torch

    from isaaclab_arena.variations.condition_scheduler import ConditionScheduler
    from isaaclab_arena.variations.variation_base import VariationBase


@dataclass
class ConditionReplayState:
    """Replay metadata and scheduler attached to a constructed environment."""

    scheduler: ConditionScheduler
    episode_results_source: str


def load_variation_conditions(
    path: str,
    variations: dict[str, list[VariationBase]],
) -> RebuildVariationRecord:
    """Load and validate rebuild conditions against the enabled variations."""
    variation_record = load_rebuild_variation_record(
        path,
        build_time_variation_keys=enabled_build_time_variation_keys(variations),
    )
    validate_condition_replay_variations(variations, variation_record)
    return variation_record


def validate_condition_replay_variations(
    variations: dict[str, list[VariationBase]],
    variation_record: RebuildVariationRecord,
) -> None:
    """Validate condition keys, lifecycle phases, and per-row presence."""
    enabled = _enabled_variations_by_key(variations)
    validate_recorded_variation_sample_keys(variation_record, set(enabled))
    for variation_key, variation in enabled.items():
        runtime_presence = [
            variation_key in episode_record.runtime_samples for episode_record in variation_record.episode_records
        ]
        appears_at_runtime = any(runtime_presence)
        assert not appears_at_runtime or all(
            runtime_presence
        ), f"Run-time variation {variation_key!r} must be present in every source record or none."
        if isinstance(variation, BuildTimeVariationBase):
            assert (
                not appears_at_runtime
            ), f"Build-time variation {variation_key!r} cannot appear in run-time samples."
        elif isinstance(variation, RunTimeVariationBase):
            assert (
                variation_key not in variation_record.build_time_samples
            ), f"Run-time variation {variation_key!r} cannot appear in build-time samples."


def bind_condition_replay_samplers(
    variations: dict[str, list[VariationBase]],
    variation_record: RebuildVariationRecord,
    scheduler: ConditionScheduler,
) -> None:
    """Replay recorded values while leaving absent enabled variations live-sampled."""
    validate_condition_replay_variations(variations, variation_record)
    for variation_key, variation in _enabled_variations_by_key(variations).items():
        if isinstance(variation, BuildTimeVariationBase):
            if variation_key not in variation_record.build_time_samples:
                variation.set_replay_sampler(None)
                continue

            def build_time_replay_sampler(
                _num_samples: int,
                env_ids: torch.Tensor | None,
                *,
                variation_key: str = variation_key,
            ) -> list:
                assert env_ids is None, f"Build-time variation {variation_key!r} received per-env ids."
                return [variation_record.build_time_samples[variation_key]]

            variation.set_replay_sampler(build_time_replay_sampler)
            continue

        is_recorded = all(
            variation_key in episode_record.runtime_samples for episode_record in variation_record.episode_records
        )
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


def enabled_build_time_variation_keys(variations: dict[str, list[VariationBase]]) -> set[str]:
    """Return record keys for enabled build-time variations."""
    return {
        key
        for key, variation in _enabled_variations_by_key(variations).items()
        if isinstance(variation, BuildTimeVariationBase)
    }


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

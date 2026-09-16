# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Helpers for applying recorded variation samples during replay."""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena.variations.condition_scheduler import ConditionScheduler
    from isaaclab_arena.variations.episode_conditions import EpisodeConditionsOverlay
    from isaaclab_arena.variations.sampler_base import SamplerBase
    from isaaclab_arena.variations.variation_base import VariationBase


@dataclass
class ConditionReplayState:
    """Replay metadata and scheduler attached to a constructed environment."""

    overlay: EpisodeConditionsOverlay
    scheduler: ConditionScheduler
    episode_results_source: str | None = None


def bind_variation_record_keys(variations: dict[str, list[VariationBase]]) -> None:
    """Set ``variation._record_key`` for every enabled variation host."""
    for asset_name, asset_variations in variations.items():
        for variation in asset_variations:
            if variation.enabled:
                variation._record_key = f"{asset_name}.{variation.name}"


def enabled_variation_record_keys(variations: dict[str, list[VariationBase]]) -> set[str]:
    """Return ``{host.variation}`` keys for enabled variations."""
    keys: set[str] = set()
    for asset_name, asset_variations in variations.items():
        for variation in asset_variations:
            if variation.enabled:
                keys.add(f"{asset_name}.{variation.name}")
    return keys


def notify_variation_sample(variation: VariationBase, sample: Any, env_ids: torch.Tensor | None) -> None:
    """Forward a sample through variation-owned recorder listeners."""
    for listener in variation._sample_listeners:
        listener(sample, env_ids)


def draw_runtime_variation_sample(
    env: ManagerBasedEnv,
    *,
    variation_key: str,
    variation: VariationBase,
    env_ids: torch.Tensor,
    sampler: SamplerBase,
    num_samples: int,
) -> Any:
    """Draw from the replay scheduler or the live sampler."""
    replay: ConditionReplayState | None = getattr(env.unwrapped, "condition_replay", None)
    if replay is not None:
        rows = replay.scheduler.runtime_sample_for(variation_key, env_ids.tolist())
        sample = _rows_to_sample(rows, sampler=sampler, device=env_ids.device)
        notify_variation_sample(variation, sample, env_ids)
        return sample
    sample = sampler.sample(num_samples=num_samples, env_ids=env_ids)
    return sample


def _rows_to_sample(rows: list[Any], *, sampler: SamplerBase, device: torch.device) -> Any:
    if not rows:
        return sampler.sample(num_samples=0)
    if isinstance(rows[0], (int, float)):
        tensor = torch.tensor(rows, device=device, dtype=torch.float32).reshape(len(rows), -1)
        return tensor
    if isinstance(rows[0], list):
        tensor = torch.tensor(rows, device=device, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.reshape(len(rows), -1)
        return tensor
    return rows

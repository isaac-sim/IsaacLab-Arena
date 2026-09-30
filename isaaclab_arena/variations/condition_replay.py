# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Helpers for applying recorded variation samples during replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from isaaclab_arena.variations.condition_scheduler import ConditionScheduler
    from isaaclab_arena.variations.episode_conditions import EpisodeConditionsOverlay
    from isaaclab_arena.variations.variation_base import VariationBase


@dataclass
class ConditionReplayState:
    """Replay metadata and scheduler attached to a constructed environment."""

    overlay: EpisodeConditionsOverlay
    scheduler: ConditionScheduler
    episode_results_source: str | None = None


def bind_condition_replay_sample_overrides(
    variations: dict[str, list[VariationBase]],
    overlay: EpisodeConditionsOverlay,
    scheduler: ConditionScheduler,
) -> None:
    """Make enabled variation samplers consume recorded condition rows."""
    for asset_name, asset_variations in variations.items():
        for variation in asset_variations:
            if not variation.enabled:
                continue
            variation_key = f"{asset_name}.{variation.name}"

            def sample_override(
                num_samples: int,
                env_ids: torch.Tensor | None,
                *,
                variation_key: str = variation_key,
            ) -> list | None:
                if env_ids is None:
                    if variation_key not in overlay.build_time_variations:
                        return None
                    rows = [overlay.build_time_variations[variation_key]]
                else:
                    rows = scheduler.runtime_sample_for(variation_key, env_ids.tolist())
                assert len(rows) == num_samples, (
                    f"Condition replay returned {len(rows)} rows for variation {variation_key!r}; "
                    f"expected {num_samples}."
                )
                return rows

            variation.set_sample_override_provider(sample_override)


def enabled_variation_record_keys(variations: dict[str, list[VariationBase]]) -> set[str]:
    """Return ``{host.variation}`` keys for enabled variations."""
    keys: set[str] = set()
    for asset_name, asset_variations in variations.items():
        for variation in asset_variations:
            if variation.enabled:
                keys.add(f"{asset_name}.{variation.name}")
    return keys

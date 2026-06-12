# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Turn an environment's enabled variations into sensitivity-analysis inputs.

Two views are derived, both keyed by the dotted ``<asset>.<variation>`` path:

* :func:`collect_variation_factor_schema` — the run-level *prior* (type + range/choices)
  each enabled variation samples from. Emitted once per run into the episode-summary header.
* :func:`collect_variation_draws` — the *realized* value each enabled variation last sampled.
  Recorded per build (build-time variations) so the episode writer can log it per episode.

Only build-time variations populate :attr:`~isaaclab_arena.variations.variation_base.VariationBase.last_draw`
today, so run-time variations contribute schema but no draws yet (see ``last_draw``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from isaaclab_arena.variations.choice_sampler import ChoiceSampler
from isaaclab_arena.variations.uniform_sampler import UniformSampler

if TYPE_CHECKING:
    from isaaclab_arena.variations.variation_base import VariationBase


def collect_variation_draws(variations: dict[str, list[VariationBase]]) -> dict[str, Any]:
    """Return ``{<asset>.<variation>: realized_value}`` for every enabled variation that sampled.

    Args:
        variations: ``{asset_name: [variation, ...]}`` as returned by the builder.

    Returns:
        One entry per enabled variation whose ``last_draw`` is set. Values are JSON primitives
        (float for continuous, str for categorical) so they drop straight into the JSONL.
    """
    draws: dict[str, Any] = {}
    for asset_name, asset_variations in variations.items():
        for variation in asset_variations:
            if not variation.enabled or variation.last_draw is None:
                continue
            draws[f"{asset_name}.{variation.name}"] = variation.last_draw
    return draws


def collect_variation_factor_schema(variations: dict[str, list[VariationBase]]) -> dict[str, dict]:
    """Return ``{<asset>.<variation>: factor_spec}`` for every enabled, recognised variation.

    Each ``factor_spec`` mirrors a ``factors.yaml`` entry: ``type`` plus ``range`` (continuous)
    derived from the sampler. Categorical ``choices`` are left out — they are inferred from the
    observed draws at analysis time, keeping this generic over the choice pool.

    Args:
        variations: ``{asset_name: [variation, ...]}`` as returned by the builder.

    Returns:
        One entry per enabled variation whose sampler family is recognised; others are skipped.
    """
    schema: dict[str, dict] = {}
    for asset_name, asset_variations in variations.items():
        for variation in asset_variations:
            if not variation.enabled:
                continue
            factor_spec = _variation_factor_spec(variation)
            if factor_spec is not None:
                schema[f"{asset_name}.{variation.name}"] = factor_spec
    return schema


def _variation_factor_spec(variation: VariationBase) -> dict | None:
    """Derive a factors.yaml-style spec from a variation's sampler, or ``None`` if unrecognised."""
    sampler = variation.sampler
    if isinstance(sampler, UniformSampler):
        low = sampler.low.flatten().tolist()
        high = sampler.high.flatten().tolist()
        return {
            "type": "continuous",
            "range": [[dim_low, dim_high] for dim_low, dim_high in zip(low, high)],
            "distribution": "uniform",
        }
    if isinstance(sampler, ChoiceSampler):
        # Choices are inferred from the observed draws at load time (the pool isn't on the sampler).
        return {"type": "categorical"}
    return None

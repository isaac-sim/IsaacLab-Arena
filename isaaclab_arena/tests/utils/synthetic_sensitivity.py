# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Synthetic sensitivity datasets with a *known* ground-truth relationship.

This is the Arena analogue of the ``simple_simulator`` in robolab's MNPE demo: it samples
factors from a uniform prior, runs them through a fixed generative model, and returns a
:class:`SensitivityDataset` of in-memory ``theta`` / ``x`` tensors — no ``factors.yaml`` or
``episode_summary.jsonl`` round-trip. Because the planted relationship is known, a test can
fit an analyzer on the data and assert the recovered posterior reflects it.

Ground truth (single-sourced in the constants below):
  - ``light_intensity`` is continuous over ``LIGHT_RANGE``; higher light raises success
    (``LIGHT_WEIGHT > 0``).
  - ``table_material`` is categorical; ``MATERIAL_BASE_LOGIT`` makes oak the most successful
    material and bamboo the least.
  - ``success`` is a binary outcome drawn from ``Bernoulli(sigmoid(logit))``.
"""

from __future__ import annotations

import torch

from isaaclab_arena.analysis.sensitivity.dataset import (
    FactorSchema,
    FactorSpec,
    OutcomeSpec,
    SensitivityDataset,
    SliceSpec,
)

LIGHT_RANGE: tuple[float, float] = (0.0, 5000.0)
"""Range of the continuous ``light_intensity`` factor."""

LIGHT_WEIGHT: float = 2.5
"""Success-logit gain per unit of normalized light. Positive ⇒ brighter is more successful."""

MATERIAL_BASE_LOGIT: dict[str, float] = {"oak": 1.5, "walnut": 0.0, "bamboo": -1.5}
"""Per-material base success logit. Ordered best→worst, so oak should dominate the posterior."""

_OUTCOME_NAME = "success"
_SLICE = SliceSpec(policy="synthetic", task="SyntheticTask", embodiment="synthetic")


def _normalized_light(light_intensity: torch.Tensor) -> torch.Tensor:
    """Map ``light_intensity`` from ``LIGHT_RANGE`` onto roughly ``[-1, 1]`` for the logit."""
    low, high = LIGHT_RANGE
    midpoint = 0.5 * (low + high)
    half_range = 0.5 * (high - low)
    return (light_intensity - midpoint) / half_range


def _sample_success(success_logit: torch.Tensor) -> torch.Tensor:
    """Draw a binary success outcome per episode from ``Bernoulli(sigmoid(logit))``."""
    return torch.bernoulli(torch.sigmoid(success_logit))


def make_continuous_dataset(seed: int, num_episodes: int = 2000) -> SensitivityDataset:
    """Single continuous factor (``light_intensity``) driving a binary ``success`` outcome.

    Dispatches to ``KDEAnalyzer``. Success probability rises monotonically with light, so the
    posterior over successful-episode light values should concentrate at the bright end.
    """
    torch.manual_seed(seed)

    low, high = LIGHT_RANGE
    light_intensity = torch.rand(num_episodes) * (high - low) + low
    success = _sample_success(LIGHT_WEIGHT * _normalized_light(light_intensity))

    schema = FactorSchema(
        slice=_SLICE,
        factors=[FactorSpec(name="light_intensity", type="continuous", range=[list(LIGHT_RANGE)])],
        outcomes=[OutcomeSpec(name=_OUTCOME_NAME, type="bool")],
    )
    theta = light_intensity.unsqueeze(1)
    x = success.unsqueeze(1)
    return SensitivityDataset(schema, theta, x)


def make_mixed_dataset(seed: int, num_episodes: int = 2000) -> SensitivityDataset:
    """Continuous ``light_intensity`` + categorical ``table_material`` driving ``success``.

    Dispatches to ``MNPEAnalyzer``. Both effects are planted: brighter light and "better"
    materials (oak > walnut > bamboo) raise success, so conditioning the posterior on success
    should favor high light values and oak.
    """
    torch.manual_seed(seed)

    materials = list(MATERIAL_BASE_LOGIT)
    material_base_logits = torch.tensor([MATERIAL_BASE_LOGIT[m] for m in materials])

    low, high = LIGHT_RANGE
    light_intensity = torch.rand(num_episodes) * (high - low) + low
    material_code = torch.randint(0, len(materials), (num_episodes,))
    success_logit = material_base_logits[material_code] + LIGHT_WEIGHT * _normalized_light(light_intensity)
    success = _sample_success(success_logit)

    schema = FactorSchema(
        slice=_SLICE,
        factors=[
            FactorSpec(name="light_intensity", type="continuous", range=[list(LIGHT_RANGE)]),
            FactorSpec(name="table_material", type="categorical", choices=materials),
        ],
        outcomes=[OutcomeSpec(name=_OUTCOME_NAME, type="bool")],
    )
    # Continuous column first, then the integer-coded categorical column — the layout
    # SensitivityDataset.factor_columns describes and the analyzers expect.
    theta = torch.stack([light_intensity, material_code.float()], dim=1)
    x = success.unsqueeze(1)
    return SensitivityDataset(schema, theta, x)

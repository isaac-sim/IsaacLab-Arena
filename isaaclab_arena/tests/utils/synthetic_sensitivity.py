# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Synthetic sensitivity datasets with a *known* ground-truth relationship.

This is the Arena analogue of the ``simple_simulator`` in robolab's MNPE demo: it samples
factors from a uniform prior, runs them through a fixed generative model, and returns a
:class:`SensitivityDataset` of in-memory ``theta`` / ``x`` tensors — no ``factors.yaml`` or
``episode_summary.jsonl`` round-trip. Because the planted relationship is known, a test can
fit a :class:`SensitivityAnalyzer` on the data and assert the recovered posterior reflects it.

Ground truth (single-sourced in the constants below):
  - ``light_intensity`` is continuous; higher light raises success (``LIGHT_WEIGHT > 0``).
  - ``grasp_offset`` is continuous; a *smaller* offset raises success (``OFFSET_WEIGHT > 0``).
  - ``table_material`` is categorical; ``MATERIAL_BASE_LOGIT`` makes oak the most successful
    material and bamboo the least.
  - ``success`` is a binary outcome drawn from ``Bernoulli(sigmoid(logit))``.

``make_mixed_dataset`` exercises the MNPE path (continuous + categorical); ``make_continuous_dataset``
exercises the NPE path with two continuous factors (NPE restricts to a Gaussian on 1-D theta).
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

GRASP_OFFSET_RANGE: tuple[float, float] = (0.0, 0.2)
"""Range (metres) of the continuous ``grasp_offset`` factor."""

OFFSET_WEIGHT: float = 2.5
"""Success-logit gain per unit of normalized offset. Subtracted ⇒ a smaller offset is more successful."""

MATERIAL_BASE_LOGIT: dict[str, float] = {"oak": 1.5, "walnut": 0.0, "bamboo": -1.5}
"""Per-material base success logit. Ordered best→worst, so oak should dominate the posterior."""

_OUTCOME_NAME = "success"
_SLICE = SliceSpec(policy="synthetic", task="SyntheticTask", embodiment="synthetic")


def _normalized(values: torch.Tensor, value_range: tuple[float, float]) -> torch.Tensor:
    """Map ``values`` from ``value_range`` onto roughly ``[-1, 1]`` for the success logit."""
    low, high = value_range
    midpoint = 0.5 * (low + high)
    half_range = 0.5 * (high - low)
    return (values - midpoint) / half_range


def _sample_uniform(value_range: tuple[float, float], num_episodes: int) -> torch.Tensor:
    """Draw ``num_episodes`` values uniformly over ``value_range``."""
    low, high = value_range
    return torch.rand(num_episodes) * (high - low) + low


def _sample_success(success_logit: torch.Tensor) -> torch.Tensor:
    """Draw a binary success outcome per episode from ``Bernoulli(sigmoid(logit))``."""
    return torch.bernoulli(torch.sigmoid(success_logit))


def make_continuous_dataset(seed: int, num_episodes: int = 2000) -> SensitivityDataset:
    """Two continuous factors (``light_intensity``, ``grasp_offset``) driving ``success``.

    Uses the NPE path. Both effects are planted — brighter light and a smaller grasp offset
    raise success — so conditioning the posterior on success should favor high light values
    and low offset values. Two factors keep theta 2-D, away from NPE's 1-D Gaussian fallback.
    """
    torch.manual_seed(seed)

    light_intensity = _sample_uniform(LIGHT_RANGE, num_episodes)
    grasp_offset = _sample_uniform(GRASP_OFFSET_RANGE, num_episodes)
    success_logit = LIGHT_WEIGHT * _normalized(light_intensity, LIGHT_RANGE) - OFFSET_WEIGHT * _normalized(
        grasp_offset, GRASP_OFFSET_RANGE
    )
    success = _sample_success(success_logit)

    schema = FactorSchema(
        slice=_SLICE,
        factors=[
            FactorSpec(name="light_intensity", type="continuous", range=[list(LIGHT_RANGE)]),
            FactorSpec(name="grasp_offset", type="continuous", range=[list(GRASP_OFFSET_RANGE)]),
        ],
        outcomes=[OutcomeSpec(name=_OUTCOME_NAME, type="bool")],
    )
    theta = torch.stack([light_intensity, grasp_offset], dim=1)
    x = success.unsqueeze(1)
    return SensitivityDataset(schema, theta, x)


def make_mixed_dataset(seed: int, num_episodes: int = 2000) -> SensitivityDataset:
    """Continuous ``light_intensity`` + categorical ``table_material`` driving ``success``.

    Uses the MNPE path. Both effects are planted: brighter light and "better" materials
    (oak > walnut > bamboo) raise success, so conditioning the posterior on success should
    favor high light values and oak.
    """
    torch.manual_seed(seed)

    materials = list(MATERIAL_BASE_LOGIT)
    material_base_logits = torch.tensor([MATERIAL_BASE_LOGIT[m] for m in materials])

    light_intensity = _sample_uniform(LIGHT_RANGE, num_episodes)
    material_code = torch.randint(0, len(materials), (num_episodes,))
    success_logit = material_base_logits[material_code] + LIGHT_WEIGHT * _normalized(light_intensity, LIGHT_RANGE)
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
    # SensitivityDataset.factor_columns describes and the estimators expect.
    theta = torch.stack([light_intensity, material_code.float()], dim=1)
    x = success.unsqueeze(1)
    return SensitivityDataset(schema, theta, x)


def _demo():
    """Run the full pipeline on a synthetic dataset and save the marginals plot.

    The Arena counterpart of running robolab's ``posterior_inference.py`` in generated-data
    mode: simulate → fit → plot, with no eval data needed. Run as::

        python -m isaaclab_arena.tests.utils.synthetic_sensitivity --kind mixed --output /tmp/demo.png
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run the sensitivity pipeline on a synthetic dataset and plot it.")
    parser.add_argument(
        "--kind", choices=["mixed", "continuous"], default="mixed", help="'mixed' (MNPE) or 'continuous' (NPE)."
    )
    parser.add_argument(
        "--output", default="./sensitivity_synthetic.png", help="Output figure path; format follows the extension."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=2000)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")

    from isaaclab_arena.analysis.sensitivity.analyzer import SensitivityAnalyzer
    from isaaclab_arena.analysis.sensitivity.plotting import plot_marginals

    builder = make_mixed_dataset if args.kind == "mixed" else make_continuous_dataset
    dataset = builder(seed=args.seed, num_episodes=args.num_episodes)
    analyzer = SensitivityAnalyzer(dataset)
    analyzer.fit()
    plot_marginals(analyzer, output_path=args.output)
    print(f"[INFO] Wrote synthetic sensitivity report → {args.output}")


if __name__ == "__main__":
    _demo()

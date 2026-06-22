# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Synthetic sensitivity datasets with a *known* ground-truth relationship.

A simple forward simulator: it samples factors from a uniform prior, runs them through a
fixed generative model, and returns a SensitivityDataset of in-memory theta / x tensors —
no factors.yaml or episode_summary.jsonl round-trip. Because the planted relationship is
known, a test can fit a SensitivityAnalyzer on the data and assert the recovered posterior
reflects it.

Ground truth (single-sourced in the factor definitions below):
  - light_intensity is continuous; brighter raises success (LIGHT.weight > 0).
  - grasp_offset is continuous; a *smaller* offset raises success (GRASP_OFFSET.weight < 0).
  - table_material is categorical; MATERIAL makes oak the most successful, bamboo the least.
  - success is a binary outcome drawn from Bernoulli(sigmoid(logit)).

make_mixed_dataset exercises the MNPE path (continuous + categorical); make_continuous_dataset
exercises the NPE path with two continuous factors (NPE restricts to a Gaussian on 1-D theta).
"""

from __future__ import annotations

import argparse
import torch
from dataclasses import dataclass

from isaaclab_arena.analysis.sensitivity.analyzer import SensitivityAnalyzer
from isaaclab_arena.analysis.sensitivity.dataset import FactorSchema, FactorSpec, SensitivityDataset
from isaaclab_arena.analysis.sensitivity.plotting import plot_marginals


@dataclass(frozen=True)
class _ContinuousFactor:
    """A continuous factor with a planted, signed effect on the success logit."""

    name: str
    value_range: tuple[float, float]
    weight: float  # success-logit gain per normalized unit; the sign sets the direction of the effect

    def sample(self, num_episodes: int) -> torch.Tensor:
        low, high = self.value_range
        return torch.rand(num_episodes) * (high - low) + low

    def logit(self, values: torch.Tensor) -> torch.Tensor:
        low, high = self.value_range
        normalized = (values - 0.5 * (low + high)) / (0.5 * (high - low))  # map value_range onto [-1, 1]
        return self.weight * normalized

    def spec(self) -> FactorSpec:
        return FactorSpec(name=self.name, type="continuous", range=[list(self.value_range)])

    def column(self, values: torch.Tensor) -> torch.Tensor:
        return values


@dataclass(frozen=True)
class _CategoricalFactor:
    """A categorical factor with a per-choice base success logit (ordered best→worst)."""

    name: str
    base_logit: dict[str, float]

    @property
    def choices(self) -> list[str]:
        return list(self.base_logit)

    def sample(self, num_episodes: int) -> torch.Tensor:
        return torch.randint(0, len(self.base_logit), (num_episodes,))

    def logit(self, codes: torch.Tensor) -> torch.Tensor:
        return torch.tensor([self.base_logit[choice] for choice in self.choices])[codes]

    def spec(self) -> FactorSpec:
        return FactorSpec(name=self.name, type="categorical", choices=self.choices)

    def column(self, codes: torch.Tensor) -> torch.Tensor:
        return codes.float()


# Planted ground truth: brighter light, a smaller grasp offset, a lighter object, a closer
# camera, and the leading category (oak / cube) all raise success.
LIGHT = _ContinuousFactor("light_intensity", (0.0, 5000.0), weight=2.5)
GRASP_OFFSET = _ContinuousFactor("grasp_offset", (0.0, 0.2), weight=-2.5)
OBJECT_MASS = _ContinuousFactor("object_mass", (0.05, 2.0), weight=-1.5)
CAMERA_DISTANCE = _ContinuousFactor("camera_distance", (0.3, 1.5), weight=-1.5)
MATERIAL = _CategoricalFactor("table_material", {"oak": 1.5, "walnut": 0.0, "bamboo": -1.5})
OBJECT_TYPE = _CategoricalFactor("object_type", {"cube": 1.2, "can": 0.0, "mug": -1.2})


def _sample_success(success_logit: torch.Tensor) -> torch.Tensor:
    """Draw a binary success outcome per episode from Bernoulli(sigmoid(logit))."""
    return torch.bernoulli(torch.sigmoid(success_logit))


def _build_dataset(
    factors_and_columns: list[tuple[_ContinuousFactor | _CategoricalFactor, torch.Tensor]],
    success: torch.Tensor,
) -> SensitivityDataset:
    """Assemble a SensitivityDataset from (factor, sampled column) pairs and the success outcome.

    Continuous factors are placed before the categorical ones, matching the layout
    SensitivityDataset.factor_columns expects.
    """
    ordered = sorted(factors_and_columns, key=lambda pair: isinstance(pair[0], _CategoricalFactor))
    schema = FactorSchema(factors=[factor.spec() for factor, _ in ordered])
    theta = torch.stack([factor.column(values) for factor, values in ordered], dim=1)
    # outcome_names defaults to ("success",), matching the single binary outcome built here.
    return SensitivityDataset(schema, theta, success.unsqueeze(1))


def make_continuous_dataset(seed: int, num_episodes: int = 2000) -> SensitivityDataset:
    """Two continuous factors (light_intensity, grasp_offset) driving success.

    Uses the NPE path. Both effects are planted — brighter light and a smaller grasp offset
    raise success — so conditioning the posterior on success should favor high light values
    and low offset values. Two factors keep theta 2-D, away from NPE's 1-D Gaussian fallback.
    """
    torch.manual_seed(seed)
    light = LIGHT.sample(num_episodes)
    grasp_offset = GRASP_OFFSET.sample(num_episodes)
    success = _sample_success(LIGHT.logit(light) + GRASP_OFFSET.logit(grasp_offset))
    return _build_dataset([(LIGHT, light), (GRASP_OFFSET, grasp_offset)], success)


def make_mixed_dataset(seed: int, num_episodes: int = 3000) -> SensitivityDataset:
    """Mixed continuous + categorical factors driving success (MNPE path).

    A realistic multi-factor sweep: three continuous factors on different scales (light,
    mass, camera distance) and two categoricals (object type, table material). Every effect
    is planted (brighter / lighter / closer / cube / oak raise success), so the posterior
    conditioned on success should recover all of them at once.
    """
    torch.manual_seed(seed)
    light = LIGHT.sample(num_episodes)
    object_mass = OBJECT_MASS.sample(num_episodes)
    camera_distance = CAMERA_DISTANCE.sample(num_episodes)
    object_type = OBJECT_TYPE.sample(num_episodes)
    material = MATERIAL.sample(num_episodes)
    success = _sample_success(
        LIGHT.logit(light)
        + OBJECT_MASS.logit(object_mass)
        + CAMERA_DISTANCE.logit(camera_distance)
        + OBJECT_TYPE.logit(object_type)
        + MATERIAL.logit(material)
    )
    return _build_dataset(
        [
            (LIGHT, light),
            (OBJECT_MASS, object_mass),
            (CAMERA_DISTANCE, camera_distance),
            (OBJECT_TYPE, object_type),
            (MATERIAL, material),
        ],
        success,
    )


def _demo():
    """Run the full pipeline on a synthetic dataset and save the marginals plot.

    Runs the pipeline end to end on generated data: simulate → fit → plot, with no eval
    data needed. Run as::

        python -m isaaclab_arena.tests.sensitivity_synthetic --kind mixed --output eval/demo.png
    """
    parser = argparse.ArgumentParser(description="Run the sensitivity pipeline on a synthetic dataset and plot it.")
    parser.add_argument(
        "--kind",
        choices=["mixed", "continuous"],
        default="mixed",
        help="'mixed' (continuous + categorical, MNPE) or 'continuous' (continuous-only, NPE).",
    )
    parser.add_argument(
        "--output",
        default="eval/sensitivity_synthetic.png",
        help="Output figure path; format follows the extension.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=2000)
    args = parser.parse_args()

    builder = {"mixed": make_mixed_dataset, "continuous": make_continuous_dataset}[args.kind]
    dataset = builder(seed=args.seed, num_episodes=args.num_episodes)
    analyzer = SensitivityAnalyzer(dataset)
    analyzer.fit()
    observation = dataset.default_observation()
    samples = analyzer.sample_posterior(observation)
    plot_marginals(samples, dataset, observation, output_path=args.output)
    print(f"[INFO] Wrote synthetic sensitivity report → {args.output}")


if __name__ == "__main__":
    _demo()

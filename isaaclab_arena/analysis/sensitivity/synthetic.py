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

Ground truth (single-sourced in the constants below):
  - light_intensity is continuous; higher light raises success (LIGHT_WEIGHT > 0).
  - grasp_offset is continuous; a *smaller* offset raises success (OFFSET_WEIGHT > 0).
  - table_material is categorical; MATERIAL_BASE_LOGIT makes oak the most successful
    material and bamboo the least.
  - success is a binary outcome drawn from Bernoulli(sigmoid(logit)).

make_mixed_dataset exercises the MNPE path (continuous + categorical); make_continuous_dataset
exercises the NPE path with two continuous factors (NPE restricts to a Gaussian on 1-D theta).
"""

from __future__ import annotations

import argparse
import torch

from isaaclab_arena.analysis.sensitivity.analyzer import SensitivityAnalyzer
from isaaclab_arena.analysis.sensitivity.dataset import (
    FactorSchema,
    FactorSpec,
    OutcomeSpec,
    SensitivityDataset,
    SliceSpec,
)
from isaaclab_arena.analysis.sensitivity.plotting import plot_marginals

LIGHT_RANGE: tuple[float, float] = (0.0, 5000.0)
"""Range of the continuous light_intensity factor."""

LIGHT_WEIGHT: float = 2.5
"""Success-logit gain per unit of normalized light. Positive ⇒ brighter is more successful."""

GRASP_OFFSET_RANGE: tuple[float, float] = (0.0, 0.2)
"""Range (metres) of the continuous grasp_offset factor."""

OFFSET_WEIGHT: float = 2.5
"""Success-logit gain per unit of normalized offset. Subtracted ⇒ a smaller offset is more successful."""

MATERIAL_BASE_LOGIT: dict[str, float] = {"oak": 1.5, "walnut": 0.0, "bamboo": -1.5}
"""Per-material base success logit. Ordered best→worst, so oak should dominate the posterior."""

OBJECT_MASS_RANGE: tuple[float, float] = (0.05, 2.0)
"""Range (kg) of the continuous object_mass factor."""

MASS_WEIGHT: float = 1.5
"""Success-logit gain per unit of normalized mass. Subtracted ⇒ a lighter object is more successful."""

CAMERA_DISTANCE_RANGE: tuple[float, float] = (0.3, 1.5)
"""Range (m) of the continuous camera_distance factor."""

DISTANCE_WEIGHT: float = 1.5
"""Success-logit gain per unit of normalized distance. Subtracted ⇒ a closer camera is more successful."""

OBJECT_TYPE_BASE_LOGIT: dict[str, float] = {"cube": 1.2, "can": 0.0, "mug": -1.2}
"""Per-object-type base success logit. Ordered best→worst, so cube should dominate the posterior."""

_OUTCOME_NAME = "success"
_SLICE = SliceSpec(policy="synthetic", task="SyntheticTask", embodiment="synthetic")


def _normalized(values: torch.Tensor, value_range: tuple[float, float]) -> torch.Tensor:
    """Map values from value_range onto roughly [-1, 1] for the success logit."""
    low, high = value_range
    midpoint = 0.5 * (low + high)
    half_range = 0.5 * (high - low)
    return (values - midpoint) / half_range


def _sample_uniform(value_range: tuple[float, float], num_episodes: int) -> torch.Tensor:
    """Draw num_episodes values uniformly over value_range."""
    low, high = value_range
    return torch.rand(num_episodes) * (high - low) + low


def _sample_success(success_logit: torch.Tensor) -> torch.Tensor:
    """Draw a binary success outcome per episode from Bernoulli(sigmoid(logit))."""
    return torch.bernoulli(torch.sigmoid(success_logit))


def _sample_categorical(
    base_logit: dict[str, float], num_episodes: int
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """Uniformly sample a categorical factor from its base-logit table.

    Returns (choices, per-episode integer codes, per-episode base success logit).
    """
    choices = list(base_logit)
    codes = torch.randint(0, len(choices), (num_episodes,))
    base_logit_per_choice = torch.tensor([base_logit[choice] for choice in choices])
    return choices, codes, base_logit_per_choice[codes]


def _build_dataset(
    continuous: list[tuple[str, tuple[float, float], torch.Tensor]],
    categorical: list[tuple[str, list[str], torch.Tensor]],
    success: torch.Tensor,
) -> SensitivityDataset:
    """Assemble a SensitivityDataset from sampled factors and the binary success outcome.

    continuous: (name, value_range, values) per continuous factor.
    categorical: (name, choices, integer codes) per categorical factor.
    """
    factors = [FactorSpec(name=n, type="continuous", range=[list(r)]) for n, r, _ in continuous]
    factors += [FactorSpec(name=n, type="categorical", choices=c) for n, c, _ in categorical]
    schema = FactorSchema(slice=_SLICE, factors=factors, outcomes=[OutcomeSpec(name=_OUTCOME_NAME, type="bool")])
    # Continuous columns first, then integer-coded categorical columns — the layout
    # SensitivityDataset.factor_columns describes and the estimators expect.
    columns = [values for _, _, values in continuous] + [codes.float() for _, _, codes in categorical]
    return SensitivityDataset(schema, torch.stack(columns, dim=1), success.unsqueeze(1))


def make_continuous_dataset(seed: int, num_episodes: int = 2000) -> SensitivityDataset:
    """Two continuous factors (light_intensity, grasp_offset) driving success.

    Uses the NPE path. Both effects are planted — brighter light and a smaller grasp offset
    raise success — so conditioning the posterior on success should favor high light values
    and low offset values. Two factors keep theta 2-D, away from NPE's 1-D Gaussian fallback.
    """
    torch.manual_seed(seed)
    light_intensity = _sample_uniform(LIGHT_RANGE, num_episodes)
    grasp_offset = _sample_uniform(GRASP_OFFSET_RANGE, num_episodes)
    success = _sample_success(
        LIGHT_WEIGHT * _normalized(light_intensity, LIGHT_RANGE)
        - OFFSET_WEIGHT * _normalized(grasp_offset, GRASP_OFFSET_RANGE)
    )
    return _build_dataset(
        continuous=[
            ("light_intensity", LIGHT_RANGE, light_intensity),
            ("grasp_offset", GRASP_OFFSET_RANGE, grasp_offset),
        ],
        categorical=[],
        success=success,
    )


def make_mixed_dataset(seed: int, num_episodes: int = 2000) -> SensitivityDataset:
    """Continuous light_intensity + categorical table_material driving success.

    Uses the MNPE path. Both effects are planted: brighter light and "better" materials
    (oak > walnut > bamboo) raise success, so conditioning the posterior on success should
    favor high light values and oak.
    """
    torch.manual_seed(seed)
    light_intensity = _sample_uniform(LIGHT_RANGE, num_episodes)
    materials, material_code, material_logit = _sample_categorical(MATERIAL_BASE_LOGIT, num_episodes)
    success = _sample_success(material_logit + LIGHT_WEIGHT * _normalized(light_intensity, LIGHT_RANGE))
    return _build_dataset(
        continuous=[("light_intensity", LIGHT_RANGE, light_intensity)],
        categorical=[("table_material", materials, material_code)],
        success=success,
    )


def make_rich_dataset(seed: int, num_episodes: int = 3000) -> SensitivityDataset:
    """A realistic mix — three continuous + two categorical factors — driving success (MNPE).

    Mirrors the kind of data a real sweep produces: several continuous factors on different
    scales (light, mass, camera distance) and several categoricals (object type, table material).
    Every effect is planted (brighter / lighter / closer / cube / oak raise success), so the
    posterior conditioned on success should recover all of them at once.
    """
    torch.manual_seed(seed)
    light_intensity = _sample_uniform(LIGHT_RANGE, num_episodes)
    object_mass = _sample_uniform(OBJECT_MASS_RANGE, num_episodes)
    camera_distance = _sample_uniform(CAMERA_DISTANCE_RANGE, num_episodes)
    object_types, object_type_code, object_type_logit = _sample_categorical(OBJECT_TYPE_BASE_LOGIT, num_episodes)
    materials, material_code, material_logit = _sample_categorical(MATERIAL_BASE_LOGIT, num_episodes)
    success = _sample_success(
        LIGHT_WEIGHT * _normalized(light_intensity, LIGHT_RANGE)
        - MASS_WEIGHT * _normalized(object_mass, OBJECT_MASS_RANGE)
        - DISTANCE_WEIGHT * _normalized(camera_distance, CAMERA_DISTANCE_RANGE)
        + object_type_logit
        + material_logit
    )
    return _build_dataset(
        continuous=[
            ("light_intensity", LIGHT_RANGE, light_intensity),
            ("object_mass", OBJECT_MASS_RANGE, object_mass),
            ("camera_distance", CAMERA_DISTANCE_RANGE, camera_distance),
        ],
        categorical=[
            ("object_type", object_types, object_type_code),
            ("table_material", materials, material_code),
        ],
        success=success,
    )


def _demo():
    """Run the full pipeline on a synthetic dataset and save the marginals plot.

    Runs the pipeline end to end on generated data: simulate → fit → plot, with no eval
    data needed. Run as::

        python -m isaaclab_arena.analysis.sensitivity.synthetic --kind mixed --output eval/demo.png
    """
    parser = argparse.ArgumentParser(description="Run the sensitivity pipeline on a synthetic dataset and plot it.")
    parser.add_argument(
        "--kind",
        choices=["mixed", "continuous", "rich"],
        default="mixed",
        help="'mixed' (1 cont + 1 cat, MNPE), 'continuous' (2 cont, NPE), or 'rich' (3 cont + 2 cat, MNPE).",
    )
    parser.add_argument(
        "--output",
        default="eval/sensitivity_synthetic.png",
        help="Output figure path; format follows the extension.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=2000)
    args = parser.parse_args()

    builder = {"mixed": make_mixed_dataset, "continuous": make_continuous_dataset, "rich": make_rich_dataset}[args.kind]
    dataset = builder(seed=args.seed, num_episodes=args.num_episodes)
    analyzer = SensitivityAnalyzer(dataset)
    analyzer.fit()
    observation = analyzer.default_observation()
    samples = analyzer.sample_posterior(observation)
    plot_marginals(samples, dataset, observation, output_path=args.output)
    print(f"[INFO] Wrote synthetic sensitivity report → {args.output}")


if __name__ == "__main__":
    _demo()

# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end sensitivity-analysis tests on synthetic data with a known ground truth.

Each test fits a *real* analyzer (via ``make_analyzer``) on a dataset whose factor→outcome
relationship is planted by ``synthetic_sensitivity`` (brighter light and oak raise success),
then asserts the recovered posterior reflects that relationship. The data is built in memory,
so these run on CPU without Isaac Sim. They cover the inference layer the way robolab's
``simple_simulator`` does — not the YAML/JSONL parsing, which has no bearing on inference.
"""

from __future__ import annotations

import numpy as np
import torch

from isaaclab_arena.analysis.sensitivity.empirical_analyzer import KDEAnalyzer
from isaaclab_arena.analysis.sensitivity.factory import make_analyzer
from isaaclab_arena.analysis.sensitivity.posterior_analyzer import MNPEAnalyzer
from isaaclab_arena.tests.utils.synthetic_sensitivity import (
    LIGHT_RANGE,
    MATERIAL_BASE_LOGIT,
    make_continuous_dataset,
    make_mixed_dataset,
)

_OUTCOME = "success"
_SUCCESS = 1.0
_FAILURE = 0.0
_NUM_GRID_POINTS = 200
_NUM_SAMPLES = 5000

_LIGHT_LOW, _LIGHT_HIGH = LIGHT_RANGE
_LIGHT_MIDPOINT = 0.5 * (_LIGHT_LOW + _LIGHT_HIGH)


def _density_weighted_mean(grid: np.ndarray, density: np.ndarray) -> float:
    """Mean light value under a (grid, density) marginal — i.e. where the posterior mass sits."""
    return float(np.sum(grid * density) / np.sum(density))


def test_kde_recovers_brighter_light_drives_success():
    """A single continuous factor: successful episodes should concentrate at high light."""
    dataset = make_continuous_dataset(seed=0)
    analyzer = make_analyzer(dataset, _OUTCOME)
    assert isinstance(analyzer, KDEAnalyzer)

    analyzer.fit()
    grid, success_density = analyzer.continuous_marginal_density("light_intensity", _SUCCESS, _NUM_GRID_POINTS)

    # Ground truth plants brighter ⇒ more successful, so the success posterior must skew well
    # above the midpoint of the light range (a uniform posterior would sit exactly at it).
    assert _density_weighted_mean(grid, success_density) > _LIGHT_MIDPOINT + 0.1 * (_LIGHT_HIGH - _LIGHT_LOW)


def test_mnpe_recovers_light_and_material_effects():
    """Mixed continuous + categorical: recover both the light trend and the material ranking."""
    dataset = make_mixed_dataset(seed=0)
    analyzer = make_analyzer(dataset, _OUTCOME)
    assert isinstance(analyzer, MNPEAnalyzer)

    torch.manual_seed(0)  # make the posterior sampling below reproducible
    analyzer.fit()

    # Continuous effect: light conditioned on success should sit higher than on failure.
    grid, success_density = analyzer.continuous_marginal_density("light_intensity", _SUCCESS, _NUM_GRID_POINTS)
    _, failure_density = analyzer.continuous_marginal_density("light_intensity", _FAILURE, _NUM_GRID_POINTS)
    assert _density_weighted_mean(grid, success_density) > _density_weighted_mean(grid, failure_density)

    # Categorical effect: oak is the planted best material, bamboo the worst.
    materials = list(MATERIAL_BASE_LOGIT)
    oak_index, bamboo_index = materials.index("oak"), materials.index("bamboo")
    success_probs = analyzer.categorical_marginal_probs("table_material", _SUCCESS, _NUM_SAMPLES)
    failure_probs = analyzer.categorical_marginal_probs("table_material", _FAILURE, _NUM_SAMPLES)

    assert success_probs.argmax() == oak_index, f"expected oak most likely given success, got {success_probs}"
    assert success_probs[oak_index] > success_probs[bamboo_index]
    # Conditioning works in both directions: oak is overrepresented among successes,
    # bamboo among failures, relative to the other outcome.
    assert success_probs[oak_index] > failure_probs[oak_index]
    assert failure_probs[bamboo_index] > success_probs[bamboo_index]

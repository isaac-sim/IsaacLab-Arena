# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end sensitivity-analysis tests on synthetic data with a known ground truth.

Each test fits a SensitivityAnalyzer on a dataset whose factor→outcome relationship is
planted by the synthetic module (brighter light, smaller grasp offset, and oak raise
success), then asserts the posterior conditioned on success recovers that relationship. The
data is built in memory, so these run on CPU without Isaac Sim. They cover both estimator
paths: MNPE for mixed schemas, NPE for continuous-only (with 2-D theta).
"""

from __future__ import annotations

import numpy as np
import torch

from isaaclab_arena.analysis.sensitivity.analyzer import SensitivityAnalyzer
from isaaclab_arena.analysis.sensitivity.synthetic import (
    GRASP_OFFSET,
    LIGHT,
    MATERIAL,
    make_continuous_dataset,
    make_mixed_dataset,
)

_NUM_SAMPLES = 5000

_LIGHT_MIDPOINT = 0.5 * (LIGHT.value_range[0] + LIGHT.value_range[1])
_OFFSET_MIDPOINT = 0.5 * (GRASP_OFFSET.value_range[0] + GRASP_OFFSET.value_range[1])


def _factor_samples(analyzer: SensitivityAnalyzer, samples: torch.Tensor, factor_name: str) -> np.ndarray:
    """Pull one factor's column out of a posterior-sample tensor as a 1-D numpy array."""
    return samples[:, analyzer.dataset.factor_columns[factor_name]].squeeze(-1).cpu().numpy()


def test_mnpe_recovers_light_and_material_effects():
    """Mixed continuous + categorical (MNPE): recover the light trend and the material ranking."""
    dataset = make_mixed_dataset(seed=0)
    analyzer = SensitivityAnalyzer(dataset)
    assert analyzer._select_inference_class().__name__ == "MNPE", "mixed schema should select MNPE"

    torch.manual_seed(0)
    analyzer.fit()
    samples = analyzer.sample_posterior(num_samples=_NUM_SAMPLES)  # conditions on success=1 by default

    # Continuous effect: brighter light is planted to raise success.
    assert _factor_samples(analyzer, samples, "light_intensity").mean() > _LIGHT_MIDPOINT

    # Categorical effect: oak is the planted best material, bamboo the worst.
    materials = MATERIAL.choices
    material_codes = np.clip(np.round(_factor_samples(analyzer, samples, "table_material")), 0, len(materials) - 1)
    probabilities = np.bincount(material_codes.astype(int), minlength=len(materials)) / len(material_codes)
    assert materials[int(probabilities.argmax())] == "oak", f"expected oak most likely, got {probabilities}"
    assert probabilities[materials.index("oak")] > probabilities[materials.index("bamboo")]


def test_npe_recovers_two_continuous_effects():
    """Two continuous factors (NPE): recover that bright light and a small grasp offset drive success."""
    dataset = make_continuous_dataset(seed=0)
    analyzer = SensitivityAnalyzer(dataset)
    assert analyzer._select_inference_class().__name__.startswith("NPE"), "continuous-only schema should select NPE"

    torch.manual_seed(0)
    analyzer.fit()
    samples = analyzer.sample_posterior(num_samples=_NUM_SAMPLES)  # conditions on success=1 by default

    # Brighter light raises success → light posterior skews high.
    assert _factor_samples(analyzer, samples, "light_intensity").mean() > _LIGHT_MIDPOINT
    # A smaller grasp offset raises success → offset posterior skews low.
    assert _factor_samples(analyzer, samples, "grasp_offset").mean() < _OFFSET_MIDPOINT

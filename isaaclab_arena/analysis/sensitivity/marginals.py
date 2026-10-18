# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pure (numpy-only) estimators over posterior samples: marginal densities and factor importance.

These functions take already-drawn posterior samples and return summary arrays. They run no
inference and do no plotting, so the static report and the interactive app share one source of
truth for what a marginal and an importance score mean.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from isaaclab_arena.analysis.sensitivity.dataset import FactorSpec, SensitivityDataset

_CONSTANT_STD = 1e-9
"""Below this sample std a continuous posterior is treated as a point mass (KDE is undefined)."""


def continuous_marginal_density(
    factor_samples: np.ndarray, value_range: tuple[float, float], num_grid: int = 200
) -> tuple[np.ndarray, np.ndarray | None]:
    """KDE of a continuous factor's posterior samples on a grid over its swept range.

    Args:
        factor_samples: 1D posterior draws for one continuous factor, in original units.
        value_range: The (low, high) the factor was swept over; the grid spans it.
        num_grid: Number of evenly spaced grid points across the range.

    Returns:
        ``(grid, density)`` where density is the KDE evaluated on grid. density is None when the
        samples are effectively constant (a point mass, where a KDE is undefined) — callers draw
        that as a single line at the sample mean.
    """
    range_low, range_high = value_range
    grid = np.linspace(range_low, range_high, num_grid)
    if float(np.std(factor_samples)) < _CONSTANT_STD:
        return grid, None
    density = gaussian_kde(factor_samples)(grid)
    return grid, density


def categorical_marginal_probs(factor_samples: np.ndarray, num_choices: int) -> np.ndarray:
    """Posterior probability per integer-coded choice of a categorical factor.

    sbi returns categorical columns as floats over the integer-code support, so samples are
    rounded to the nearest code in [0, num_choices - 1] and tallied into frequencies.

    Args:
        factor_samples: 1D posterior draws for one categorical factor (float-coded).
        num_choices: Number of choices the factor can take.

    Returns:
        A length-``num_choices`` array of probabilities summing to 1.
    """
    codes = np.clip(np.round(factor_samples), 0, num_choices - 1).astype(int)
    return np.bincount(codes, minlength=num_choices) / len(codes)


def factor_importance(factor_samples: np.ndarray, factor: FactorSpec) -> float:
    """How far a factor's posterior marginal moved from its uniform prior, in [0, 1].

    The score is the total-variation distance between the posterior marginal and the uniform
    prior the factor was swept from, normalized so 1 means the posterior collapsed onto a single
    value and 0 means it is indistinguishable from the prior (the factor did not affect the
    outcome). Continuous and categorical factors share the [0, 1] scale, so a single ranking
    compares them directly.

    Args:
        factor_samples: 1D posterior draws for this factor, in the factor's own units/codes.
        factor: The factor spec (its type, range, and choices define the prior to compare against).
    """
    if factor.type == "continuous":
        assert factor.range is not None, f"Continuous factor {factor.name!r} has no range."
        grid, density = continuous_marginal_density(factor_samples, factor.range)
        if density is None:
            return 1.0  # a point mass is maximally far from the uniform prior
        span = factor.range[1] - factor.range[0]
        if span <= 0:
            return 0.0
        # Normalize the KDE over the (truncated) range so it integrates to 1 there, then compare
        # it to the uniform prior 1/span. TV = 0.5 * integral|posterior - prior|, already in [0, 1].
        area = np.trapz(density, grid)
        if area <= 0:
            return 0.0
        posterior = density / area
        return float(0.5 * np.trapz(np.abs(posterior - 1.0 / span), grid))

    assert factor.choices is not None, f"Categorical factor {factor.name!r} has no choices."
    num_choices = len(factor.choices)
    probs = categorical_marginal_probs(factor_samples, num_choices)
    total_variation = 0.5 * float(np.sum(np.abs(probs - 1.0 / num_choices)))
    # A categorical TV maxes out at 1 - 1/k (all mass on one choice), so rescale to [0, 1] to keep
    # it comparable with the continuous score.
    ceiling = 1.0 - 1.0 / num_choices
    return total_variation / ceiling if ceiling > 0 else 0.0


def factor_importances(samples: torch.Tensor, dataset: SensitivityDataset) -> list[tuple[str, float]]:
    """Importance score for every factor, sorted most- to least-important.

    A thin dataset-level wrapper over factor_importance: it slices each factor's column out of the
    joint posterior samples and scores it against its prior.

    Args:
        samples: ``(num_samples, num_factors)`` posterior draws in the dataset's factor layout.
        dataset: The dataset, for the factor schema and column layout.

    Returns:
        ``(factor_name, score)`` pairs sorted by descending score; each score is in [0, 1].
    """
    sample_array = samples.cpu().numpy()
    scored = [
        (factor.name, factor_importance(sample_array[:, dataset.factor_columns[factor.name]].squeeze(-1), factor))
        for factor in dataset.factors
    ]
    return sorted(scored, key=lambda name_score: name_score[1], reverse=True)

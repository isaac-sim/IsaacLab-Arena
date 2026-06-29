# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path
from scipy.stats import gaussian_kde
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from isaaclab_arena.analysis.sensitivity.dataset import FactorSpec, SensitivityDataset

_CONTINUOUS_COLOR = "steelblue"
_CATEGORICAL_COLOR = "steelblue"
_PRIOR_COLOR = "grey"


def plot_marginals(
    samples: torch.Tensor,
    dataset: SensitivityDataset,
    observation: torch.Tensor,
    output_path: str | None = None,
):
    """Plot the posterior marginal of every factor in a single figure.

    A pure renderer: it draws already-sampled posterior draws and does not run inference.
    One panel per factor — a density curve for continuous factors, a probability bar chart
    for categorical ones, wrapped into a grid. Panels for components of the same vector
    variation share a y-axis, so their densities compare directly.

    Args:
        samples: ``(num_samples, num_factors)`` posterior draws in the dataset's factor
            layout (continuous-first, original units), e.g. from ``SensitivityAnalyzer.sample_posterior``.
        dataset: The dataset, for the factor schema and column layout.
        observation: The outcome vector the samples were conditioned on (shown in the title).
        output_path: If given, save the figure here. The format follows the path's
            extension (.png, .pdf, …); parent directories are created.

    Returns:
        The matplotlib Figure.
    """
    samples = samples.cpu().numpy()
    factors = dataset.factors
    # Wrap panels into a grid (at most 3 columns) so many factors stay readable.
    num_columns = min(3, len(factors))
    num_rows = math.ceil(len(factors) / num_columns)
    figure, axes = plt.subplots(num_rows, num_columns, figsize=(6.0 * num_columns, 4.5 * num_rows), squeeze=False)
    flat_axes = axes.flatten()
    continuous_axes_by_variation: dict[str, list] = {}
    for axis_index, factor in enumerate(factors):
        ax = flat_axes[axis_index]
        factor_samples = samples[:, dataset.factor_columns[factor.name]].squeeze(-1)
        if factor.type == "continuous":
            _draw_continuous_marginal(ax, factor, factor_samples)
            # Components of one vector variation (name[0], name[1], ...) share a scale.
            variation_name = re.sub(r"\[\d+\]$", "", factor.name)
            continuous_axes_by_variation.setdefault(variation_name, []).append(ax)
        else:
            _draw_categorical_marginal(ax, factor, factor_samples)
        ax.set_title(factor.name, fontsize=11)
    for unused_index in range(len(factors), len(flat_axes)):
        flat_axes[unused_index].axis("off")

    # Give the components of a vector variation a common y-axis so their densities compare directly.
    # A standalone scalar factor keeps its own scale, since unrelated factors can differ in magnitude.
    for grouped_axes in continuous_axes_by_variation.values():
        if len(grouped_axes) < 2:
            continue
        shared_top = max(grouped_ax.get_ylim()[1] for grouped_ax in grouped_axes)
        for grouped_ax in grouped_axes:
            grouped_ax.set_ylim(0, shared_top)

    observation_label = ", ".join(
        f"{name}={value:g}" for name, value in zip(dataset.outcome_names, observation.tolist())
    )
    figure.suptitle(
        f"Posterior marginals — {dataset.num_episodes} episodes  (observed: {observation_label})",
        fontsize=12,
        fontweight="bold",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
    return figure


def _draw_continuous_marginal(ax, factor: FactorSpec, factor_samples: np.ndarray) -> None:
    """Posterior density of a continuous factor over its swept range.

    Draws the KDE of the posterior samples, the uniform prior as a flat reference, and shades
    the central 5-95% of the posterior. Reading the posterior against the prior shows whether
    conditioning on the outcome concentrated the factor, which a mean alone would miss for a
    factor swept symmetrically around its nominal value.
    """
    range_low, range_high = factor.range
    span = range_high - range_low

    if float(np.std(factor_samples)) >= 1e-9:
        grid = np.linspace(range_low, range_high, 200)
        density = gaussian_kde(factor_samples)(grid)
        ax.plot(grid, density, color=_CONTINUOUS_COLOR, linewidth=2, label="posterior")
        ax.fill_between(grid, 0, density, color=_CONTINUOUS_COLOR, alpha=0.2)
        ax.set_ylim(bottom=0)
        low_percentile, high_percentile = np.percentile(factor_samples, [5, 95])
        ax.axvspan(low_percentile, high_percentile, color=_CONTINUOUS_COLOR, alpha=0.15, label="5-95%")
    else:
        ax.axvline(float(np.mean(factor_samples)), color=_CONTINUOUS_COLOR, linewidth=2, label="constant")
        ax.set_ylim(bottom=0)

    if span > 0:
        # The uniform prior is the "no effect" reference the posterior is read against.
        ax.axhline(1.0 / span, color=_PRIOR_COLOR, linestyle="--", linewidth=1.5, label="prior (uniform)")

    ax.set_xlim(range_low, range_high)
    ax.set_xlabel(factor.name)
    ax.set_ylabel("posterior density")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)


def _draw_categorical_marginal(ax, factor: FactorSpec, factor_samples: np.ndarray) -> None:
    """Bar chart of a categorical factor's posterior probability per choice.

    sbi returns categorical columns as floats over the integer-code support, so samples are
    rounded to the nearest code in [0, num_choices - 1] and tallied into frequencies.
    """
    assert factor.choices is not None
    num_choices = len(factor.choices)
    codes = np.clip(np.round(factor_samples), 0, num_choices - 1).astype(int)
    probabilities = np.bincount(codes, minlength=num_choices) / len(codes)

    ax.bar(range(num_choices), probabilities, color=_CATEGORICAL_COLOR, alpha=0.8)
    ax.set_xticks(range(num_choices))
    ax.set_xticklabels(factor.choices, rotation=30, ha="right")
    ax.set_xlabel(factor.name)
    ax.set_ylabel("posterior probability")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, axis="y")

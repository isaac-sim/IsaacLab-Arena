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
    from isaaclab_arena.analysis.sensitivity.empirical import EmpiricalMarginal, EmpiricalSensitivityResult

_CONTINUOUS_COLOR = "steelblue"
_CATEGORICAL_COLOR = "steelblue"
_PRIOR_COLOR = "grey"


def plot_empirical_marginals(
    result: EmpiricalSensitivityResult,
    dataset: SensitivityDataset,
    output_path: str | None = None,
):
    """Plot empirical posterior-to-sampling ratios for every factor.

    Each panel compares the distribution among exactly matching episodes with the distribution
    among all episodes. A value of one means the factor bin appears just as often in both
    populations.

    Args:
        result: Precomputed empirical marginals and bootstrap intervals.
        dataset: The dataset, for factor metadata and outcome labels.
        output_path: If given, save the figure here. The format follows the extension.

    Returns:
        The matplotlib Figure.
    """
    factors_by_name = {factor.name: factor for factor in dataset.factors}
    num_columns = min(3, len(result.marginals))
    num_rows = math.ceil(len(result.marginals) / num_columns)
    figure, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=(6.0 * num_columns, 4.5 * num_rows),
        squeeze=False,
    )

    ratio_axes_by_variation: dict[str, list] = {}
    flat_axes = axes.flatten()
    for marginal_index, marginal in enumerate(result.marginals):
        ratio_axis = flat_axes[marginal_index]
        factor = factors_by_name[marginal.factor_name]

        if factor.type == "continuous":
            _draw_empirical_continuous_marginal(ratio_axis, marginal)
            variation_name = re.sub(r"\[\d+\]$", "", factor.name)
            ratio_axes_by_variation.setdefault(variation_name, []).append(ratio_axis)
        else:
            _draw_empirical_categorical_marginal(ratio_axis, marginal)

        ratio_axis.set_title(factor.name, fontsize=11)
        ratio_axis.set_ylabel("posterior / sampled")

    for unused_index in range(len(result.marginals), len(flat_axes)):
        flat_axes[unused_index].axis("off")

    for grouped_axes in ratio_axes_by_variation.values():
        if len(grouped_axes) < 2:
            continue
        shared_top = max(grouped_axis.get_ylim()[1] for grouped_axis in grouped_axes)
        for grouped_axis in grouped_axes:
            grouped_axis.set_ylim(0, shared_top)

    observation_label = ", ".join(
        f"{name}={value:g}" for name, value in zip(dataset.outcome_names, result.observation.tolist())
    )
    confidence_percent = 100 * result.confidence_level
    figure.suptitle(
        f"Empirical sensitivity — {result.num_matching_episodes} of {result.num_episodes} episodes match\n"
        f"Observed: {observation_label}  ({confidence_percent:g}% bootstrap intervals)",
        fontsize=12,
        fontweight="bold",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.93])

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
    return figure


def _draw_empirical_continuous_marginal(
    ratio_axis,
    marginal: EmpiricalMarginal,
) -> None:
    """Draw fixed-bin empirical results for one continuous factor."""
    assert marginal.bin_edges is not None
    bin_edges = marginal.bin_edges
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = np.diff(bin_edges)
    bar_widths = 0.88 * bin_widths

    ratio_axis.bar(
        bin_centers,
        marginal.posterior_to_sampling_ratio,
        width=bar_widths,
        color=_CONTINUOUS_COLOR,
        alpha=0.8,
    )
    _draw_confidence_intervals(
        ratio_axis,
        bin_centers,
        marginal.posterior_to_sampling_ratio,
        marginal.ratio_confidence_low,
        marginal.ratio_confidence_high,
    )
    ratio_axis.axhline(1.0, color=_PRIOR_COLOR, linestyle="--", linewidth=1.5, label="no association")
    ratio_axis.set_xlim(bin_edges[0], bin_edges[-1])
    ratio_axis.set_ylim(bottom=0)
    ratio_axis.legend(loc="best", fontsize=9)
    ratio_axis.grid(alpha=0.3, axis="y")


def _draw_empirical_categorical_marginal(
    ratio_axis,
    marginal: EmpiricalMarginal,
) -> None:
    """Draw empirical results for one categorical factor."""
    assert marginal.bin_labels is not None
    category_positions = np.arange(len(marginal.bin_labels))
    ratio_axis.bar(category_positions, marginal.posterior_to_sampling_ratio, color=_CATEGORICAL_COLOR, alpha=0.8)
    _draw_confidence_intervals(
        ratio_axis,
        category_positions,
        marginal.posterior_to_sampling_ratio,
        marginal.ratio_confidence_low,
        marginal.ratio_confidence_high,
    )
    ratio_axis.axhline(1.0, color=_PRIOR_COLOR, linestyle="--", linewidth=1.5, label="no association")
    ratio_axis.set_ylim(bottom=0)
    ratio_axis.set_xticks(category_positions)
    ratio_axis.set_xticklabels(marginal.bin_labels, rotation=30, ha="right")
    ratio_axis.legend(loc="best", fontsize=9)
    ratio_axis.grid(alpha=0.3, axis="y")


def _draw_confidence_intervals(axis, positions, values, interval_low, interval_high) -> None:
    """Draw asymmetric confidence intervals for finite point estimates."""
    finite = np.isfinite(values) & np.isfinite(interval_low) & np.isfinite(interval_high)
    if not np.any(finite):
        return
    lower_errors = np.maximum(0.0, values[finite] - interval_low[finite])
    upper_errors = np.maximum(0.0, interval_high[finite] - values[finite])
    axis.errorbar(
        np.asarray(positions)[finite],
        values[finite],
        yerr=np.vstack([lower_errors, upper_errors]),
        fmt="none",
        ecolor="dimgray",
        elinewidth=0.9,
        alpha=0.75,
        capsize=3,
        capthick=0.9,
    )


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

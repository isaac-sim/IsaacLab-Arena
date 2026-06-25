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
from typing import TYPE_CHECKING

from isaaclab_arena.analysis.sensitivity.marginals import (
    categorical_marginal_probs,
    continuous_marginal_density,
    factor_importances,
)

if TYPE_CHECKING:
    import torch

    from isaaclab_arena.analysis.sensitivity.dataset import FactorSpec, SensitivityDataset

_CONTINUOUS_COLOR = "steelblue"
_CATEGORICAL_COLOR = "steelblue"
_PRIOR_COLOR = "grey"
_IMPORTANT_COLOR = "indianred"


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


def _draw_continuous_marginal(ax, factor: FactorSpec, factor_samples: np.ndarray, compact: bool = False) -> None:
    """Posterior density of a continuous factor over its swept range.

    Draws the KDE of the posterior samples, the uniform prior as a flat reference, and shades
    the central 5-95% of the posterior. Reading the posterior against the prior shows whether
    conditioning on the outcome concentrated the factor, which a mean alone would miss for a
    factor swept symmetrically around its nominal value.

    Args:
        ax: Axis to draw on.
        factor: The continuous factor (its range bounds the x-axis and the prior).
        factor_samples: 1D posterior draws for this factor.
        compact: Drop the legend and axis labels, for use as a corner-plot diagonal where edge
            axes carry the labels.
    """
    range_low, range_high = factor.range
    span = range_high - range_low

    grid, density = continuous_marginal_density(factor_samples, factor.range)
    if density is not None:
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
    ax.grid(alpha=0.3)
    if compact:
        return
    ax.set_xlabel(factor.name)
    ax.set_ylabel("posterior density")
    ax.legend(loc="best", fontsize=9)


def _draw_categorical_marginal(ax, factor: FactorSpec, factor_samples: np.ndarray, compact: bool = False) -> None:
    """Bar chart of a categorical factor's posterior probability per choice.

    sbi returns categorical columns as floats over the integer-code support, so samples are
    rounded to the nearest code in [0, num_choices - 1] and tallied into frequencies.

    Args:
        ax: Axis to draw on.
        factor: The categorical factor (its choices label the bars).
        factor_samples: 1D posterior draws for this factor (float-coded).
        compact: Drop the axis labels, for use as a corner-plot diagonal.
    """
    assert factor.choices is not None
    num_choices = len(factor.choices)
    probabilities = categorical_marginal_probs(factor_samples, num_choices)

    ax.bar(range(num_choices), probabilities, color=_CATEGORICAL_COLOR, alpha=0.8)
    ax.set_xticks(range(num_choices))
    ax.set_xticklabels(factor.choices, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, axis="y")
    if compact:
        return
    ax.set_xlabel(factor.name)
    ax.set_ylabel("posterior probability")


def plot_marginal(
    samples: torch.Tensor,
    dataset: SensitivityDataset,
    factor_name: str,
    observation: torch.Tensor,
    output_path: str | None = None,
):
    """Posterior marginal of a single named factor, on its own figure.

    The one-panel counterpart to plot_marginals, for drawing one factor from an arbitrary set of
    draws — e.g. a conditioned subset, where samples is already sliced to a pinned region.

    Args:
        samples: ``(num_samples, num_factors)`` posterior draws in the dataset's factor layout
            (may be a conditioned subset).
        dataset: The dataset, for the factor schema and column layout.
        factor_name: Name of the factor to draw.
        observation: The outcome vector the samples were conditioned on (shown in the title).
        output_path: If given, save the figure here; the format follows the path's extension.

    Returns:
        The matplotlib Figure.
    """
    sample_array = samples.cpu().numpy()
    factor = {factor.name: factor for factor in dataset.factors}[factor_name]
    factor_samples = sample_array[:, dataset.factor_columns[factor_name]].squeeze(-1)

    figure, ax = plt.subplots(figsize=(5.0, 3.5))
    if factor.type == "continuous":
        _draw_continuous_marginal(ax, factor, factor_samples)
    else:
        _draw_categorical_marginal(ax, factor, factor_samples)

    observation_label = ", ".join(
        f"{name}={value:g}" for name, value in zip(dataset.outcome_names, observation.tolist())
    )
    ax.set_title(f"{factor_name}  (observed: {observation_label}; n={len(factor_samples)})", fontsize=11)
    figure.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
    return figure


def plot_joint(
    samples: torch.Tensor,
    dataset: SensitivityDataset,
    factor_x_name: str,
    factor_y_name: str,
    observation: torch.Tensor,
    output_path: str | None = None,
):
    """Single-pair joint posterior of two named factors, on its own figure.

    The one-cell counterpart to plot_corner, for picking an interaction to look at (e.g. the
    interactive app's factor-pair selector) without rendering the whole grid.

    Args:
        samples: ``(num_samples, num_factors)`` posterior draws in the dataset's factor layout.
        dataset: The dataset, for the factor schema and column layout.
        factor_x_name: Name of the factor on the horizontal axis.
        factor_y_name: Name of the factor on the vertical axis.
        observation: The outcome vector the samples were conditioned on (shown in the title).
        output_path: If given, save the figure here; the format follows the path's extension.

    Returns:
        The matplotlib Figure.
    """
    sample_array = samples.cpu().numpy()
    factors_by_name = {factor.name: factor for factor in dataset.factors}
    factor_x, factor_y = factors_by_name[factor_x_name], factors_by_name[factor_y_name]
    columns = dataset.factor_columns
    samples_x = sample_array[:, columns[factor_x_name]].squeeze(-1)
    samples_y = sample_array[:, columns[factor_y_name]].squeeze(-1)

    figure, ax = plt.subplots(figsize=(6.0, 5.0))
    _draw_joint(ax, factor_x, factor_y, samples_x, samples_y)
    ax.set_xlabel(factor_x_name)
    ax.set_ylabel(factor_y_name)

    observation_label = ", ".join(
        f"{name}={value:g}" for name, value in zip(dataset.outcome_names, observation.tolist())
    )
    ax.set_title(f"Joint posterior  (observed: {observation_label})", fontsize=12, fontweight="bold")
    figure.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
    return figure


def plot_corner(
    samples: torch.Tensor,
    dataset: SensitivityDataset,
    observation: torch.Tensor,
    output_path: str | None = None,
):
    """Corner plot: per-factor marginals on the diagonal, pairwise joints below it.

    A pure renderer over already-sampled posterior draws. The diagonal repeats the 1D marginals;
    each lower-triangle cell is the joint posterior of its column-factor (x) and row-factor (y),
    which is where factor interactions show up (e.g. the outcome needing low light *and* a small
    camera offset together) — something the 1D marginals cannot reveal.

    Args:
        samples: ``(num_samples, num_factors)`` posterior draws in the dataset's factor layout.
        dataset: The dataset, for the factor schema and column layout.
        observation: The outcome vector the samples were conditioned on (shown in the title).
        output_path: If given, save the figure here. The format follows the path's extension
            (.png, .pdf, …); parent directories are created.

    Returns:
        The matplotlib Figure.
    """
    sample_array = samples.cpu().numpy()
    factors = dataset.factors
    num_factors = len(factors)
    columns = dataset.factor_columns

    def factor_samples(factor: FactorSpec) -> np.ndarray:
        return sample_array[:, columns[factor.name]].squeeze(-1)

    figure, axes = plt.subplots(num_factors, num_factors, figsize=(3.2 * num_factors, 3.2 * num_factors), squeeze=False)
    for row in range(num_factors):
        for col in range(num_factors):
            ax = axes[row][col]
            if col > row:
                ax.axis("off")  # upper triangle is the mirror of the lower, so leave it blank
                continue
            factor_x, factor_y = factors[col], factors[row]
            if row == col:
                if factor_x.type == "continuous":
                    _draw_continuous_marginal(ax, factor_x, factor_samples(factor_x), compact=True)
                else:
                    _draw_categorical_marginal(ax, factor_x, factor_samples(factor_x), compact=True)
            else:
                _draw_joint(ax, factor_x, factor_y, factor_samples(factor_x), factor_samples(factor_y))
            # Label only the edge axes (bottom row gets x names, left column gets y names).
            if row == num_factors - 1:
                ax.set_xlabel(factor_x.name, fontsize=9)
            if col == 0:
                ax.set_ylabel(factor_y.name, fontsize=9)

    observation_label = ", ".join(
        f"{name}={value:g}" for name, value in zip(dataset.outcome_names, observation.tolist())
    )
    figure.suptitle(
        f"Posterior corner plot — {dataset.num_episodes} episodes  (observed: {observation_label})",
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.97])

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
    return figure


def _draw_joint(ax, factor_x: FactorSpec, factor_y: FactorSpec, samples_x: np.ndarray, samples_y: np.ndarray) -> None:
    """Joint posterior of two factors, dispatched by their type combination.

    continuous × continuous draws a 2D-histogram heatmap; continuous × categorical draws a violin
    of the continuous factor per category; categorical × categorical draws a joint-probability
    heatmap. factor_x is the horizontal axis and factor_y the vertical one.
    """
    x_continuous = factor_x.type == "continuous"
    y_continuous = factor_y.type == "continuous"
    if x_continuous and y_continuous:
        ax.hist2d(samples_x, samples_y, bins=40, range=[factor_x.range, factor_y.range], cmap="Blues")
        ax.set_xlim(*factor_x.range)
        ax.set_ylim(*factor_y.range)
    elif x_continuous and not y_continuous:
        _draw_violin_per_category(ax, factor_y, samples_y, factor_x, samples_x, continuous_axis="x")
    elif not x_continuous and y_continuous:
        _draw_violin_per_category(ax, factor_x, samples_x, factor_y, samples_y, continuous_axis="y")
    else:
        _draw_categorical_heatmap(ax, factor_x, samples_x, factor_y, samples_y)
    ax.grid(alpha=0.3)


def _draw_violin_per_category(
    ax,
    categorical_factor: FactorSpec,
    categorical_samples: np.ndarray,
    continuous_factor: FactorSpec,
    continuous_samples: np.ndarray,
    continuous_axis: str,
) -> None:
    """Violins of a continuous factor split by a categorical factor's choices.

    Each violin is the continuous factor's distribution among the posterior draws taking one
    categorical choice. continuous_axis ("x" or "y") says which axis the continuous factor runs
    along, so the same routine serves both lower-triangle orientations.
    """
    assert categorical_factor.choices is not None
    num_choices = len(categorical_factor.choices)
    codes = np.clip(np.round(categorical_samples), 0, num_choices - 1).astype(int)
    grouped = [continuous_samples[codes == choice] for choice in range(num_choices)]
    # violinplot rejects empty groups, so drop choices no posterior draw landed on.
    positions = [choice for choice, values in enumerate(grouped) if len(values) > 0]
    populated = [values for values in grouped if len(values) > 0]
    if not populated:
        return
    vertical = continuous_axis == "y"
    ax.violinplot(populated, positions=positions, vert=vertical, showmeans=True, widths=0.8)
    if vertical:
        ax.set_xticks(range(num_choices))
        ax.set_xticklabels(categorical_factor.choices, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(*continuous_factor.range)
    else:
        ax.set_yticks(range(num_choices))
        ax.set_yticklabels(categorical_factor.choices, fontsize=8)
        ax.set_xlim(*continuous_factor.range)


def _draw_categorical_heatmap(
    ax, factor_x: FactorSpec, samples_x: np.ndarray, factor_y: FactorSpec, samples_y: np.ndarray
) -> None:
    """Heatmap of the joint posterior probability over two categorical factors' choices."""
    assert factor_x.choices is not None and factor_y.choices is not None
    num_x, num_y = len(factor_x.choices), len(factor_y.choices)
    codes_x = np.clip(np.round(samples_x), 0, num_x - 1).astype(int)
    codes_y = np.clip(np.round(samples_y), 0, num_y - 1).astype(int)
    # joint[row=y, col=x] = P(x=col, y=row); origin="lower" puts code 0 at the bottom-left.
    joint = np.zeros((num_y, num_x))
    np.add.at(joint, (codes_y, codes_x), 1.0)
    joint /= joint.sum()
    ax.imshow(joint, origin="lower", aspect="auto", cmap="Blues", extent=(-0.5, num_x - 0.5, -0.5, num_y - 0.5))
    ax.set_xticks(range(num_x))
    ax.set_xticklabels(factor_x.choices, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(num_y))
    ax.set_yticklabels(factor_y.choices, fontsize=8)


def plot_importance(
    samples: torch.Tensor,
    dataset: SensitivityDataset,
    observation: torch.Tensor,
    output_path: str | None = None,
):
    """Tornado chart ranking factors by how far their posterior moved from the prior.

    A pure renderer over already-sampled posterior draws. Each bar is a factor's importance score
    in [0, 1] (see marginals.factor_importance): near 0 the factor barely affected the outcome,
    near 1 the outcome pins it down. Bars are sorted most- to least-important, the most important
    on top, giving the report a single "what matters" summary across continuous and categorical
    factors alike.

    Args:
        samples: ``(num_samples, num_factors)`` posterior draws in the dataset's factor layout.
        dataset: The dataset, for the factor schema and column layout.
        observation: The outcome vector the samples were conditioned on (shown in the title).
        output_path: If given, save the figure here. The format follows the path's extension
            (.png, .pdf, …); parent directories are created.

    Returns:
        The matplotlib Figure.
    """
    scored = factor_importances(samples, dataset)
    names = [name for name, _ in scored]
    scores = [score for _, score in scored]

    # barh fills bottom-up, so reverse to put the most important factor on top.
    positions = range(len(names))
    figure, ax = plt.subplots(figsize=(8.0, max(2.0, 0.5 * len(names) + 1.0)))
    ax.barh(positions, scores[::-1], color=_IMPORTANT_COLOR, alpha=0.85)
    ax.set_yticks(positions)
    ax.set_yticklabels(names[::-1])
    ax.set_xlim(0, 1)
    ax.set_xlabel("importance (total-variation distance from prior, 0–1)")
    ax.grid(alpha=0.3, axis="x")

    observation_label = ", ".join(
        f"{name}={value:g}" for name, value in zip(dataset.outcome_names, observation.tolist())
    )
    ax.set_title(
        f"Factor importance — {dataset.num_episodes} episodes  (observed: {observation_label})",
        fontsize=12,
        fontweight="bold",
    )
    figure.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
    return figure

# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.stats import gaussian_kde
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from isaaclab_arena.analysis.sensitivity.dataset import FactorSpec, SensitivityDataset

_CONTINUOUS_COLOR = "steelblue"
_CATEGORICAL_COLOR = "steelblue"
_MEAN_COLOR = "firebrick"


def plot_marginals(
    samples: torch.Tensor,
    dataset: SensitivityDataset,
    observation: torch.Tensor,
    output_path: str | None = None,
):
    """Plot the posterior marginal of every factor in a single figure.

    A pure renderer: it draws already-sampled posterior draws and does not run inference.
    One panel per factor — a density curve for continuous factors, a probability bar chart
    for categorical ones, wrapped into a grid.

    Args:
        samples: ``(num_samples, total_factor_dim)`` posterior draws in the dataset's factor
            layout (continuous-first, original units), e.g. from ``SensitivityAnalyzer.sample_posterior``.
        dataset: The dataset, for the factor schema and column layout.
        observation: The outcome vector the samples were conditioned on (shown in the title).
        output_path: If given, save the figure here. The format follows the path's
            extension (.png, .pdf, …); parent directories are created.

    Returns:
        The matplotlib Figure.
    """
    samples = samples.cpu().numpy()
    factors = dataset.schema.factors
    # Wrap panels into a grid (at most 3 columns) so many factors stay readable.
    num_columns = min(3, len(factors))
    num_rows = math.ceil(len(factors) / num_columns)
    figure, axes = plt.subplots(num_rows, num_columns, figsize=(6.0 * num_columns, 4.5 * num_rows), squeeze=False)
    flat_axes = axes.flatten()
    for axis_index, factor in enumerate(factors):
        ax = flat_axes[axis_index]
        factor_samples = samples[:, dataset.factor_columns[factor.name]].squeeze(-1)
        if factor.type == "continuous":
            _draw_continuous_marginal(ax, factor, factor_samples)
        else:
            _draw_categorical_marginal(ax, factor, factor_samples)
        ax.set_title(factor.name, fontsize=11)
    for unused_index in range(len(factors), len(flat_axes)):
        flat_axes[unused_index].axis("off")

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
    """Smooth posterior density (filled KDE curve) of a continuous factor, with a mean line.

    A KDE line over the posterior samples reads the shape of a continuous posterior better
    than a binned histogram. Falls back to a single line at the mean when the samples have
    no spread (KDE bandwidth is then undefined).
    """
    range_low, range_high = factor.range[0]
    sample_mean = float(np.mean(factor_samples))
    if float(np.std(factor_samples)) >= 1e-9:
        grid = np.linspace(range_low, range_high, 200)
        density = gaussian_kde(factor_samples)(grid)
        ax.plot(grid, density, color=_CONTINUOUS_COLOR, linewidth=2)
        ax.fill_between(grid, 0, density, color=_CONTINUOUS_COLOR, alpha=0.2)
        ax.set_ylim(bottom=0)
    ax.axvline(sample_mean, color=_MEAN_COLOR, linestyle="--", linewidth=2, label=f"mean = {sample_mean:.3g}")
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

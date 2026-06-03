# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Plot renderers for sensitivity analysis.

Pure-visualization module. Calls into the analyzer's public posterior queries
(``continuous_marginal_density`` and ``categorical_marginal_probs``) and renders matplotlib
figures. Decoupled from the analyzer hierarchy so new plot types can be added without
touching inference code, and so existing plot code can be tested with mock posteriors.

The single entry point is ``plot_marginal(analyzer, factor_name, output_path, ...)``,
which dispatches by factor type to the right renderer.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_arena.analysis.sensitivity.analyzer import BaseAnalyzer
    from isaaclab_arena.analysis.sensitivity.dataset import FactorSpec


def plot_marginal(
    analyzer: BaseAnalyzer,
    factor_name: str,
    output_path,
    outcome_value: float = 1.0,
    num_samples: int = 10_000,
    num_grid_points: int = 200,
) -> None:
    """Render the marginal posterior for ``factor_name``, dispatching by factor type.

    For continuous factors, the analyzer must expose ``continuous_marginal_density``
    (only ``PosteriorAnalyzer`` does — ``EmpiricalAnalyzer`` rejects continuous factors at
    construction time, so this branch isn't reachable through ``make_analyzer``).
    """
    factor_spec = analyzer._factor_spec(factor_name)
    if factor_spec.type == "continuous":
        if not hasattr(analyzer, "continuous_marginal_density"):
            raise NotImplementedError(
                f"{type(analyzer).__name__} cannot plot continuous factors; expected a PosteriorAnalyzer (NPE/MNPE)."
            )
        _plot_continuous_marginal(analyzer, factor_spec, output_path, outcome_value, num_grid_points)
    elif factor_spec.type == "categorical":
        _plot_categorical_marginal(analyzer, factor_spec, output_path, outcome_value, num_samples)
    else:
        raise NotImplementedError(f"Unsupported factor type {factor_spec.type!r}")


def _plot_continuous_marginal(
    analyzer: BaseAnalyzer,
    factor_spec: FactorSpec,
    output_path,
    outcome_value: float,
    num_grid_points: int,
) -> None:
    """Render a continuous factor's marginal posterior as a density curve.

    The blue curve shows ``P(factor_value | outcome=outcome_value)`` from the analyzer.
    Below the x-axis is an empirical "rug" — small vertical ticks at the actual recorded
    theta values, coloured green for episodes where the outcome was achieved (``≥ 0.5``)
    and red for episodes where it was not. The rug lets a human eyeball whether the
    smooth posterior actually agrees with where the successful episodes lived.
    """
    import matplotlib.pyplot as plt

    grid, density = analyzer.continuous_marginal_density(factor_spec.name, outcome_value, num_grid_points)
    factor_column_slice = analyzer.dataset.factor_columns[factor_spec.name]
    outcome_column_index = analyzer.dataset.outcome_columns[analyzer.outcome_name]
    empirical_theta_values = analyzer.dataset.theta[:, factor_column_slice].squeeze(-1).cpu().numpy()
    # For log_uniform factors the dataset stores log10(theta); the rug ticks need to be at
    # the actual intensity values to align with the linear-scale grid returned above.
    if factor_spec.distribution == "log_uniform":
        empirical_theta_values = np.power(10.0, empirical_theta_values)
    empirical_outcomes = analyzer.dataset.x[:, outcome_column_index].cpu().numpy()

    figure, axes = plt.subplots(figsize=(8, 5))
    axes.plot(
        grid,
        density,
        color="steelblue",
        linewidth=2,
        label=f"P({factor_spec.name} | {analyzer.outcome_name}={outcome_value:g})",
    )
    axes.fill_between(grid, 0, density, color="steelblue", alpha=0.2)

    # Rug coloring depends on outcome shape. For binary outcomes (only 0/1 observed) the
    # green/red ≥/<0.5 split gives a meaningful "successes vs failures" picture. For
    # continuous outcomes (e.g. task_duration) the same threshold is nonsensical (every
    # sample is ≥ 0.5), so we drop the split and just show all samples as one neutral rug.
    is_binary_outcome = set(empirical_outcomes.flatten().tolist()).issubset({0.0, 1.0})
    if is_binary_outcome:
        success_mask = empirical_outcomes >= 0.5
        axes.scatter(
            empirical_theta_values[success_mask],
            np.full(success_mask.sum(), -0.05 * density.max()),
            marker="|",
            color="seagreen",
            s=80,
            label=f"{analyzer.outcome_name} ≥ 0.5  (n={success_mask.sum()})",
        )
        axes.scatter(
            empirical_theta_values[~success_mask],
            np.full((~success_mask).sum(), -0.1 * density.max()),
            marker="|",
            color="firebrick",
            s=80,
            label=f"{analyzer.outcome_name} < 0.5  (n={(~success_mask).sum()})",
        )
    else:
        axes.scatter(
            empirical_theta_values,
            np.full(len(empirical_theta_values), -0.05 * density.max()),
            marker="|",
            color="slategray",
            s=80,
            label=f"observed samples  (n={len(empirical_theta_values)})",
        )
    axes.set_xlabel(factor_spec.name)
    axes.set_ylabel("posterior density")
    axes.set_title(_plot_title(analyzer, factor_spec.name))
    if factor_spec.distribution == "log_uniform":
        axes.set_xscale("log")
    axes.legend(loc="best", fontsize=9)
    axes.grid(alpha=0.3)
    figure.tight_layout()
    _save_figure(figure, output_path)


def _plot_categorical_marginal(
    analyzer: BaseAnalyzer,
    factor_spec: FactorSpec,
    output_path,
    outcome_value: float,
    num_samples: int,
) -> None:
    """Render a categorical factor's marginal as side-by-side bars per category.

    The blue bar (left of each category) is the analyzer's ``P(category | outcome)``.
    The green bar (right of each category) is the *empirical* per-category outcome rate
    — independent of the analyzer's posterior, computed directly from the raw data.
    For the ``EmpiricalAnalyzer`` the two will agree exactly (up to normalization); for
    a posterior-based analyzer they may differ slightly if the model smooths.

    Each green bar is annotated with the sample count ``n`` for that category, so the
    user can see how trustworthy each bar is.
    """
    import matplotlib.pyplot as plt

    assert factor_spec.choices is not None
    choices = factor_spec.choices
    num_choices = len(choices)
    factor_column_slice = analyzer.dataset.factor_columns[factor_spec.name]
    outcome_column_index = analyzer.dataset.outcome_columns[analyzer.outcome_name]

    # Posterior probs come from the analyzer; empirical rate and counts are raw data,
    # rendered alongside as a sanity reference.
    posterior_probabilities = analyzer.categorical_marginal_probs(factor_spec.name, outcome_value, num_samples)

    empirical_theta_codes = analyzer.dataset.theta[:, factor_column_slice].squeeze(-1).long().cpu().numpy()
    empirical_outcomes = analyzer.dataset.x[:, outcome_column_index].cpu().numpy()
    empirical_rates = np.zeros(num_choices)
    empirical_counts = np.zeros(num_choices, dtype=int)
    for code in range(num_choices):
        category_mask = empirical_theta_codes == code
        empirical_counts[code] = int(category_mask.sum())
        if category_mask.any():
            empirical_rates[code] = float((empirical_outcomes[category_mask] >= 0.5).mean())

    figure, axes = plt.subplots(figsize=(max(8, 1.0 * num_choices), 5))
    bar_x_positions = np.arange(num_choices)
    bar_width = 0.4
    axes.bar(
        bar_x_positions - bar_width / 2,
        posterior_probabilities,
        bar_width,
        color="steelblue",
        alpha=0.8,
        label=f"P(category | {analyzer.outcome_name}={outcome_value:g})",
    )
    axes.bar(
        bar_x_positions + bar_width / 2,
        empirical_rates,
        bar_width,
        color="seagreen",
        alpha=0.7,
        label=f"empirical {analyzer.outcome_name} rate per category",
    )
    for category_index, count in enumerate(empirical_counts):
        axes.text(
            category_index + bar_width / 2,
            empirical_rates[category_index] + 0.02,
            f"n={count}",
            ha="center",
            fontsize=8,
        )

    axes.set_xticks(bar_x_positions)
    axes.set_xticklabels(choices, rotation=30, ha="right")
    axes.set_ylabel("probability")
    axes.set_ylim(0, 1.05)
    axes.set_title(_plot_title(analyzer, factor_spec.name))
    axes.legend(loc="best", fontsize=9)
    axes.grid(alpha=0.3, axis="y")
    figure.tight_layout()
    _save_figure(figure, output_path)


def _plot_title(analyzer: BaseAnalyzer, factor_name: str) -> str:
    """Format the plot title as ``"Sensitivity of <outcome> to <factor>" / slice block``."""
    return (
        f"Sensitivity of {analyzer.outcome_name} to {factor_name}\n"
        f"slice: {analyzer.dataset.schema.slice.policy} / "
        f"{analyzer.dataset.schema.slice.task} / {analyzer.dataset.schema.slice.embodiment}"
    )


def _save_figure(figure, destination) -> None:
    """Save a matplotlib figure to ``destination`` (a path or a writable file-like object).

    Accepts either a filesystem path (``str`` / ``Path``) or any seekable file-like buffer
    (e.g. ``io.BytesIO``). Paths get parent-dir creation; buffers are written to directly.
    The figure is closed after save regardless of destination type.
    """
    import matplotlib.pyplot as plt

    if isinstance(destination, (str, Path)):
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=150)
    else:
        figure.savefig(destination, dpi=150, format="png")
    plt.close(figure)

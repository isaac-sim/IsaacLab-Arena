# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_arena.analysis.sensitivity.analyzer_base import BaseAnalyzer
    from isaaclab_arena.analysis.sensitivity.dataset import FactorSpec


def draw_marginal(
    ax,
    analyzer: BaseAnalyzer,
    factor_name: str,
    outcome_value: float = 1.0,
    num_samples: int = 10_000,
    num_grid_points: int = 200,
) -> None:
    """Draw ``factor_name``'s marginal posterior onto ``ax``, dispatching by factor type.

    Sets axis labels, scale, legend and grid but NOT the title — the caller titles the
    Axes (a standalone plot wants the full slice block; a grid cell wants a compact label).

    For continuous factors, the analyzer must expose ``continuous_marginal_density``
    (``PosteriorAnalyzer`` and ``KDEAnalyzer`` do); the categorical-only analyzers reject
    continuous factors at construction time, so they never reach this branch.
    """
    factor_spec = analyzer._factor_spec(factor_name)
    if factor_spec.type == "continuous":
        if not hasattr(analyzer, "continuous_marginal_density"):
            raise NotImplementedError(
                f"{type(analyzer).__name__} cannot plot continuous factors; expected a PosteriorAnalyzer (NPE/MNPE)."
            )
        _draw_continuous_marginal(ax, analyzer, factor_spec, outcome_value, num_grid_points)
    elif factor_spec.type == "categorical":
        _draw_categorical_marginal(ax, analyzer, factor_spec, outcome_value, num_samples)
    else:
        raise NotImplementedError(f"Unsupported factor type {factor_spec.type!r}")


def _draw_continuous_marginal(
    ax,
    analyzer: BaseAnalyzer,
    factor_spec: FactorSpec,
    outcome_value: float,
    num_grid_points: int,
) -> None:
    """Draw a continuous factor's marginal posterior onto ``ax`` as a density curve.

    The blue curve shows ``P(factor_value | outcome=outcome_value)`` from the analyzer.
    Below the x-axis is an empirical "rug" — small vertical ticks at the actual recorded
    theta values, coloured green for episodes where the outcome was achieved (``≥ 0.5``)
    and red for episodes where it was not. The rug lets a human eyeball whether the
    smooth posterior actually agrees with where the successful episodes lived.
    """
    grid, density = analyzer.continuous_marginal_density(factor_spec.name, outcome_value, num_grid_points)
    factor_column_slice = analyzer.dataset.factor_columns[factor_spec.name]
    outcome_column_index = analyzer.dataset.outcome_columns[analyzer.outcome_name]
    empirical_theta_values = analyzer.dataset.theta[:, factor_column_slice].squeeze(-1).cpu().numpy()
    empirical_outcomes = analyzer.dataset.x[:, outcome_column_index].cpu().numpy()

    ax.plot(
        grid,
        density,
        color="steelblue",
        linewidth=2,
        label=f"P({factor_spec.name} | {analyzer.outcome_name}={outcome_value:g})",
    )
    ax.fill_between(grid, 0, density, color="steelblue", alpha=0.2)

    # Binary outcomes: split the rug green/red at 0.5 (successes vs failures). Continuous
    # outcomes (e.g. task_duration): the threshold is meaningless, so show one neutral rug.
    is_binary_outcome = set(empirical_outcomes.flatten().tolist()).issubset({0.0, 1.0})
    if is_binary_outcome:
        success_mask = empirical_outcomes >= 0.5
        ax.scatter(
            empirical_theta_values[success_mask],
            np.full(success_mask.sum(), -0.05 * density.max()),
            marker="|",
            color="seagreen",
            s=80,
            label=f"{analyzer.outcome_name} ≥ 0.5  (n={success_mask.sum()})",
        )
        ax.scatter(
            empirical_theta_values[~success_mask],
            np.full((~success_mask).sum(), -0.1 * density.max()),
            marker="|",
            color="firebrick",
            s=80,
            label=f"{analyzer.outcome_name} < 0.5  (n={(~success_mask).sum()})",
        )
    else:
        ax.scatter(
            empirical_theta_values,
            np.full(len(empirical_theta_values), -0.05 * density.max()),
            marker="|",
            color="slategray",
            s=80,
            label=f"observed samples  (n={len(empirical_theta_values)})",
        )
    ax.set_xlabel(factor_spec.name)
    ax.set_ylabel("posterior density")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)


def _draw_categorical_marginal(
    ax,
    analyzer: BaseAnalyzer,
    factor_spec: FactorSpec,
    outcome_value: float,
    num_samples: int,
) -> None:
    """Draw a categorical factor's marginal onto ``ax`` as side-by-side bars per category.

    The blue bar (left of each category) is the analyzer's ``P(category | outcome)``.
    The green bar (right of each category) is the *empirical* per-category outcome rate
    — independent of the analyzer's posterior, computed directly from the raw data.
    For a direct empirical analyzer the two agree exactly (up to normalization); for
    a posterior-based analyzer (e.g. ``MNPEAnalyzer``) they may differ slightly if the
    model smooths.

    Each green bar is annotated with the sample count ``n`` for that category, so the
    user can see how trustworthy each bar is.
    """
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

    bar_x_positions = np.arange(num_choices)
    bar_width = 0.4
    ax.bar(
        bar_x_positions - bar_width / 2,
        posterior_probabilities,
        bar_width,
        color="steelblue",
        alpha=0.8,
        label=f"P(category | {analyzer.outcome_name}={outcome_value:g})",
    )
    ax.bar(
        bar_x_positions + bar_width / 2,
        empirical_rates,
        bar_width,
        color="seagreen",
        alpha=0.7,
        label=f"empirical {analyzer.outcome_name} rate per category",
    )
    for category_index, count in enumerate(empirical_counts):
        ax.text(
            category_index + bar_width / 2,
            empirical_rates[category_index] + 0.02,
            f"n={count}",
            ha="center",
            fontsize=8,
        )

    ax.set_xticks(bar_x_positions)
    ax.set_xticklabels(choices, rotation=30, ha="right")
    ax.set_ylabel("probability")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3, axis="y")

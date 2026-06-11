# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from isaaclab_arena.analysis.sensitivity.analyzer_base import SUCCESS_THRESHOLD

if TYPE_CHECKING:
    from isaaclab_arena.analysis.sensitivity.analyzer_base import BaseAnalyzer
    from isaaclab_arena.analysis.sensitivity.dataset import FactorSpec, SensitivityDataset

# Shared styling — kept in one place so the continuous and categorical drawers stay
# visually consistent (same colours mean the same thing across both plot types).
_POSTERIOR_COLOR = "steelblue"  # analyzer posterior: density curve / left bar
_SUCCESS_COLOR = "seagreen"  # outcome achieved: success rug / empirical-rate bar
_FAILURE_COLOR = "firebrick"  # outcome not achieved: failure rug
_NEUTRAL_COLOR = "slategray"  # rug for non-binary outcomes (no success/failure split)
_RUG_MARKER_SIZE = 80  # scatter marker size for empirical rug ticks
_RUG_SUCCESS_OFFSET = -0.05  # rug y-offset (× density.max()) for successes / neutral ticks
_RUG_FAILURE_OFFSET = -0.10  # rug y-offset (× density.max()) for failures


def plot_posterior_marginals(
    dataset: SensitivityDataset,
    outcome_name: str,
    outcome_value: float | None = None,
    output_path: str | None = None,
):
    """robolab-style quick look: one marginal-posterior panel per factor in a single figure.

    Fits the analyzer for ``outcome_name`` and draws each factor's marginal conditioned on a
    single observation — the simple "did the posterior recover the relationship?" view, the
    counterpart to the full grid in ``generate_pdf_report``. Operates on an in-memory dataset
    (e.g. a synthetic simulator's output), so no ``factors.yaml`` / JSONL round-trip is needed.

    Args:
        dataset: The factors+outcomes to analyze.
        outcome_name: Which outcome to condition the posteriors on.
        outcome_value: Value to condition on. Defaults to ``1.0`` for a binary outcome
            (the "success" branch) or the empirical median otherwise.
        output_path: If given, save the figure here (parent dirs created). PNG/PDF/etc.

    Returns:
        The matplotlib ``Figure`` (so the caller can show, tweak, or save it).
    """
    import matplotlib.pyplot as plt

    from isaaclab_arena.analysis.sensitivity.factory import make_analyzer

    analyzer = make_analyzer(dataset, outcome_name)
    analyzer.fit()

    outcome_column = dataset.x[:, dataset.outcome_columns[outcome_name]].cpu().numpy()
    if outcome_value is None:
        is_binary = set(outcome_column.flatten().tolist()).issubset({0.0, 1.0})
        outcome_value = 1.0 if is_binary else float(np.median(outcome_column))

    factors = dataset.schema.factors
    figure, axes = plt.subplots(1, len(factors), figsize=(6.0 * len(factors), 4.5), squeeze=False)
    for column_index, factor in enumerate(factors):
        ax = axes[0][column_index]
        draw_marginal(ax, analyzer, factor.name, outcome_value=outcome_value)
        ax.set_title(f"{factor.name}  (| {outcome_name}={outcome_value:g})", fontsize=10)

    slice_info = dataset.schema.slice
    figure.suptitle(
        f"Posterior marginals — {dataset.num_episodes} episodes  ({outcome_name}={outcome_value:g})\n"
        f"{slice_info.policy} / {slice_info.task} / {slice_info.embodiment}",
        fontsize=12,
        fontweight="bold",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.92])

    if output_path is not None:
        from pathlib import Path

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
    return figure


def draw_marginal(
    ax,
    analyzer: BaseAnalyzer,
    factor_name: str,
    outcome_value: float = 1.0,
    num_samples: int = 10_000,
    num_grid_points: int = 200,
) -> None:
    """Draw ``factor_name``'s marginal posterior onto ``ax``, dispatching by factor type.

    Sets axis labels, legend and grid but not the title — the caller titles the Axes.
    """
    factor_spec = analyzer._factor_spec(factor_name)
    if factor_spec.type == "continuous":
        if not hasattr(analyzer, "continuous_marginal_density"):
            raise NotImplementedError(
                f"{type(analyzer).__name__} cannot plot continuous factors:"
                " it does not implement continuous_marginal_density."
            )
        _draw_continuous_marginal(ax, analyzer, factor_spec, outcome_value, num_grid_points)
    elif factor_spec.type == "categorical":
        _draw_categorical_marginal(ax, analyzer, factor_spec, outcome_value, num_samples)
    else:
        raise NotImplementedError(f"Unsupported factor type {factor_spec.type!r}")


def draw_success_rate(
    ax,
    dataset: SensitivityDataset,
    factor_name: str,
    outcome_name: str,
    group_by: str | None = None,
    num_bins: int = 12,
) -> None:
    """Plot the empirical success rate of a binary outcome against a continuous factor.

    Bins the factor (log-spaced when it spans more than ~2 decades) and plots the per-bin
    success rate with a 95% Wilson confidence band on a 0-1 axis. The rate is read straight
    off the y-axis, and because each bin is counted independently the result is correct
    regardless of how the factor was sampled.

    If ``group_by`` names a categorical factor, one curve is drawn per category (sharing the
    same bins) so an interaction — e.g. the light gate differing by object — is visible
    instead of being averaged away.
    """
    factor_column_slice = dataset.factor_columns[factor_name]
    factor_values = dataset.theta[:, factor_column_slice].squeeze(-1).cpu().numpy()
    outcome_column_index = dataset.outcome_columns[outcome_name]
    outcome_values = dataset.x[:, outcome_column_index].cpu().numpy()

    # Log-spaced bins when the factor is positive and spans many decades, so a wide-range
    # factor (e.g. light) resolves its low end instead of collapsing into the first bin.
    # Bins are shared across groups so the per-category curves are directly comparable.
    range_low, range_high = float(factor_values.min()), float(factor_values.max())
    use_log_axis = range_low > 0 and np.log10(range_high / range_low) > 2
    if use_log_axis:
        bin_edges = np.logspace(np.log10(range_low), np.log10(range_high), num_bins + 1)
    else:
        bin_edges = np.linspace(range_low, range_high, num_bins + 1)

    # One pooled curve by default, or one per category of the group_by factor.
    if group_by is None:
        groups = [(f"empirical {outcome_name}", np.ones(len(factor_values), dtype=bool))]
    else:
        group_spec = next(factor for factor in dataset.schema.factors if factor.name == group_by)
        assert group_spec.choices is not None, f"group_by factor {group_by!r} must be categorical"
        group_codes = dataset.theta[:, dataset.factor_columns[group_by]].squeeze(-1).long().cpu().numpy()
        groups = [
            (f"{choice}  (n={int((group_codes == code).sum())})", group_codes == code)
            for code, choice in enumerate(group_spec.choices)
        ]

    annotate_counts = group_by is None  # per-bin n labels only when there's a single curve
    for group_label, group_mask in groups:
        centers, rates, ci_low, ci_high, counts = _binned_success_rate(
            factor_values[group_mask], outcome_values[group_mask], bin_edges, use_log_axis
        )
        if not centers:
            continue
        (line,) = ax.plot(centers, rates, "o-", linewidth=2, label=group_label)
        ax.fill_between(centers, ci_low, ci_high, color=line.get_color(), alpha=0.15)
        if annotate_counts:
            for center_x, rate, count in zip(centers, rates, counts):
                ax.text(center_x, min(rate + 0.04, 1.04), f"n={count}", ha="center", fontsize=7, color="gray")

    if use_log_axis:
        ax.set_xscale("log")
    ax.set_xlabel(factor_name)
    ax.set_ylabel(f"{outcome_name} (success rate)")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)


def _binned_success_rate(factor_values, outcome_values, bin_edges, use_log_axis):
    """Per-bin success rate with a 95% Wilson confidence interval.

    Returns parallel lists ``(centers, rates, ci_low, ci_high, counts)``, skipping empty
    bins. Centers are geometric (log axis) or arithmetic (linear axis) bin midpoints.
    """
    z = 1.959963985  # 95% normal quantile
    num_bins = len(bin_edges) - 1
    centers, rates, ci_low, ci_high, counts = [], [], [], [], []
    for bin_index in range(num_bins):
        # Last bin is closed on the right so the maximum value isn't dropped.
        upper = bin_edges[bin_index + 1]
        in_bin = (factor_values >= bin_edges[bin_index]) & (
            factor_values <= upper if bin_index == num_bins - 1 else factor_values < upper
        )
        num_in_bin = int(in_bin.sum())
        if num_in_bin == 0:
            continue
        rate = int((outcome_values[in_bin] >= SUCCESS_THRESHOLD).sum()) / num_in_bin
        # Wilson score interval — stays inside [0, 1] even at rate 0/1 or small n.
        denominator = 1 + z * z / num_in_bin
        center = (rate + z * z / (2 * num_in_bin)) / denominator
        margin = z / denominator * np.sqrt(rate * (1 - rate) / num_in_bin + z * z / (4 * num_in_bin * num_in_bin))
        if use_log_axis:
            centers.append(float(np.sqrt(bin_edges[bin_index] * upper)))
        else:
            centers.append(float(0.5 * (bin_edges[bin_index] + upper)))
        rates.append(rate)
        ci_low.append(max(0.0, center - margin))
        ci_high.append(min(1.0, center + margin))
        counts.append(num_in_bin)
    return centers, rates, ci_low, ci_high, counts


def draw_conditional_marginal(
    ax,
    analyzer: BaseAnalyzer,
    factor_name: str,
    num_curves: int = 3,
    num_grid_points: int = 200,
) -> None:
    """Overlay the posterior ``P(factor | outcome=v)`` at several outcome values ``v``.

    For a continuous outcome there is no single "success" value to condition on, so the
    conditioning value is swept across the observed outcome range (5th-95th percentile)
    and the resulting posteriors are overlaid. The shift between curves shows how the
    factor's posterior depends on the outcome — e.g. fast vs slow episodes favouring
    brighter vs darker light — the conditional query only a posterior model provides.
    """
    from matplotlib import colormaps

    factor_spec = analyzer._factor_spec(factor_name)
    outcome_column_index = analyzer.dataset.outcome_columns[analyzer.outcome_name]
    outcome_data = analyzer.dataset.x[:, outcome_column_index].cpu().numpy()
    low_value, high_value = np.percentile(outcome_data, [5, 95])
    conditioning_values = np.linspace(low_value, high_value, num_curves)
    colors = colormaps["coolwarm"](np.linspace(0, 1, num_curves))

    for conditioning_value, color in zip(conditioning_values, colors):
        grid, density = analyzer.continuous_marginal_density(factor_name, float(conditioning_value), num_grid_points)
        ax.plot(grid, density, color=color, linewidth=2, label=f"{analyzer.outcome_name}={conditioning_value:g}")

    # Multi-decade factor: show the conditioning shift on a log axis instead of cramping it.
    if (
        factor_spec.range is not None
        and factor_spec.range[0][0] > 0
        and np.log10(factor_spec.range[0][1] / factor_spec.range[0][0]) > 2
    ):
        ax.set_xscale("log")
    ax.set_xlabel(factor_name)
    ax.set_ylabel("posterior density")
    ax.legend(loc="best", fontsize=8, title=f"conditioned on {analyzer.outcome_name}")
    ax.grid(alpha=0.3)


def draw_posterior_overlay(
    ax,
    dataset: SensitivityDataset,
    factor_name: str,
    outcome_name: str,
    group_by: str | None = None,
    num_grid_points: int = 200,
) -> None:
    """Overlay ``P(factor | outcome=1)`` as filled KDE curves, one per group (robolab style).

    For each group, fits a Gaussian KDE to the factor values of the *successful* episodes
    (``outcome >= SUCCESS_THRESHOLD``) — the posterior over the factor given success under a
    uniform prior. Curves are filled and overlaid for comparison. A wide-range factor is fit
    in log space so the density is sharp instead of crammed at the low end.
    """
    from scipy.stats import gaussian_kde

    factor_column_slice = dataset.factor_columns[factor_name]
    factor_values = dataset.theta[:, factor_column_slice].squeeze(-1).cpu().numpy()
    outcome_column_index = dataset.outcome_columns[outcome_name]
    success_mask = dataset.x[:, outcome_column_index].cpu().numpy() >= SUCCESS_THRESHOLD

    range_low, range_high = float(factor_values.min()), float(factor_values.max())
    use_log = range_low > 0 and np.log10(range_high / range_low) > 2
    to_fit_space = (lambda v: np.log10(v)) if use_log else (lambda v: v)
    grid_fit = np.linspace(to_fit_space(range_low), to_fit_space(range_high), num_grid_points)
    grid_display = 10**grid_fit if use_log else grid_fit

    if group_by is None:
        groups = [(f"{outcome_name}=1", np.ones(len(factor_values), dtype=bool))]
    else:
        group_spec = next(factor for factor in dataset.schema.factors if factor.name == group_by)
        assert group_spec.choices is not None, f"group_by factor {group_by!r} must be categorical"
        group_codes = dataset.theta[:, dataset.factor_columns[group_by]].squeeze(-1).long().cpu().numpy()
        groups = [(choice, group_codes == code) for code, choice in enumerate(group_spec.choices)]

    # Draw the curves, collecting each group's successful samples for a rug below the axis.
    max_density = 0.0
    rug_series = []  # (sample display-values, color) per group
    for group_label, group_mask in groups:
        successful_factor = factor_values[group_mask & success_mask]
        fit_values = to_fit_space(successful_factor)
        if len(successful_factor) < 2 or float(np.std(fit_values)) < 1e-9:
            continue  # KDE undefined for fewer than 2 points or zero spread
        density = gaussian_kde(fit_values)(grid_fit)
        max_density = max(max_density, float(density.max()))
        (line,) = ax.plot(grid_display, density, linewidth=2, label=f"{group_label}  (n={len(successful_factor)})")
        ax.fill_between(grid_display, 0, density, color=line.get_color(), alpha=0.3)
        rug_series.append((successful_factor, line.get_color()))

    # Rug of the actual successful samples per group, stacked just below the axis: it shows
    # where the data really is, so the KDE's taper reads as thinning data, not as signal.
    for rug_index, (sample_values, color) in enumerate(rug_series):
        rug_y = -(0.03 + 0.025 * rug_index) * max_density
        ax.scatter(
            sample_values, np.full(len(sample_values), rug_y), marker="|", color=color, s=_RUG_MARKER_SIZE, alpha=0.5
        )

    if use_log:
        ax.set_xscale("log")
    if max_density > 0:
        ax.set_ylim(-(0.03 + 0.025 * len(rug_series)) * max_density, max_density * 1.1)
    ax.set_xlabel(factor_name)
    ax.set_ylabel("posterior density")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)


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
        color=_POSTERIOR_COLOR,
        linewidth=2,
        label=f"P({factor_spec.name} | {analyzer.outcome_name}={outcome_value:g})",
    )
    ax.fill_between(grid, 0, density, color=_POSTERIOR_COLOR, alpha=0.2)

    # Binary outcomes: split the rug green/red at the success threshold (successes vs failures).
    # Continuous outcomes: the threshold is meaningless, so show one neutral rug.
    is_binary_outcome = set(empirical_outcomes.flatten().tolist()).issubset({0.0, 1.0})
    if is_binary_outcome:
        success_mask = empirical_outcomes >= SUCCESS_THRESHOLD
        ax.scatter(
            empirical_theta_values[success_mask],
            np.full(success_mask.sum(), _RUG_SUCCESS_OFFSET * density.max()),
            marker="|",
            color=_SUCCESS_COLOR,
            s=_RUG_MARKER_SIZE,
            label=f"{analyzer.outcome_name} = 1  (n={success_mask.sum()})",
        )
        ax.scatter(
            empirical_theta_values[~success_mask],
            np.full((~success_mask).sum(), _RUG_FAILURE_OFFSET * density.max()),
            marker="|",
            color=_FAILURE_COLOR,
            s=_RUG_MARKER_SIZE,
            label=f"{analyzer.outcome_name} = 0  (n={(~success_mask).sum()})",
        )
    else:
        ax.scatter(
            empirical_theta_values,
            np.full(len(empirical_theta_values), _RUG_SUCCESS_OFFSET * density.max()),
            marker="|",
            color=_NEUTRAL_COLOR,
            s=_RUG_MARKER_SIZE,
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

    Blue bar: the analyzer's ``P(category | outcome)``. Green bar: the empirical
    per-category outcome rate from the raw data, annotated with its sample count ``n``.
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
            empirical_rates[code] = float((empirical_outcomes[category_mask] >= SUCCESS_THRESHOLD).mean())

    bar_x_positions = np.arange(num_choices)
    bar_width = 0.4
    ax.bar(
        bar_x_positions - bar_width / 2,
        posterior_probabilities,
        bar_width,
        color=_POSTERIOR_COLOR,
        alpha=0.8,
        label=f"P(category | {analyzer.outcome_name}={outcome_value:g})",
    )
    ax.bar(
        bar_x_positions + bar_width / 2,
        empirical_rates,
        bar_width,
        color=_SUCCESS_COLOR,
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

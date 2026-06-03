# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Interactive HTML sensitivity report generator.

Single function entry point: :func:`generate_report` reads a (factors.yaml, JSONL) pair,
runs the analyzer pipeline for every declared (outcome, factor) combination, and emits a
self-contained HTML file with interactive Plotly plots embedded inline. Bootstrap CSS via
CDN provides the visual chrome (tabs, cards, accordion). The deliverable is a single .html
file that opens in any modern browser — no server, no Python at view time.

Why Plotly + Jinja2 + Bootstrap (vs the earlier static-PNG version):

  - Plotly plots support hover (exact (factor, density) readout), drag-to-zoom into
    specific regions (critical for sweeps spanning multiple decades), legend-click to
    hide/show traces. The static matplotlib version had none of this.
  - Bootstrap nav-tabs let users switch between outcomes (success_rate vs task_duration
    etc.) without scrolling — only the active outcome's section is visible at a time.
  - Jinja2 templating keeps the HTML structure separate from the Python data, which
    makes the template editable without touching plot generation.

The generator produces *one* HTML file. Plotly.js is loaded from the CDN by default,
which keeps file size ~500 KB. For offline viewing, pass ``plotlyjs_mode="inline"``
to embed the ~3.5 MB Plotly library directly in the HTML.

The CLI wrapper is ``isaaclab_arena.scripts.generate_sensitivity_report``.
"""

from __future__ import annotations

import datetime
import html as html_module
import json
import numpy as np
from pathlib import Path
from typing import Any, Literal

import plotly.graph_objects as go
from jinja2 import Template

from isaaclab_arena.analysis.sensitivity.analyzer import make_analyzer
from isaaclab_arena.analysis.sensitivity.dataset import FactorSpec, OutcomeSpec, SensitivityDataset

_BOOTSTRAP_CSS_URL = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"
_BOOTSTRAP_JS_URL = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"


def generate_report(
    factors_yaml_path: str | Path,
    jsonl_path: str | Path,
    output_html_path: str | Path,
    plotlyjs_mode: Literal["cdn", "inline"] = "cdn",
) -> Path:
    """Build a self-contained interactive HTML sensitivity report.

    Reads the schema from ``factors_yaml_path`` and the per-episode data from
    ``jsonl_path``, fits one analyzer per declared outcome, and renders one Plotly
    figure per (outcome, factor) pair. All figures end up inline in a single HTML file
    arranged as Bootstrap nav-tabs (one tab per outcome) with one card per factor inside
    each tab.

    Args:
        factors_yaml_path: Schema file declaring factors and outcomes.
        jsonl_path: episode_summary.jsonl from eval_runner.
        output_html_path: Destination for the report file.
        plotlyjs_mode: ``"cdn"`` for a small file that needs internet to load Plotly;
            ``"inline"`` to embed Plotly.js (~3.5 MB) for offline viewing.

    Returns:
        The resolved output path.
    """
    factors_yaml_path = Path(factors_yaml_path)
    jsonl_path = Path(jsonl_path)
    output_html_path = Path(output_html_path)

    dataset = SensitivityDataset(factors_yaml_path, jsonl_path)

    print(f"[INFO] Generating report: {len(dataset.schema.outcomes)} outcomes × {len(dataset.schema.factors)} factors")
    outcome_blocks = []
    for outcome in dataset.schema.outcomes:
        analyzer = make_analyzer(dataset, outcome.name)
        print(f"[INFO]   Fitting analyzer for outcome={outcome.name!r}  ({type(analyzer).__name__})")
        analyzer.fit()
        outcome_value = _default_outcome_value_for_analysis(dataset, outcome)

        sections = []
        for factor in dataset.schema.factors:
            print(f"[INFO]     Rendering ({outcome.name}, {factor.name}) @ outcome_value={outcome_value:g}")
            plot_html = _render_marginal_to_plotly_html(analyzer, factor, outcome_value)
            stats = _compute_summary_stats(dataset, factor, outcome, outcome_value)
            sections.append({
                "factor_name": factor.name,
                "plot_html": plot_html,
                "stats": _format_stats_for_display(stats),
            })
        outcome_blocks.append({
            "name": outcome.name,
            "conditioning_value": _format_number(outcome_value),
            "analyzer_name": type(analyzer).__name__,
            "sections": sections,
        })

    factors_yaml_text = factors_yaml_path.read_text(encoding="utf-8")
    raw_jsonl_text = _read_first_rows(jsonl_path, max_rows=10)

    html_text = _render_template(
        slice_info=dataset.schema.slice,
        num_episodes=len(dataset.rows),
        num_factors=len(dataset.schema.factors),
        num_outcomes=len(dataset.schema.outcomes),
        outcome_blocks=outcome_blocks,
        factors_yaml_text=factors_yaml_text,
        raw_jsonl_text=raw_jsonl_text,
        plotlyjs_mode=plotlyjs_mode,
    )

    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    output_html_path.write_text(html_text, encoding="utf-8")
    print(f"[INFO] Wrote report → {output_html_path}")
    return output_html_path


def _default_outcome_value_for_analysis(dataset: SensitivityDataset, outcome: OutcomeSpec) -> float:
    """Pick a sensible value to condition the posterior on for this outcome.

    Binary outcomes (only ``{0, 1}`` observed) → ``1.0`` (the "success" branch).
    Continuous outcomes → empirical median; a "typical case" value always inside the data range.
    """
    outcome_column_index = dataset.outcome_columns[outcome.name]
    values = dataset.x[:, outcome_column_index].cpu().numpy()
    if set(values.flatten().tolist()).issubset({0.0, 1.0}):
        return 1.0
    return float(np.median(values))


def _render_marginal_to_plotly_html(analyzer, factor: FactorSpec, outcome_value: float) -> str:
    """Build the Plotly figure for one (analyzer, factor) pair and return its HTML div.

    Dispatches by factor type. Continuous-factor plots get a built-in Plotly slider over
    conditioning values when the outcome is continuous (more than 2 distinct values
    observed) — the user can drag through "what if we condition on outcome=X instead?"
    and watch the posterior curve update in-place. Binary outcomes (only 0/1 observed)
    keep the static single-value plot since a 2-step slider has no value.

    Plotly.js itself is *not* embedded per-plot — the page loads it once globally via
    CDN or inline (see :func:`_render_template`), so each plot here is just the div +
    the constructor JS.
    """
    if factor.type == "continuous":
        if _outcome_is_continuous(analyzer):
            figure = _build_continuous_figure_with_slider(analyzer, factor)
        else:
            figure = _build_continuous_figure(analyzer, factor, outcome_value)
    elif factor.type == "categorical":
        figure = _build_categorical_figure(analyzer, factor, outcome_value)
    else:
        raise NotImplementedError(f"Unsupported factor type {factor.type!r}")
    return figure.to_html(include_plotlyjs=False, full_html=False, config={"displaylogo": False})


def _outcome_is_continuous(analyzer) -> bool:
    """Heuristic: an outcome is "continuous" (slider-worthy) if it has >2 distinct observed values."""
    outcome_column_index = analyzer.dataset.outcome_columns[analyzer.outcome_name]
    values = analyzer.dataset.x[:, outcome_column_index].cpu().numpy().flatten()
    unique_values = set(values.tolist())
    if unique_values.issubset({0.0, 1.0}):
        return False
    return len(unique_values) > 2


def _build_continuous_figure(analyzer, factor: FactorSpec, outcome_value: float) -> go.Figure:
    """Continuous-factor density curve + empirical rug. Hover, zoom, pan all native to Plotly."""
    grid, density = analyzer.continuous_marginal_density(factor.name, outcome_value, num_grid_points=200)
    factor_column_slice = analyzer.dataset.factor_columns[factor.name]
    outcome_column_index = analyzer.dataset.outcome_columns[analyzer.outcome_name]
    empirical_theta = analyzer.dataset.theta[:, factor_column_slice].squeeze(-1).cpu().numpy()
    # log_uniform factors store theta in log10 space; un-transform for display so rug ticks
    # land at the actual intensity values that align with the (linear-scale) curve grid.
    if factor.distribution == "log_uniform":
        empirical_theta = np.power(10.0, empirical_theta)
    empirical_outcomes = analyzer.dataset.x[:, outcome_column_index].cpu().numpy()
    density_max = float(np.max(density)) if len(density) else 1.0
    rug_y_value = -0.05 * density_max

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=grid,
            y=density,
            mode="lines",
            fill="tozeroy",
            line={"color": "steelblue", "width": 2},
            fillcolor="rgba(70, 130, 180, 0.2)",
            name=f"P({factor.name} | {analyzer.outcome_name}={outcome_value:g})",
            hovertemplate=f"{factor.name}=%{{x:.4g}}<br>density=%{{y:.4g}}<extra></extra>",
        )
    )

    is_binary_outcome = set(empirical_outcomes.flatten().tolist()).issubset({0.0, 1.0})
    if is_binary_outcome:
        success_mask = empirical_outcomes >= 0.5
        _add_rug_trace(
            figure,
            empirical_theta[success_mask],
            rug_y_value,
            color="seagreen",
            name=f"{analyzer.outcome_name} ≥ 0.5 (n={int(success_mask.sum())})",
        )
        _add_rug_trace(
            figure,
            empirical_theta[~success_mask],
            rug_y_value * 2,
            color="firebrick",
            name=f"{analyzer.outcome_name} < 0.5 (n={int((~success_mask).sum())})",
        )
    else:
        _add_rug_trace(
            figure,
            empirical_theta,
            rug_y_value,
            color="slategray",
            name=f"observed (n={len(empirical_theta)})",
            hover_values=empirical_outcomes,
            hover_label=analyzer.outcome_name,
        )

    xaxis_kwargs = {"title": factor.name}
    if factor.distribution == "log_uniform":
        xaxis_kwargs["type"] = "log"
    figure.update_layout(
        title=_plot_title(analyzer, factor.name),
        xaxis=xaxis_kwargs,
        yaxis_title="posterior density",
        template="plotly_white",
        hovermode="closest",
        height=480,
        margin={"l": 60, "r": 30, "t": 70, "b": 50},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return figure


def _build_continuous_figure_with_slider(analyzer, factor: FactorSpec) -> go.Figure:
    """Continuous-factor density curve with a draggable slider over outcome conditioning values.

    Pre-computes the posterior density at ``num_slider_steps`` evenly-spaced conditioning
    values across the empirical outcome range, packs them as Plotly frames keyed on the
    outcome value, and binds a slider to navigate them. The rug (empirical samples) is
    invariant across frames — same data, different conditional curve — so it's drawn once
    and held static while only trace 0 (the density curve) updates per frame.

    Total fit cost stays one analyzer.fit() in the outer loop; this adds ~num_slider_steps
    calls to ``continuous_marginal_density`` at report-gen time. Each call is a posterior
    sample + histogram (~50 ms for KDE, ~100 ms for NPE), so ~1-2 s extra per plot. Trivial
    at browse time — the user just drags the slider, no compute.
    """
    num_slider_steps = 15

    outcome_column_index = analyzer.dataset.outcome_columns[analyzer.outcome_name]
    empirical_outcomes = analyzer.dataset.x[:, outcome_column_index].cpu().numpy().flatten()
    factor_column_slice = analyzer.dataset.factor_columns[factor.name]
    empirical_theta = analyzer.dataset.theta[:, factor_column_slice].squeeze(-1).cpu().numpy()
    # log_uniform factors store theta in log10 space; un-transform for display so the rug
    # aligns with the curve's linear-grid x-coordinates and Plotly's log-axis ticks read as
    # actual intensity values.
    if factor.distribution == "log_uniform":
        empirical_theta = np.power(10.0, empirical_theta)

    min_outcome = float(np.min(empirical_outcomes))
    max_outcome = float(np.max(empirical_outcomes))
    if min_outcome == max_outcome:
        # Degenerate (no spread) — fall back to the static plot at that one value.
        return _build_continuous_figure(analyzer, factor, min_outcome)

    slider_outcome_values = np.linspace(min_outcome, max_outcome, num_slider_steps)
    # Default-active slider step: the empirical median, snapped to the nearest slider value.
    default_outcome_value = float(np.median(empirical_outcomes))
    active_step_index = int(np.argmin(np.abs(slider_outcome_values - default_outcome_value)))

    # Pre-compute density curves for each slider step.
    density_grids = []
    density_values = []
    for outcome_value in slider_outcome_values:
        grid, density = analyzer.continuous_marginal_density(factor.name, float(outcome_value), num_grid_points=200)
        density_grids.append(grid)
        density_values.append(density)
    density_max = float(max(np.max(d) for d in density_values)) if density_values else 1.0
    rug_y_value = -0.05 * density_max

    # Initial figure: trace 0 = density at the default-active slider step, trace 1 = rug.
    initial_density = density_values[active_step_index]
    initial_grid = density_grids[active_step_index]
    figure = go.Figure(
        data=[
            go.Scatter(
                x=initial_grid,
                y=initial_density,
                mode="lines",
                fill="tozeroy",
                line={"color": "steelblue", "width": 2},
                fillcolor="rgba(70, 130, 180, 0.2)",
                name=f"P({factor.name} | {analyzer.outcome_name})",
                hovertemplate=f"{factor.name}=%{{x:.4g}}<br>density=%{{y:.4g}}<extra></extra>",
            ),
            go.Scatter(
                x=empirical_theta,
                y=np.full(len(empirical_theta), rug_y_value),
                mode="markers",
                marker={"symbol": "line-ns-open", "size": 14, "color": "slategray", "line": {"width": 2}},
                name=f"observed (n={len(empirical_theta)})",
                customdata=empirical_outcomes,
                hovertemplate=f"{factor.name}=%{{x:.4g}}<br>{analyzer.outcome_name}=%{{customdata:.4g}}<extra></extra>",
            ),
        ],
    )

    # Frames update only trace[0] (density). traces=[0] keeps rug static at trace[1].
    figure.frames = [
        go.Frame(
            data=[
                go.Scatter(
                    x=density_grids[step_index],
                    y=density_values[step_index],
                    mode="lines",
                    fill="tozeroy",
                    line={"color": "steelblue", "width": 2},
                    fillcolor="rgba(70, 130, 180, 0.2)",
                    hovertemplate=f"{factor.name}=%{{x:.4g}}<br>density=%{{y:.4g}}<extra></extra>",
                )
            ],
            name=f"{slider_outcome_values[step_index]:.3g}",
            traces=[0],
        )
        for step_index in range(num_slider_steps)
    ]

    slider_steps = [
        {
            "method": "animate",
            "args": [
                [f"{slider_outcome_values[step_index]:.3g}"],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0},
                },
            ],
            "label": f"{slider_outcome_values[step_index]:.3g}",
        }
        for step_index in range(num_slider_steps)
    ]

    xaxis_kwargs = {"title": factor.name}
    if factor.distribution == "log_uniform":
        xaxis_kwargs["type"] = "log"
    figure.update_layout(
        title=_plot_title(analyzer, factor.name),
        xaxis=xaxis_kwargs,
        yaxis_title="posterior density",
        template="plotly_white",
        hovermode="closest",
        height=560,
        margin={"l": 60, "r": 30, "t": 70, "b": 110},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        sliders=[{
            "active": active_step_index,
            "currentvalue": {
                "prefix": f"Conditioning on {analyzer.outcome_name} = ",
                "font": {"size": 14},
            },
            "steps": slider_steps,
            "pad": {"t": 50, "b": 10},
            "len": 0.9,
            "x": 0.05,
        }],
    )
    return figure


def _add_rug_trace(
    figure: go.Figure,
    x_values: np.ndarray,
    y_value: float,
    color: str,
    name: str,
    hover_values: np.ndarray | None = None,
    hover_label: str | None = None,
) -> None:
    """Add a single rug (vertical-tick scatter) trace to ``figure``.

    Uses ``line-ns-open`` marker symbol for the classic rug look. If ``hover_values`` is
    supplied, hover reveals the per-sample outcome value alongside the factor value.
    """
    customdata = None
    if hover_values is not None and hover_label is not None:
        customdata = hover_values
        hovertemplate = f"%{{x:.4g}}<br>{hover_label}=%{{customdata:.4g}}<extra></extra>"
    else:
        hovertemplate = f"%{{x:.4g}}<extra>{html_module.escape(name)}</extra>"
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=np.full(len(x_values), y_value),
            mode="markers",
            marker={"symbol": "line-ns-open", "size": 14, "color": color, "line": {"width": 2}},
            name=name,
            customdata=customdata,
            hovertemplate=hovertemplate,
        )
    )


def _build_categorical_figure(analyzer, factor: FactorSpec, outcome_value: float) -> go.Figure:
    """Categorical-factor side-by-side bars: analyzer posterior vs empirical rate per category."""
    assert factor.choices is not None
    choices = factor.choices
    num_choices = len(choices)
    factor_column_slice = analyzer.dataset.factor_columns[factor.name]
    outcome_column_index = analyzer.dataset.outcome_columns[analyzer.outcome_name]

    posterior_probabilities = analyzer.categorical_marginal_probs(factor.name, outcome_value, num_samples=10_000)
    empirical_theta_codes = analyzer.dataset.theta[:, factor_column_slice].squeeze(-1).long().cpu().numpy()
    empirical_outcomes = analyzer.dataset.x[:, outcome_column_index].cpu().numpy()
    empirical_rates = np.zeros(num_choices)
    empirical_counts = np.zeros(num_choices, dtype=int)
    for code in range(num_choices):
        category_mask = empirical_theta_codes == code
        empirical_counts[code] = int(category_mask.sum())
        if category_mask.any():
            empirical_rates[code] = float((empirical_outcomes[category_mask] >= 0.5).mean())

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=choices,
            y=posterior_probabilities,
            name=f"P(category | {analyzer.outcome_name}={outcome_value:g})",
            marker_color="steelblue",
            hovertemplate="%{x}<br>posterior=%{y:.4g}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            x=choices,
            y=empirical_rates,
            name=f"empirical {analyzer.outcome_name} rate",
            marker_color="seagreen",
            customdata=empirical_counts,
            hovertemplate="%{x}<br>empirical=%{y:.4g}<br>n=%{customdata}<extra></extra>",
        )
    )
    figure.update_layout(
        title=_plot_title(analyzer, factor.name),
        barmode="group",
        xaxis_title=factor.name,
        yaxis_title="probability",
        template="plotly_white",
        yaxis={"range": [0, 1.05]},
        height=480,
        margin={"l": 60, "r": 30, "t": 70, "b": 80},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return figure


def _plot_title(analyzer, factor_name: str) -> str:
    return (
        f"Sensitivity of {analyzer.outcome_name} to {factor_name}"
        f" — {analyzer.dataset.schema.slice.policy}"
        f" / {analyzer.dataset.schema.slice.task}"
        f" / {analyzer.dataset.schema.slice.embodiment}"
    )


def _compute_summary_stats(
    dataset: SensitivityDataset, factor: FactorSpec, outcome: OutcomeSpec, outcome_value: float
) -> dict:
    """Empirical summary stats kept distinct from the analyzer's posterior for cross-checking."""
    outcome_column_index = dataset.outcome_columns[outcome.name]
    outcome_values = dataset.x[:, outcome_column_index].cpu().numpy()
    is_binary_outcome = set(outcome_values.flatten().tolist()).issubset({0.0, 1.0})

    stats: dict[str, Any] = {
        "num_episodes": int(len(dataset.rows)),
        "is_binary_outcome": is_binary_outcome,
    }
    if is_binary_outcome:
        success_count = int((outcome_values >= 0.5).sum())
        stats["success_count"] = success_count
        stats["failure_count"] = int(len(outcome_values) - success_count)
        stats["overall_success_rate"] = float(outcome_values.mean())
    else:
        stats["outcome_min"] = float(outcome_values.min())
        stats["outcome_max"] = float(outcome_values.max())
        stats["outcome_median"] = float(np.median(outcome_values))
        stats["outcome_mean"] = float(outcome_values.mean())

    factor_column_slice = dataset.factor_columns[factor.name]
    factor_values = dataset.theta[:, factor_column_slice].squeeze(-1).cpu().numpy()
    if factor.type == "continuous":
        stats["factor_min_observed"] = float(factor_values.min())
        stats["factor_max_observed"] = float(factor_values.max())
        stats["factor_unique_count"] = int(np.unique(np.round(factor_values, 6)).size)
        if factor.range is not None and len(factor.range) == 1:
            range_low, range_high = factor.range[0]
            stats["factor_range"] = [float(range_low), float(range_high)]
    elif factor.type == "categorical" and factor.choices is not None:
        per_category_counts = {}
        for code, choice in enumerate(factor.choices):
            per_category_counts[choice] = int((factor_values == code).sum())
        stats["per_category_counts"] = per_category_counts
    return stats


def _format_stats_for_display(stats: dict) -> list[tuple[str, str]]:
    """Flatten a stats dict into ordered (label, html_value) pairs for the template."""
    formatted: list[tuple[str, str]] = []
    for key, value in stats.items():
        formatted.append((key, _format_value(value)))
    return formatted


def _format_value(value: Any) -> str:
    """Render one stats value with light formatting (numbers compact, dicts as nested list)."""
    if isinstance(value, dict):
        items = "".join(
            f"<li><code>{html_module.escape(str(k))}</code>: {_format_value(v)}</li>" for k, v in value.items()
        )
        return f"<ul class='mb-0'>{items}</ul>"
    if isinstance(value, list):
        return html_module.escape(
            ", ".join(_format_number(item) if isinstance(item, (int, float)) else str(item) for item in value)
        )
    if isinstance(value, bool):
        return "✓" if value else "✗"
    if isinstance(value, (int, float)):
        return _format_number(value)
    return html_module.escape(str(value))


def _format_number(value) -> str:
    """Compact number formatting: int-ish values stay integer; floats use 4-significant-digit g-format."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value == int(value):
            return f"{int(value)}"
        return f"{value:.4g}"
    return str(value)


def _read_first_rows(jsonl_path: Path, max_rows: int) -> str:
    """Return the first ``max_rows`` JSONL rows pretty-printed for the accordion snippet."""
    lines: list[str] = []
    with open(jsonl_path, encoding="utf-8") as jsonl_file:
        for row_index, raw_line in enumerate(jsonl_file):
            if row_index >= max_rows:
                lines.append("…")
                break
            try:
                lines.append(json.dumps(json.loads(raw_line), indent=2))
            except json.JSONDecodeError:
                lines.append(raw_line.rstrip())
    return "\n".join(lines)


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ title }}</title>
  <link rel="stylesheet" href="{{ bootstrap_css }}">
  {{ plotlyjs_block|safe }}
  <style>
    body { padding: 2em; max-width: 1300px; margin: 0 auto; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    h1 { border-bottom: 2px solid #333; padding-bottom: 0.3em; margin-bottom: 1em; }
    .meta-card { background: linear-gradient(135deg, #f8f9fa, #eef2f7); }
    .meta-card .row > div { padding: 0.25em 1em; }
    .factor-card { margin-bottom: 1.5em; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    .nav-tabs .nav-link { font-weight: 500; }
    .nav-tabs .badge { font-weight: 400; margin-left: 0.4em; }
    .stats-table th { width: 35%; font-weight: 500; color: #555; }
    .stats-table td { font-family: ui-monospace, "SF Mono", Monaco, monospace; }
    pre.scroll-pre { max-height: 400px; overflow-y: auto; background: #f8f9fa; padding: 1em;
                    border: 1px solid #dee2e6; border-radius: 4px; font-size: 0.85em; }
    .conditioning-note { color: #6c757d; font-size: 0.95em; margin-top: 0.5em; }
    footer { margin-top: 4em; padding-top: 1em; border-top: 1px solid #dee2e6; color: #888; font-size: 0.85em; }
  </style>
</head>
<body>

  <h1>{{ title }}</h1>

  <div class="card meta-card mb-4">
    <div class="card-body">
      <div class="row">
        <div class="col-md-4"><strong>Policy:</strong> <code>{{ slice.policy }}</code></div>
        <div class="col-md-4"><strong>Task:</strong> <code>{{ slice.task }}</code></div>
        <div class="col-md-4"><strong>Embodiment:</strong> <code>{{ slice.embodiment }}</code></div>
        <div class="col-md-4"><strong>Episodes:</strong> {{ num_episodes }}</div>
        <div class="col-md-4"><strong>Factors declared:</strong> {{ num_factors }}</div>
        <div class="col-md-4"><strong>Outcomes declared:</strong> {{ num_outcomes }}</div>
        <div class="col-md-12 text-muted"><small>Generated {{ timestamp }}</small></div>
      </div>
    </div>
  </div>

  <ul class="nav nav-tabs" id="outcome-tabs" role="tablist">
    {% for outcome in outcome_blocks %}
    <li class="nav-item" role="presentation">
      <button class="nav-link {% if loop.first %}active{% endif %}"
              id="tab-{{ outcome.name }}"
              data-bs-toggle="tab"
              data-bs-target="#pane-{{ outcome.name }}"
              type="button" role="tab"
              aria-controls="pane-{{ outcome.name }}"
              aria-selected="{% if loop.first %}true{% else %}false{% endif %}">
        {{ outcome.name }}
        <span class="badge bg-secondary">{{ outcome.analyzer_name }}</span>
      </button>
    </li>
    {% endfor %}
  </ul>

  <div class="tab-content" id="outcome-tab-content">
    {% for outcome in outcome_blocks %}
    <div class="tab-pane fade {% if loop.first %}show active{% endif %}"
         id="pane-{{ outcome.name }}" role="tabpanel"
         aria-labelledby="tab-{{ outcome.name }}">
      <div class="mt-3">
        <p class="conditioning-note">
          Posterior conditioned on <code>{{ outcome.name }} = {{ outcome.conditioning_value }}</code>
          (analyzer: <code>{{ outcome.analyzer_name }}</code>).
        </p>
        {% for section in outcome.sections %}
        <div class="card factor-card">
          <div class="card-header">
            <strong>Factor:</strong> <code>{{ section.factor_name }}</code>
          </div>
          <div class="card-body">
            {{ section.plot_html|safe }}
            <hr>
            <h6 class="text-muted">Empirical summary</h6>
            <table class="table table-sm stats-table">
              <tbody>
              {% for key, value in section.stats %}
              <tr><th>{{ key }}</th><td>{{ value|safe }}</td></tr>
              {% endfor %}
              </tbody>
            </table>
          </div>
        </div>
        {% endfor %}
      </div>
    </div>
    {% endfor %}
  </div>

  <div class="accordion mt-5" id="extras">
    <div class="accordion-item">
      <h2 class="accordion-header">
        <button class="accordion-button collapsed" type="button"
                data-bs-toggle="collapse" data-bs-target="#factors-yaml-pane">
          Factors schema (factors.yaml)
        </button>
      </h2>
      <div id="factors-yaml-pane" class="accordion-collapse collapse" data-bs-parent="#extras">
        <div class="accordion-body"><pre class="scroll-pre">{{ factors_yaml_text }}</pre></div>
      </div>
    </div>
    <div class="accordion-item">
      <h2 class="accordion-header">
        <button class="accordion-button collapsed" type="button"
                data-bs-toggle="collapse" data-bs-target="#raw-jsonl-pane">
          Raw episode_summary snippet (first {{ raw_jsonl_max_rows }} rows)
        </button>
      </h2>
      <div id="raw-jsonl-pane" class="accordion-collapse collapse" data-bs-parent="#extras">
        <div class="accordion-body"><pre class="scroll-pre">{{ raw_jsonl_text }}</pre></div>
      </div>
    </div>
  </div>

  <footer>
    Generated by <code>isaaclab_arena.analysis.sensitivity.report</code>. Plots are
    interactive: hover for exact values, drag-rectangle to zoom, double-click to reset,
    click legend entries to hide/show traces.
  </footer>

  <script src="{{ bootstrap_js }}"></script>
</body>
</html>
"""


def _render_template(
    slice_info,
    num_episodes: int,
    num_factors: int,
    num_outcomes: int,
    outcome_blocks: list[dict],
    factors_yaml_text: str,
    raw_jsonl_text: str,
    plotlyjs_mode: Literal["cdn", "inline"],
) -> str:
    """Apply the Jinja2 template to the assembled context."""
    title = f"Sensitivity report — {slice_info.policy} / {slice_info.task} / {slice_info.embodiment}"
    if plotlyjs_mode == "inline":
        from plotly.offline import get_plotlyjs

        plotlyjs_block = f"<script type='text/javascript'>{get_plotlyjs()}</script>"
    else:
        plotlyjs_block = "<script src='https://cdn.plot.ly/plotly-2.35.2.min.js' charset='utf-8'></script>"

    rendered = Template(_HTML_TEMPLATE).render(
        title=title,
        slice=slice_info,
        num_episodes=num_episodes,
        num_factors=num_factors,
        num_outcomes=num_outcomes,
        outcome_blocks=outcome_blocks,
        factors_yaml_text=factors_yaml_text,
        raw_jsonl_text=raw_jsonl_text,
        raw_jsonl_max_rows=10,
        timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
        bootstrap_css=_BOOTSTRAP_CSS_URL,
        bootstrap_js=_BOOTSTRAP_JS_URL,
        plotlyjs_block=plotlyjs_block,
    )
    return rendered

# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
from pathlib import Path

from isaaclab_arena.analysis.sensitivity import make_analyzer
from isaaclab_arena.analysis.sensitivity.dataset import SensitivityDataset
from isaaclab_arena.analysis.sensitivity.plotting import draw_marginal


def generate_pdf_report(
    factors_yaml_path: str | Path,
    jsonl_path: str | Path,
    output_pdf_path: str | Path,
) -> Path:
    """Build a single-PDF sensitivity report covering every (outcome, factor) pair.

    Args:
        factors_yaml_path: Schema file declaring factors and outcomes.
        jsonl_path: episode_summary.jsonl from eval_runner.
        output_pdf_path: Destination ``.pdf`` file (parent dirs created if absent).

    Returns:
        The resolved output path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dataset = SensitivityDataset(Path(factors_yaml_path), Path(jsonl_path))
    outcomes = dataset.schema.outcomes
    factors = dataset.schema.factors
    n_rows, n_cols = len(outcomes), len(factors)
    print(f"[INFO] PDF report: {n_rows} outcomes × {n_cols} factors  ({len(dataset.rows)} episodes)")

    figure, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 4.5 * n_rows), squeeze=False)
    for row_index, outcome in enumerate(outcomes):
        analyzer = make_analyzer(dataset, outcome.name)
        print(f"[INFO]   Fitting {type(analyzer).__name__} for outcome={outcome.name!r}")
        analyzer.fit()
        outcome_value = _default_outcome_value_for_analysis(dataset, outcome)
        for col_index, factor in enumerate(factors):
            ax = axes[row_index][col_index]
            draw_marginal(ax, analyzer, factor.name, outcome_value=outcome_value)
            ax.set_title(
                f"{outcome.name} vs {factor.name}\n(conditioned on {outcome.name}={outcome_value:g})", fontsize=10
            )

    slice_info = dataset.schema.slice
    # Two lines so the title doesn't clip on narrow (single-factor) figures.
    figure.suptitle(
        f"Sensitivity report — {len(dataset.rows)} episodes\n"
        f"{slice_info.policy} / {slice_info.task} / {slice_info.embodiment}",
        fontsize=12,
        fontweight="bold",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.94])  # leave room for the two-line suptitle

    output_pdf_path = Path(output_pdf_path)
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_pdf_path)  # .pdf extension → matplotlib's PDF backend
    plt.close(figure)
    print(f"[INFO] Wrote report → {output_pdf_path}")
    return output_pdf_path


def _default_outcome_value_for_analysis(dataset: SensitivityDataset, outcome) -> float:
    """Pick a sensible value to condition the posterior on for this outcome.

    Binary outcomes (only ``{0, 1}`` observed) → ``1.0`` (the "success" branch).
    Continuous outcomes → empirical median; a "typical case" value always inside the data range.
    """
    outcome_column_index = dataset.outcome_columns[outcome.name]
    values = dataset.x[:, outcome_column_index].cpu().numpy()
    if set(values.flatten().tolist()).issubset({0.0, 1.0}):
        return 1.0
    return float(np.median(values))

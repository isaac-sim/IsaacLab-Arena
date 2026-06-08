# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_arena.analysis.sensitivity.analyzer_base import BaseAnalyzer
    from isaaclab_arena.analysis.sensitivity.dataset import SensitivityDataset


def make_analyzer(dataset: SensitivityDataset, outcome_name: str) -> BaseAnalyzer:
    """Construct the right analyzer for the dataset's factor mix and outcome shape.

    This MVP ships two analyzers, one from each family:
      - any continuous + any categorical (mixed) → :class:`MNPEAnalyzer`
      - 1 continuous + 0 categorical AND a binary outcome → :class:`KDEAnalyzer`
        (avoids sbi NPE's 1D-theta Gaussian-shape constraint; exact under uniform prior)

    The remaining factor mixes are deferred to ``cvolk/feature/sensitivity_deferred_analyzers``
    and asserted against here so the gap fails loudly instead of mis-dispatching:
      - all categorical (zero continuous) → ``FrequencyTableAnalyzer``
      - multi-continuous, or 1 continuous + non-binary outcome → ``NPEAnalyzer``

    The binary check reads the outcome column off the dataset rather than the schema,
    since outcome ``type: float`` in factors.yaml covers both continuous durations and
    binary 0/1 success rates.

    Analyzer classes are imported lazily so that importing this module (and the package
    that re-exports it, which happens on the eval-time ``episode_writer`` path) doesn't
    pull in torch/sbi until analysis runs.
    """
    from isaaclab_arena.analysis.sensitivity.empirical_analyzer import KDEAnalyzer
    from isaaclab_arena.analysis.sensitivity.posterior_analyzer import MNPEAnalyzer

    num_continuous_factors = sum(1 for factor in dataset.schema.factors if factor.type == "continuous")
    num_categorical_factors = sum(1 for factor in dataset.schema.factors if factor.type == "categorical")
    assert num_continuous_factors + num_categorical_factors > 0, "Schema declares no factors"

    # Mixed continuous + categorical → the sbi mixed-density port from robolab.
    if num_continuous_factors > 0 and num_categorical_factors > 0:
        return MNPEAnalyzer(dataset, outcome_name)

    # Pure-categorical needs the (deferred) frequency-table analyzer.
    assert num_continuous_factors > 0, (
        "Pure-categorical schemas need FrequencyTableAnalyzer, parked on "
        "cvolk/feature/sensitivity_deferred_analyzers for this MVP."
    )

    # All-continuous from here. Only the single-factor binary case (KDE) ships; the
    # multi-continuous case needs the (deferred) NPE analyzer.
    assert num_continuous_factors == 1, (
        "Multi-continuous-factor schemas need NPEAnalyzer, parked on "
        "cvolk/feature/sensitivity_deferred_analyzers for this MVP."
    )
    outcome_column_index = dataset.outcome_columns[outcome_name]
    unique_outcome_values = set(dataset.x[:, outcome_column_index].flatten().tolist())
    assert unique_outcome_values.issubset({0.0, 1.0}), (
        "A single continuous factor with a non-binary outcome needs NPEAnalyzer, parked on "
        "cvolk/feature/sensitivity_deferred_analyzers for this MVP. (Binary outcomes use KDEAnalyzer.)"
    )
    return KDEAnalyzer(dataset, outcome_name)

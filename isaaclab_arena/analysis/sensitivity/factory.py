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

    Dispatch table (checked top-to-bottom):
      - 1 continuous + 0 categorical AND the outcome is binary → :class:`KDEAnalyzer`
        (avoids sbi NPE's 1D-theta Gaussian-shape constraint; exact under uniform prior)
      - any continuous + any categorical → :class:`MNPEAnalyzer`
      - all categorical (zero continuous) → :class:`FrequencyTableAnalyzer`
      - all continuous (zero categorical) → :class:`NPEAnalyzer`
        (the multi-continuous-factor case; theta is multi-D so no Gaussian fallback)

    The KDE branch checks the outcome column's binary-ness on the dataset itself rather
    than the schema, since outcome ``type: float`` in factors.yaml covers both continuous
    durations and binary 0/1 success rates.

    Analyzer classes are imported lazily so that importing this module (and the package
    that re-exports it, which happens on the eval-time ``episode_writer`` path) doesn't
    pull in torch/sbi until analysis runs.
    """
    from isaaclab_arena.analysis.sensitivity.empirical_analyzer import FrequencyTableAnalyzer, KDEAnalyzer
    from isaaclab_arena.analysis.sensitivity.posterior_analyzer import MNPEAnalyzer, NPEAnalyzer

    num_continuous_factors = sum(1 for factor in dataset.schema.factors if factor.type == "continuous")
    num_categorical_factors = sum(1 for factor in dataset.schema.factors if factor.type == "categorical")
    assert num_continuous_factors + num_categorical_factors > 0, "Schema declares no factors"

    if num_continuous_factors == 1 and num_categorical_factors == 0:
        outcome_column_index = dataset.outcome_columns[outcome_name]
        outcome_values = dataset.x[:, outcome_column_index]
        unique_outcome_values = set(outcome_values.flatten().tolist())
        if unique_outcome_values.issubset({0.0, 1.0}):
            return KDEAnalyzer(dataset, outcome_name)

    if num_continuous_factors > 0 and num_categorical_factors > 0:
        return MNPEAnalyzer(dataset, outcome_name)
    if num_categorical_factors > 0:
        return FrequencyTableAnalyzer(dataset, outcome_name)
    return NPEAnalyzer(dataset, outcome_name)

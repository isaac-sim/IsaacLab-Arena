# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_arena.analysis.sensitivity.analyzer_base import BaseAnalyzer
    from isaaclab_arena.analysis.sensitivity.dataset import SensitivityDataset


def make_analyzer(dataset: SensitivityDataset, outcome_name: str) -> BaseAnalyzer:
    """Construct the analyzer matching the dataset's factor mix and outcome.

      - mixed continuous + categorical → :class:`MNPEAnalyzer`
      - one continuous factor with a binary outcome → :class:`KDEAnalyzer`

    Other mixes are asserted against so an unsupported case fails loudly. The binary
    check reads the outcome values off the dataset, since factors.yaml types all
    outcomes as float.
    """
    # Import lazily so importing this module stays light — torch/sbi load only when an
    # analysis actually runs.
    from isaaclab_arena.analysis.sensitivity.empirical_analyzer import KDEAnalyzer
    from isaaclab_arena.analysis.sensitivity.posterior_analyzer import MNPEAnalyzer

    num_continuous_factors = sum(1 for factor in dataset.schema.factors if factor.type == "continuous")
    num_categorical_factors = sum(1 for factor in dataset.schema.factors if factor.type == "categorical")
    assert num_continuous_factors + num_categorical_factors > 0, "Schema declares no factors"

    # Mixed continuous + categorical → mixed neural posterior estimation.
    if num_continuous_factors > 0 and num_categorical_factors > 0:
        return MNPEAnalyzer(dataset, outcome_name)

    assert num_continuous_factors > 0, "Pure-categorical schemas are not supported yet."

    # All-continuous from here. Only the single-factor binary-outcome case is supported.
    assert num_continuous_factors == 1, "Schemas with more than one continuous factor are not supported yet."
    outcome_column_index = dataset.outcome_columns[outcome_name]
    unique_outcome_values = set(dataset.x[:, outcome_column_index].flatten().tolist())
    assert unique_outcome_values.issubset(
        {0.0, 1.0}
    ), "A single continuous factor is only supported with a binary outcome."
    return KDEAnalyzer(dataset, outcome_name)

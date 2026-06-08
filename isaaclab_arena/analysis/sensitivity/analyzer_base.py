# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod

from isaaclab_arena.analysis.sensitivity.dataset import FactorSpec, SensitivityDataset


class BaseAnalyzer(ABC):
    """Abstract base for sensitivity analyzers.

    Validates the dataset and outcome on construction. Subclasses implement ``fit`` and
    ``categorical_marginal_probs``.
    """

    def __init__(self, dataset: SensitivityDataset, outcome_name: str):
        self.dataset = dataset
        self.outcome_name = outcome_name
        assert (
            outcome_name in dataset.outcome_columns
        ), f"Outcome {outcome_name!r} not found in schema; available: {list(dataset.outcome_columns)}"
        assert len(dataset.schema.factors) > 0, "Schema declares no factors"

    @abstractmethod
    def fit(self, training_batch_size: int = 50) -> None:
        """Prepare the analyzer so its query methods can be called.

        An implementation may train an estimator or do nothing if its queries read from
        the data directly.
        """

    @abstractmethod
    def categorical_marginal_probs(self, factor_name: str, outcome_value: float, num_samples: int) -> np.ndarray:
        """Return ``P(category | outcome=outcome_value)`` for one categorical factor.

        The result is a 1D numpy array of length ``len(factor.choices)`` summing to 1.
        """

    def _factor_spec(self, factor_name: str) -> FactorSpec:
        """Return the ``FactorSpec`` for ``factor_name``, asserting it exists in the schema."""
        assert (
            factor_name in self.dataset.factor_columns
        ), f"Factor {factor_name!r} not in schema; available: {list(self.dataset.factor_columns)}"
        return next(factor for factor in self.dataset.schema.factors if factor.name == factor_name)

# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod

from isaaclab_arena.analysis.sensitivity.dataset import FactorSpec, SensitivityDataset


class BaseAnalyzer(ABC):
    """Abstract base — owns state validation and the abstract posterior-query surface.

    Subclasses must implement:
      - ``fit`` — train (or no-op) so queries can be called afterwards.
      - ``categorical_marginal_probs`` — return ``P(category | outcome)`` for a categorical factor.
    Continuous-factor queries (``continuous_marginal_density``) live on the analyzers that
    provide them (``PosteriorAnalyzer`` and ``KDEAnalyzer``); the categorical-only analyzers
    never need them by construction.
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
        """Train the posterior (or no-op for empirical) so queries can be called afterwards.

        For NPE/MNPE this trains a neural density estimator on ``(theta, x_selected)``,
        where ``x_selected`` is the single outcome column named by ``outcome_name``. For
        the empirical analyzer this is a no-op — the categorical posterior is computed
        directly from the data at query time.
        """

    @abstractmethod
    def categorical_marginal_probs(self, factor_name: str, outcome_value: float, num_samples: int) -> np.ndarray:
        """Return ``P(category | outcome=outcome_value)`` for one categorical factor.

        Output is a 1D numpy array of length ``len(factor.choices)`` whose entries sum to 1.
        For posterior analyzers this is computed by sampling the trained posterior and
        counting category frequencies; for the empirical analyzer it's the normalized
        per-category empirical success rate.
        """

    def _factor_spec(self, factor_name: str) -> FactorSpec:
        """Return the ``FactorSpec`` for ``factor_name``, asserting it exists in the schema."""
        assert (
            factor_name in self.dataset.factor_columns
        ), f"Factor {factor_name!r} not in schema; available: {list(self.dataset.factor_columns)}"
        return next(factor for factor in self.dataset.schema.factors if factor.name == factor_name)

# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from isaaclab_arena.analysis.sensitivity.analyzer_base import SUCCESS_THRESHOLD, BaseAnalyzer
from isaaclab_arena.analysis.sensitivity.dataset import SensitivityDataset


class EmpiricalAnalyzer(BaseAnalyzer):
    """Abstract base for analyzers that read the posterior directly from the data.

    Under a uniform prior, ``P(theta | success) ∝ P(success | theta)``, so the
    distribution of successful-theta samples is the posterior. Outcome is treated as
    binary: an episode is a success when its selected outcome column is
    ``>= SUCCESS_THRESHOLD``.
    """

    def _success_mask(self) -> np.ndarray:
        """Boolean array over episodes: True where the selected outcome counts as a success."""
        outcome_column_index = self.dataset.outcome_columns[self.outcome_name]
        outcome_values = self.dataset.x[:, outcome_column_index].cpu().numpy()
        return outcome_values >= SUCCESS_THRESHOLD


class KDEAnalyzer(EmpiricalAnalyzer):
    """Analyzer for a single continuous factor with a binary outcome.

    Fits a Gaussian KDE over the theta values of successful episodes. Under a uniform
    prior this density is proportional to ``P(success | theta)``, so its shape shows
    which factor values drove success.
    """

    def __init__(self, dataset: SensitivityDataset, outcome_name: str):
        super().__init__(dataset, outcome_name)
        num_continuous = sum(1 for factor in dataset.schema.factors if factor.type == "continuous")
        num_categorical = sum(1 for factor in dataset.schema.factors if factor.type == "categorical")
        assert num_continuous == 1 and num_categorical == 0, (
            f"KDEAnalyzer requires exactly one continuous factor and no categoricals; got {num_continuous} continuous,"
            f" {num_categorical} categorical."
        )
        self._kde = None
        self._num_successful_samples = 0
        self._num_total_samples = 0

    def fit(self, training_batch_size: int = 50) -> None:
        """Fit a Gaussian KDE over the theta values of successful episodes."""
        from scipy.stats import gaussian_kde

        theta_values = self.dataset.theta[:, 0].cpu().numpy()
        success_mask = self._success_mask()
        self._num_total_samples = int(len(theta_values))
        self._num_successful_samples = int(success_mask.sum())

        if self._num_successful_samples < 2:
            print(
                f"[WARN] KDEAnalyzer: only {self._num_successful_samples} successful samples"
                f" / {self._num_total_samples} total — KDE undefined, marginal will be uniform."
            )
            return

        successful_theta = theta_values[success_mask]
        if float(np.std(successful_theta)) < 1e-9:
            print(
                "[WARN] KDEAnalyzer: all successful theta values are identical — KDE bandwidth"
                " degenerate, marginal will be uniform."
            )
            return

        self._kde = gaussian_kde(successful_theta)
        print(
            f"[INFO] KDEAnalyzer: fit Gaussian KDE on {self._num_successful_samples} successful"
            f" theta samples / {self._num_total_samples} total."
        )

    def continuous_marginal_density(
        self, factor_name: str, outcome_value: float, num_grid_points: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the posterior density over the factor's range on a uniform grid.

        Returns ``(grid, density)``. Success conditioning (``outcome_value >=
        SUCCESS_THRESHOLD``) returns the fitted KDE; otherwise a uniform density.
        """
        factor_spec = self._factor_spec(factor_name)
        assert factor_spec.type == "continuous", "KDEAnalyzer only handles continuous factors"
        assert (
            factor_spec.range is not None and len(factor_spec.range) == 1
        ), "Continuous-factor marginal expects a populated 1D range"
        range_low, range_high = factor_spec.range[0]
        grid = np.linspace(range_low, range_high, num_grid_points)

        if outcome_value < SUCCESS_THRESHOLD or self._kde is None:
            uniform_density = 1.0 / max(range_high - range_low, 1e-9)
            return grid, np.full_like(grid, uniform_density)
        return grid, self._kde(grid)

    def categorical_marginal_probs(self, factor_name: str, outcome_value: float, num_samples: int) -> np.ndarray:
        raise NotImplementedError(
            "KDEAnalyzer handles a single continuous factor only; it has no categorical"
            " factors by construction. Mixed schemas dispatch to MNPEAnalyzer via make_analyzer."
        )

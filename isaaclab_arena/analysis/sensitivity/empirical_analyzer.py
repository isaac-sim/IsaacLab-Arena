# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from isaaclab_arena.analysis.sensitivity.analyzer_base import BaseAnalyzer
from isaaclab_arena.analysis.sensitivity.dataset import SensitivityDataset


class EmpiricalAnalyzer(BaseAnalyzer):
    """Abstract base for the direct (non-neural) analyzers.

    Subclasses exploit the same fact: under a uniform prior,
    ``P(theta | success) ∝ P(success | theta)``, so the posterior is read directly off
    the data — no neural density estimator, no parametric shape constraint. They differ
    only in factor type, which dictates the estimator:

      - :class:`KDEAnalyzer` (continuous) — a Gaussian KDE over the successful-theta
        samples: the empirical measure, kernel-smoothed because a raw continuous
        empirical measure is a sum of Diracs. The supported subclass.
      - ``FrequencyTableAnalyzer`` (categorical) — the per-category empirical success
        rate (the raw empirical measure). Not part of this MVP; plugs in here unchanged.

    Outcome is treated as binary: an episode is a "success" when its selected outcome
    column is ``>= SUCCESS_THRESHOLD``.
    """

    SUCCESS_THRESHOLD = 0.5
    """Outcome value at or above which an episode counts as a success."""

    def _success_mask(self) -> np.ndarray:
        """Boolean array over episodes: True where the selected outcome counts as a success."""
        outcome_column_index = self.dataset.outcome_columns[self.outcome_name]
        outcome_values = self.dataset.x[:, outcome_column_index].cpu().numpy()
        return outcome_values >= self.SUCCESS_THRESHOLD


class KDEAnalyzer(EmpiricalAnalyzer):
    """KDE-based analyzer for the 1-continuous-factor + binary-outcome case.

    Under a uniform prior and a binary outcome, Bayes' rule reduces to
    ``P(theta | success=1) ∝ P(success=1 | theta) · P(theta) = P(success=1 | theta) · const``.
    The empirical density of *successful*-theta samples (i.e. rows where the chosen outcome
    is 1) is directly proportional to ``P(success=1 | theta)``, and a Gaussian KDE over
    those samples gives a smoothed estimate of that conditional density. No neural fit,
    no Gaussian-shape constraint.

    This is the right primitive for the 1-continuous-factor + binary-outcome case:
    sbi NPE forces a Gaussian shape when theta is 1D — biasing the recovered peak toward
    the mean of successful-theta values rather than the true mode of the success curve.
    KDE has no such constraint and recovers multi-modal / plateau / skewed shapes faithfully.

    Sits under the shared :class:`EmpiricalAnalyzer` base; its categorical sibling
    ``FrequencyTableAnalyzer`` (same trick via frequency counts) is not part of this MVP.
    For mixed-factor workloads, :func:`make_analyzer` dispatches to ``MNPEAnalyzer``.

    Caveats:
      - Bandwidth is scipy's Scott rule default; haven't tuned for non-uniform sample
        distributions or sparse data. May over-smooth the empirical mode.
      - Only ``continuous_marginal_density`` is implemented and only for
        ``outcome_value >= 0.5`` (i.e. success conditioning). Failure conditioning would
        require fitting a second KDE over failed-theta samples; left out for simplicity.
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
        """Fit a Gaussian KDE on the successful-theta samples (no neural network involved)."""
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
        """Evaluate the KDE-based posterior over the factor's prior range.

        ``outcome_value >= 0.5`` is treated as "success conditioning" (the only case
        currently supported); the KDE is evaluated on a uniform grid spanning the
        declared factor range. ``outcome_value < 0.5`` (failure conditioning) returns
        a uniform density as a placeholder — extend by fitting a second KDE on failed
        samples if/when that case is needed.
        """
        factor_spec = self._factor_spec(factor_name)
        assert factor_spec.type == "continuous", "KDEAnalyzer only handles continuous factors"
        assert (
            factor_spec.range is not None and len(factor_spec.range) == 1
        ), "Continuous-factor marginal expects a populated 1D range"
        range_low, range_high = factor_spec.range[0]
        grid = np.linspace(range_low, range_high, num_grid_points)

        if outcome_value < self.SUCCESS_THRESHOLD or self._kde is None:
            uniform_density = 1.0 / max(range_high - range_low, 1e-9)
            return grid, np.full_like(grid, uniform_density)
        return grid, self._kde(grid)

    def categorical_marginal_probs(self, factor_name: str, outcome_value: float, num_samples: int) -> np.ndarray:
        raise NotImplementedError(
            "KDEAnalyzer handles a single continuous factor only; it has no categorical"
            " factors by construction. Mixed schemas dispatch to MNPEAnalyzer via make_analyzer."
        )

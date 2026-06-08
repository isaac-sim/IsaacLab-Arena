# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch

from isaaclab_arena.analysis.sensitivity.analyzer_base import BaseAnalyzer
from isaaclab_arena.analysis.sensitivity.dataset import SensitivityDataset


class PosteriorAnalyzer(BaseAnalyzer):
    """Base for sbi-driven analyzers that fit a neural posterior estimator.

    Subclasses choose the sbi inference class via ``_inference_cls``; the training loop
    and the density/sample queries are shared. After ``fit()``, ``self.posterior`` is an
    sbi posterior supporting ``sample(shape, x=...)`` and ``log_prob(theta, x=...)``.
    """

    def __init__(self, dataset: SensitivityDataset, outcome_name: str):
        super().__init__(dataset, outcome_name)
        self.posterior = None

    def _inference_cls(self):
        """Return the sbi inference *class* to train with (e.g. ``sbi.inference.MNPE``)."""
        raise NotImplementedError("PosteriorAnalyzer subclasses must implement _inference_cls")

    def _make_inference(self):
        """Instantiate the chosen sbi inference class on the dataset's uniform prior."""
        return self._inference_cls()(prior=self.dataset.prior)

    def fit(self, training_batch_size: int = 50) -> None:
        """Train the sbi estimator and store the posterior on ``self``.

        Conditions on the single outcome column named by ``outcome_name``.
        """
        outcome_column_index = self.dataset.outcome_columns[self.outcome_name]
        selected_outcome_column = self.dataset.x[:, outcome_column_index : outcome_column_index + 1]

        print(
            f"[INFO] {type(self).__name__}: fitting on {self.dataset.theta.shape[0]} samples"
            f" (theta dim={self.dataset.theta.shape[1]},"
            f" x dim={selected_outcome_column.shape[1]})."
        )
        inference = self._make_inference()
        inference.append_simulations(self.dataset.theta, selected_outcome_column)
        density_estimator = inference.train(training_batch_size=training_batch_size)
        self.posterior = inference.build_posterior(density_estimator)

    def continuous_marginal_density(
        self, factor_name: str, outcome_value: float, num_grid_points: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate ``P(factor_value | outcome=outcome_value)`` over the factor's range.

        Returns ``(grid, density)`` as numpy arrays of length ``num_grid_points``.
        """
        assert self.posterior is not None, "Call fit() before querying the posterior"
        factor_spec = self._factor_spec(factor_name)
        assert (
            factor_spec.type == "continuous"
        ), f"continuous_marginal_density expects a continuous factor; {factor_name!r} is {factor_spec.type!r}"
        assert (
            factor_spec.range is not None and len(factor_spec.range) == 1
        ), "Continuous-factor marginal expects a populated 1D range"

        factor_column_slice = self.dataset.factor_columns[factor_name]
        observed_outcome = torch.tensor([outcome_value], dtype=torch.float32)
        range_low, range_high = factor_spec.range[0]

        if self.dataset.theta.shape[1] == 1:
            # This is the only factor: evaluate log_prob directly on a grid (exact, no sampling).
            grid_tensor = torch.linspace(range_low, range_high, num_grid_points, dtype=torch.float32).unsqueeze(1)
            with torch.no_grad():
                log_probabilities = self.posterior.log_prob(grid_tensor, x=observed_outcome)
            density_numpy = torch.exp(log_probabilities).cpu().numpy()
            grid_numpy = grid_tensor.squeeze(-1).cpu().numpy()
        else:
            # Other factors present: sample, take this factor's column (marginalizing the
            # rest), then histogram-and-interpolate onto the grid.
            with torch.no_grad():
                posterior_samples = self.posterior.sample((10_000,), x=observed_outcome)
            factor_column_samples = posterior_samples[:, factor_column_slice].squeeze(-1).cpu().numpy()
            grid_numpy = np.linspace(range_low, range_high, num_grid_points)
            histogram_density, bin_edges = np.histogram(
                factor_column_samples, bins=40, range=(range_low, range_high), density=True
            )
            density_numpy = np.interp(grid_numpy, 0.5 * (bin_edges[:-1] + bin_edges[1:]), histogram_density)

        return grid_numpy, density_numpy

    def categorical_marginal_probs(self, factor_name: str, outcome_value: float, num_samples: int) -> np.ndarray:
        """Estimate ``P(category | outcome=outcome_value)`` by sampling the trained posterior.

        Returns a length-``num_choices`` numpy array that sums to 1.
        """
        assert self.posterior is not None, "Call fit() before querying the posterior"
        factor_spec = self._factor_spec(factor_name)
        assert factor_spec.type == "categorical"
        assert factor_spec.choices is not None
        factor_column_slice = self.dataset.factor_columns[factor_name]
        num_choices = len(factor_spec.choices)

        observed_outcome = torch.tensor([outcome_value], dtype=torch.float32)
        with torch.no_grad():
            posterior_samples = self.posterior.sample((num_samples,), x=observed_outcome)
        # sbi returns the categorical column as floats over the BoxUniform support; round
        # to the nearest code in [0, num_choices - 1] and tally into normalized frequencies.
        factor_column_samples = posterior_samples[:, factor_column_slice].squeeze(-1).cpu().numpy()
        clipped_codes = np.clip(np.round(factor_column_samples), 0, num_choices - 1).astype(int)
        return np.bincount(clipped_codes, minlength=num_choices) / num_samples


class MNPEAnalyzer(PosteriorAnalyzer):
    """Analyzer for schemas that mix continuous and categorical factors.

    Trains ``sbi.inference.MNPE``, whose mixed density estimator handles continuous and
    categorical theta columns together. Requires at least one continuous factor.
    """

    def _inference_cls(self):
        # Import sbi lazily — it is heavy and only needed once an analysis actually fits.
        from sbi.inference import MNPE

        return MNPE

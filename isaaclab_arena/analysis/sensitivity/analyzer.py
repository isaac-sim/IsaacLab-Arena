# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from isaaclab_arena.analysis.sensitivity.dataset import SensitivityDataset


class SensitivityAnalyzer:
    """Fits a neural posterior over all factors, conditioned on all outcomes (robolab-style).

    Picks the sbi estimator from the schema:

    - MNPE when any factor is categorical (it handles mixed continuous + categorical theta).
    - NPE when every factor is continuous.

    It then trains on the full (theta, x) and samples the joint posterior at a chosen
    observation. The single observation conditions on *all* outcome columns at once, so a
    query like "which factors produced success?" is answered for every factor jointly.

    Continuous factors are normalized to [0, 1] before fitting and denormalized when
    sampling, so factors on very different scales (e.g. light in thousands, an offset in
    hundredths) train on equal footing. Categorical columns keep their integer codes.
    """

    def __init__(self, dataset: SensitivityDataset):
        self.dataset = dataset
        self.posterior = None
        # Continuous factors occupy the leading theta columns (one each); cache their
        # [low, high] bounds for the normalize/denormalize round-trip around sbi.
        continuous_factors = [factor for factor in dataset.schema.factors if factor.type == "continuous"]
        self._num_continuous = len(continuous_factors)
        self._continuous_low = torch.tensor([factor.range[0][0] for factor in continuous_factors])
        self._continuous_high = torch.tensor([factor.range[0][1] for factor in continuous_factors])

    def _inference_cls(self):
        """Return the sbi inference class for this schema (MNPE if categoricals, else NPE)."""
        # Import sbi lazily — it is heavy and only needed once an analysis actually fits.
        from sbi.inference import MNPE, NPE

        return MNPE if self.dataset.has_categorical_factors else NPE

    def _normalized_prior(self):
        """Uniform prior matching the normalized theta: continuous dims [0, 1], categoricals [0, k-1]."""
        from sbi.utils import BoxUniform

        low_bounds = [0.0] * self._num_continuous
        high_bounds = [1.0] * self._num_continuous
        for factor in self.dataset.schema.factors:
            if factor.type == "categorical":
                low_bounds.append(0.0)
                high_bounds.append(float(len(factor.choices) - 1))
        return BoxUniform(low=torch.tensor(low_bounds), high=torch.tensor(high_bounds))

    def _normalize(self, theta: torch.Tensor) -> torch.Tensor:
        """Scale the continuous (leading) theta columns to [0, 1]; leave categoricals untouched."""
        normalized = theta.clone()
        span = (self._continuous_high - self._continuous_low).clamp_min(1e-12)
        normalized[:, : self._num_continuous] = (theta[:, : self._num_continuous] - self._continuous_low) / span
        return normalized

    def _denormalize(self, theta: torch.Tensor) -> torch.Tensor:
        """Inverse of _normalize: map the continuous columns back to their original ranges."""
        denormalized = theta.clone()
        span = self._continuous_high - self._continuous_low
        denormalized[:, : self._num_continuous] = theta[:, : self._num_continuous] * span + self._continuous_low
        return denormalized

    def fit(self, training_batch_size: int = 50) -> None:
        """Train the estimator on the full (theta, x) and store the posterior on self."""
        print(
            f"[INFO] SensitivityAnalyzer: fitting {self._inference_cls().__name__} on"
            f" {self.dataset.num_episodes} episodes"
            f" (theta dim={self.dataset.theta.shape[1]}, x dim={self.dataset.x.shape[1]})."
        )
        inference = self._inference_cls()(prior=self._normalized_prior())
        inference.append_simulations(self._normalize(self.dataset.theta), self.dataset.x)
        density_estimator = inference.train(training_batch_size=training_batch_size)
        self.posterior = inference.build_posterior(density_estimator)

    def default_observation(self) -> torch.Tensor:
        """Default observation to condition on: 1.0 for binary outcomes, the mean otherwise.

        A binary outcome's interesting query is "what produced success?" (condition on 1.0);
        a continuous outcome has no such value, so its mean is the natural "typical case".
        """
        outcome_values = []
        for column_index in range(self.dataset.x.shape[1]):
            column = self.dataset.x[:, column_index]
            is_binary = set(column.tolist()).issubset({0.0, 1.0})
            outcome_values.append(1.0 if is_binary else float(column.mean()))
        return torch.tensor(outcome_values, dtype=torch.float32)

    def sample_posterior(self, observation: torch.Tensor | None = None, num_samples: int = 5000) -> torch.Tensor:
        """Sample the joint posterior over all factors at observation (default: see above).

        Returns a (num_samples, total_factor_dim) tensor laid out like theta — continuous
        columns first (in original, denormalized units), then integer-coded categorical columns.
        """
        assert self.posterior is not None, "Call fit() before sampling the posterior"
        if observation is None:
            observation = self.default_observation()
        with torch.no_grad():
            normalized_samples = self.posterior.sample((num_samples,), x=observation)
        return self._denormalize(normalized_samples)

# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from contextlib import nullcontext
from typing import Literal

from sbi.inference import MNPE, NPE
from sbi.utils import BoxUniform

from isaaclab_arena.analysis.sensitivity.dataset import SensitivityDataset
from isaaclab_arena.analysis.sensitivity.empirical import EmpiricalSensitivityResult, compute_empirical_marginals


class SensitivityAnalyzer:
    """Analyze factor sensitivity empirically or with a fitted posterior.

    ``analyze()`` is the main entry point. It returns empirical marginals from exact outcome
    matches or a tensor of samples from a fitted posterior. ``fit()`` and ``sample_posterior()`` remain
    available for callers that need direct control over the fitted lifecycle.

    Fitted analysis picks the sbi estimator from the schema:

    - MNPE when any factor is categorical (it handles mixed continuous + categorical theta).
    - NPE when every factor is continuous.

    Following sbi's convention, ``theta`` is the per-episode factor values (the inputs the
    posterior is inferred over) and ``x`` is the per-episode outcomes (the observations a query
    conditions on). It trains on the full (theta, x) and samples the joint posterior at a chosen
    observation. The single observation conditions on *all* outcome columns at once, so a
    query like "which factors produced success?" is answered for every factor jointly.

    Continuous factors are normalized to [0, 1] before fitting and denormalized when
    sampling, so factors on very different scales (e.g. light in thousands, an offset in
    hundredths) train on equal footing. Categorical columns keep their integer codes.
    """

    def __init__(self, dataset: SensitivityDataset):
        self.dataset = dataset
        self.posterior = None
        continuous_factors = [factor for factor in dataset.factors if factor.type == "continuous"]
        # theta is laid out continuous-first then categorical — built that way by
        # SensitivityDataset and defined by its factor_columns — so the leading
        # self._num_continuous columns are the continuous factors that _normalize/_denormalize slice.
        self._num_continuous = len(continuous_factors)
        for factor in continuous_factors:
            assert factor.range is not None, (
                f"Continuous factor {factor.name!r} has no range to normalize against. Set a range on"
                " the FactorSpec, or build the dataset via dataset_from_episode_results() so the range is"
                " inferred from the data before constructing the analyzer."
            )
        self._continuous_low = torch.tensor([factor.range[0] for factor in continuous_factors])
        self._continuous_high = torch.tensor([factor.range[1] for factor in continuous_factors])

    def analyze(
        self,
        method: Literal["fitted", "empirical"],
        observation: torch.Tensor | list[float] | tuple[float, ...] | None = None,
        *,
        seed: int | None = 0,
        training_batch_size: int = 50,
        num_posterior_samples: int = 5000,
        num_bins: int = 6,
        num_bootstrap_samples: int = 1000,
        confidence_level: float = 0.95,
    ) -> torch.Tensor | EmpiricalSensitivityResult:
        """Analyze the dataset with an explicitly selected analysis method.

        Fitted analysis trains a new posterior on every call. Call ``fit()`` followed by
        ``sample_posterior()`` directly when reusing one fitted posterior for several observations.

        Args:
            method: ``empirical`` for exact-match marginals or ``fitted`` for NPE/MNPE samples.
            observation: One value per outcome column. None uses the dataset default.
            seed: Seed for fitted inference or empirical bootstrap resampling. None leaves it random.
                A fitted seed is scoped to this call and does not change the caller's random state.
            training_batch_size: Training batch size used only by fitted analysis.
            num_posterior_samples: Number of posterior draws used only by fitted analysis.
            num_bins: Number of continuous bins used only by empirical analysis.
            num_bootstrap_samples: Number of bootstrap resamples used only by empirical analysis.
            confidence_level: Bootstrap confidence level used only by empirical analysis.

        Returns:
            Empirical marginals or fitted posterior samples, depending on ``method``.
        """
        assert method in {"fitted", "empirical"}, f"Unknown sensitivity method {method!r}."
        assert seed is None or seed >= 0, f"seed must be non-negative or None; got {seed}."
        observation_tensor = self.dataset.prepare_observation_tensor(observation)

        if method == "empirical":
            return compute_empirical_marginals(
                self.dataset,
                observation_tensor,
                num_bins=num_bins,
                num_bootstrap_samples=num_bootstrap_samples,
                confidence_level=confidence_level,
                seed=seed,
            )

        assert num_posterior_samples > 0, f"num_posterior_samples must be positive; got {num_posterior_samples}."
        random_state_scope = nullcontext() if seed is None else torch.random.fork_rng(devices=[])
        with random_state_scope:
            if seed is not None:
                torch.random.default_generator.manual_seed(seed)
            self.fit(training_batch_size=training_batch_size)
            posterior_samples = self.sample_posterior(observation_tensor, num_samples=num_posterior_samples)
        return posterior_samples

    def _select_inference_class(self):
        """Choose the sbi inference class for this schema.

        Returns MNPE when any factor is categorical (its mixed density estimator handles
        continuous + categorical theta together), and NPE when every factor is continuous.
        """
        return MNPE if self.dataset.has_categorical_factors else NPE

    def _normalized_prior(self):
        """Uniform prior matching the normalized theta: continuous dims [0, 1], categoricals [0, k-1]."""
        low_bounds = [0.0] * self._num_continuous
        high_bounds = [1.0] * self._num_continuous
        for factor in self.dataset.factors:
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

    def fit(self, training_batch_size: int = 50):
        """Train the estimator on the full (theta, x); store and return the fitted posterior."""
        assert training_batch_size > 0, f"training_batch_size must be positive; got {training_batch_size}."
        print(
            f"[INFO] SensitivityAnalyzer: fitting {self._select_inference_class().__name__} on"
            f" {self.dataset.num_episodes} episodes"
            f" (theta dim={self.dataset.theta.shape[1]}, x dim={self.dataset.x.shape[1]})."
        )
        inference = self._select_inference_class()(prior=self._normalized_prior())
        inference.append_simulations(self._normalize(self.dataset.theta), self.dataset.x)
        density_estimator = inference.train(training_batch_size=training_batch_size)
        self.posterior = inference.build_posterior(density_estimator)
        return self.posterior

    def sample_posterior(
        self,
        observation: torch.Tensor | list[float] | tuple[float, ...] | None = None,
        num_samples: int = 5000,
    ) -> torch.Tensor:
        """Sample the joint posterior over all factors at observation.

        Defaults to the dataset's default observation (condition on success). Returns a
        (num_samples, num_factors) tensor laid out like theta — continuous columns first
        (in original, denormalized units), then integer-coded categorical columns.
        """
        assert self.posterior is not None, "Call fit() before sampling the posterior"
        assert num_samples > 0, f"num_samples must be positive; got {num_samples}."
        observation_tensor = self.dataset.prepare_observation_tensor(observation)
        with torch.no_grad():
            normalized_samples = self.posterior.sample((num_samples,), x=observation_tensor)
        return self._denormalize(normalized_samples)

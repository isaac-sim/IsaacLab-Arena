# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Literal

from isaaclab_arena.analysis.sensitivity.dataset import FactorSpec, FactorType, SensitivityDataset


@dataclass(frozen=True)
class EmpiricalMarginal:
    """The empirical posterior and outcome-rate statistics for one factor."""

    factor_name: str
    """The recorded factor name."""

    bin_edges: np.ndarray | None
    """Continuous bin edges, or None for a categorical factor."""

    bin_labels: tuple[str, ...] | None
    """Categorical bin labels in declared order, or None for a continuous factor."""

    sampled_counts: np.ndarray
    """Episode counts in each bin across the complete dataset."""

    matching_counts: np.ndarray
    """Episode counts in each bin whose full outcome vector matches the observation."""

    sampled_probabilities: np.ndarray
    """Bin probabilities across the complete dataset."""

    posterior_probabilities: np.ndarray
    """Bin probabilities among episodes that match the observation."""

    posterior_to_sampling_ratio: np.ndarray
    """Posterior bin probability divided by the corresponding sampled probability."""

    ratio_confidence_low: np.ndarray
    """Lower percentile-bootstrap bound for the posterior-to-sampling ratio."""

    ratio_confidence_high: np.ndarray
    """Upper percentile-bootstrap bound for the posterior-to-sampling ratio."""

    outcome_match_rates: np.ndarray
    """Fraction of sampled episodes in each bin that match the observation."""

    outcome_rate_confidence_low: np.ndarray
    """Lower percentile-bootstrap bound for each bin's outcome match rate."""

    outcome_rate_confidence_high: np.ndarray
    """Upper percentile-bootstrap bound for each bin's outcome match rate."""


@dataclass(frozen=True)
class EmpiricalSensitivityResult:
    """Empirical sensitivity marginals for one exact outcome observation."""

    observation: torch.Tensor
    """The full outcome vector matched against every episode."""

    num_episodes: int
    """Number of episodes in the complete dataset."""

    num_matching_episodes: int
    """Number of episodes whose full outcome vector exactly matches the observation."""

    overall_match_rate: float
    """Fraction of all episodes that exactly match the observation."""

    confidence_level: float
    """Confidence level used for the percentile-bootstrap intervals."""

    marginals: tuple[EmpiricalMarginal, ...]
    """Per-factor marginals in the dataset's declared factor order."""

    method: Literal["empirical"] = field(default="empirical", init=False)
    """The analysis method represented by this result."""


@dataclass(frozen=True)
class _PreparedFactorBins:
    """Fixed bin metadata and per-episode assignments used by every bootstrap draw."""

    factor: FactorSpec
    """The factor described by these bins."""

    bin_indices: np.ndarray
    """Fixed zero-based bin assignment for every episode."""

    bin_edges: np.ndarray | None
    """Continuous bin edges, or None for a categorical factor."""

    bin_labels: tuple[str, ...] | None
    """Categorical labels, or None for a continuous factor."""

    @property
    def num_bins(self) -> int:
        """Number of fixed bins for this factor."""
        if self.bin_edges is not None:
            return len(self.bin_edges) - 1
        assert self.bin_labels is not None
        return len(self.bin_labels)


def _prepare_factor_bins(dataset: SensitivityDataset, factor: FactorSpec, num_bins: int) -> _PreparedFactorBins:
    """Assign every episode to a fixed continuous bin or declared categorical choice."""
    factor_values = dataset.theta[:, dataset.factor_columns[factor.name]].squeeze(-1).detach().cpu().numpy()

    if factor.type == FactorType.CONTINUOUS:
        assert factor.range is not None, f"Continuous factor {factor.name!r} has no range."
        range_low, range_high = factor.range
        assert (
            range_high > range_low
        ), f"Continuous factor {factor.name!r} must have an increasing range; got {factor.range}."
        bin_edges = np.linspace(range_low, range_high, num_bins + 1, dtype=np.float64)
        # theta is float32 while an inferred or declared range can hold Python floats. Allow only
        # their endpoint-representation difference; a genuinely out-of-range value is a data error.
        range_scale = max(abs(range_low), abs(range_high), range_high - range_low, np.finfo(np.float32).tiny)
        endpoint_tolerance = 8 * np.finfo(np.float32).eps * range_scale
        outside_range = (factor_values < range_low - endpoint_tolerance) | (
            factor_values > range_high + endpoint_tolerance
        )
        assert not outside_range.any(), (
            f"Continuous factor {factor.name!r} contains {int(outside_range.sum())} value(s) outside "
            f"its declared range {factor.range}; observed min/max are "
            f"({float(factor_values.min())}, {float(factor_values.max())})."
        )
        bounded_values = np.clip(factor_values.astype(np.float64), bin_edges[0], bin_edges[-1])
        bin_indices = np.searchsorted(bin_edges, bounded_values, side="right") - 1
        bin_indices = np.clip(bin_indices, 0, num_bins - 1).astype(np.int64)
        return _PreparedFactorBins(factor, bin_indices, bin_edges, None)

    assert (
        factor.choices is not None and len(factor.choices) > 0
    ), f"Categorical factor {factor.name!r} has no declared choices."
    rounded_codes = np.rint(factor_values).astype(np.int64)
    assert np.allclose(factor_values, rounded_codes), f"Categorical factor {factor.name!r} contains non-integer codes."
    assert (
        (rounded_codes >= 0) & (rounded_codes < len(factor.choices))
    ).all(), f"Categorical factor {factor.name!r} contains a code outside its declared choices."
    return _PreparedFactorBins(factor, rounded_codes, None, tuple(factor.choices))


def _probabilities_and_rates(
    sampled_counts: np.ndarray,
    matching_counts: np.ndarray,
    num_episodes: int,
    num_matching_episodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute sampled/posterior probabilities, their ratio, and per-bin outcome rates."""
    sampled_probabilities = sampled_counts.astype(np.float64) / num_episodes
    posterior_probabilities = matching_counts.astype(np.float64) / num_matching_episodes

    posterior_to_sampling_ratio = np.full(sampled_counts.shape, np.nan, dtype=np.float64)
    outcome_match_rates = np.full(sampled_counts.shape, np.nan, dtype=np.float64)
    bins_with_support = sampled_counts > 0
    posterior_to_sampling_ratio[bins_with_support] = (
        posterior_probabilities[bins_with_support] / sampled_probabilities[bins_with_support]
    )
    outcome_match_rates[bins_with_support] = matching_counts[bins_with_support] / sampled_counts[bins_with_support]
    return sampled_probabilities, posterior_probabilities, posterior_to_sampling_ratio, outcome_match_rates


def _bootstrap_confidence_bounds(
    bootstrap_values: np.ndarray, confidence_level: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-bin percentile bounds while retaining NaN for bins without sampled support."""
    lower_quantile = 0.5 * (1.0 - confidence_level)
    upper_quantile = 1.0 - lower_quantile
    confidence_low = np.full(bootstrap_values.shape[1], np.nan, dtype=np.float64)
    confidence_high = np.full(bootstrap_values.shape[1], np.nan, dtype=np.float64)
    for bin_index in range(bootstrap_values.shape[1]):
        finite_values = bootstrap_values[:, bin_index]
        finite_values = finite_values[np.isfinite(finite_values)]
        if len(finite_values) > 0:
            confidence_low[bin_index], confidence_high[bin_index] = np.quantile(
                finite_values, [lower_quantile, upper_quantile]
            )
    return confidence_low, confidence_high


def compute_empirical_marginals(
    dataset: SensitivityDataset,
    observation: torch.Tensor | list[float] | tuple[float, ...] | None = None,
    num_bins: int = 6,
    num_bootstrap_samples: int = 1000,
    confidence_level: float = 0.95,
    seed: int | None = 0,
) -> EmpiricalSensitivityResult:
    """Compute fixed-bin empirical posterior ratios for episodes matching an observation.

    Args:
        dataset: Per-episode factor values and outcomes to analyze.
        observation: Full outcome vector to match exactly. None uses the dataset default.
        num_bins: Number of equal-width bins for every continuous factor.
        num_bootstrap_samples: Number of episode-level bootstrap resamples.
        confidence_level: Percentile-bootstrap confidence level in the open interval (0, 1).
        seed: Seed for a local NumPy random generator. None selects nondeterministic entropy.

    Returns:
        Empirical probabilities, ratios, outcome rates, and confidence intervals per factor.
    """
    assert num_bins > 0, f"num_bins must be positive; got {num_bins}."
    assert num_bootstrap_samples > 0, f"num_bootstrap_samples must be positive; got {num_bootstrap_samples}."
    assert 0.0 < confidence_level < 1.0, f"confidence_level must lie strictly between 0 and 1; got {confidence_level}."

    observation_tensor = dataset.resolve_observation(observation)

    matching_episode_mask = (dataset.x == observation_tensor).all(dim=1)
    num_matching_episodes = int(matching_episode_mask.sum().item())
    assert num_matching_episodes > 0, (
        f"No episodes exactly match observation {observation_tensor.detach().cpu().tolist()} "
        f"for outcomes {dataset.outcome_names}."
    )

    num_episodes = dataset.num_episodes
    matching_episode_mask_numpy = matching_episode_mask.detach().cpu().numpy()
    prepared_factors = tuple(_prepare_factor_bins(dataset, factor, num_bins) for factor in dataset.factors)

    sampled_counts_by_factor: list[np.ndarray] = []
    matching_counts_by_factor: list[np.ndarray] = []
    for prepared_factor in prepared_factors:
        sampled_counts_by_factor.append(np.bincount(prepared_factor.bin_indices, minlength=prepared_factor.num_bins))
        matching_counts_by_factor.append(
            np.bincount(
                prepared_factor.bin_indices[matching_episode_mask_numpy],
                minlength=prepared_factor.num_bins,
            )
        )

    bootstrap_ratios = [
        np.full((num_bootstrap_samples, factor.num_bins), np.nan, dtype=np.float64) for factor in prepared_factors
    ]
    bootstrap_outcome_rates = [
        np.full((num_bootstrap_samples, factor.num_bins), np.nan, dtype=np.float64) for factor in prepared_factors
    ]
    random_generator = np.random.default_rng(seed)
    for bootstrap_index in range(num_bootstrap_samples):
        resampled_episode_indices = random_generator.integers(0, num_episodes, size=num_episodes)
        resampled_matching_mask = matching_episode_mask_numpy[resampled_episode_indices]
        resampled_num_matching_episodes = int(resampled_matching_mask.sum())

        for factor_index, prepared_factor in enumerate(prepared_factors):
            resampled_bin_indices = prepared_factor.bin_indices[resampled_episode_indices]
            resampled_sampled_counts = np.bincount(resampled_bin_indices, minlength=prepared_factor.num_bins)
            resampled_matching_counts = np.bincount(
                resampled_bin_indices[resampled_matching_mask], minlength=prepared_factor.num_bins
            )
            bins_with_support = resampled_sampled_counts > 0
            bootstrap_outcome_rates[factor_index][bootstrap_index, bins_with_support] = (
                resampled_matching_counts[bins_with_support] / resampled_sampled_counts[bins_with_support]
            )
            if resampled_num_matching_episodes > 0:
                resampled_sampled_probabilities = resampled_sampled_counts / num_episodes
                resampled_posterior_probabilities = resampled_matching_counts / resampled_num_matching_episodes
                bootstrap_ratios[factor_index][bootstrap_index, bins_with_support] = (
                    resampled_posterior_probabilities[bins_with_support]
                    / resampled_sampled_probabilities[bins_with_support]
                )

    marginals: list[EmpiricalMarginal] = []
    for factor_index, prepared_factor in enumerate(prepared_factors):
        sampled_counts = sampled_counts_by_factor[factor_index]
        matching_counts = matching_counts_by_factor[factor_index]
        (
            sampled_probabilities,
            posterior_probabilities,
            posterior_to_sampling_ratio,
            outcome_match_rates,
        ) = _probabilities_and_rates(
            sampled_counts,
            matching_counts,
            num_episodes,
            num_matching_episodes,
        )
        ratio_confidence_low, ratio_confidence_high = _bootstrap_confidence_bounds(
            bootstrap_ratios[factor_index], confidence_level
        )
        outcome_rate_confidence_low, outcome_rate_confidence_high = _bootstrap_confidence_bounds(
            bootstrap_outcome_rates[factor_index], confidence_level
        )
        marginals.append(
            EmpiricalMarginal(
                factor_name=prepared_factor.factor.name,
                bin_edges=prepared_factor.bin_edges,
                bin_labels=prepared_factor.bin_labels,
                sampled_counts=sampled_counts,
                matching_counts=matching_counts,
                sampled_probabilities=sampled_probabilities,
                posterior_probabilities=posterior_probabilities,
                posterior_to_sampling_ratio=posterior_to_sampling_ratio,
                ratio_confidence_low=ratio_confidence_low,
                ratio_confidence_high=ratio_confidence_high,
                outcome_match_rates=outcome_match_rates,
                outcome_rate_confidence_low=outcome_rate_confidence_low,
                outcome_rate_confidence_high=outcome_rate_confidence_high,
            )
        )

    return EmpiricalSensitivityResult(
        observation=observation_tensor.detach().cpu().clone(),
        num_episodes=num_episodes,
        num_matching_episodes=num_matching_episodes,
        overall_match_rate=num_matching_episodes / num_episodes,
        confidence_level=confidence_level,
        marginals=tuple(marginals),
    )

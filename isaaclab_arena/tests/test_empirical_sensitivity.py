# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Deterministic regression tests for empirical sensitivity marginals."""

from __future__ import annotations

import numpy as np
import torch

import pytest

from isaaclab_arena.analysis.sensitivity.dataset import FactorSpec, SensitivityDataset
from isaaclab_arena.analysis.sensitivity.empirical import (
    EmpiricalMarginal,
    EmpiricalSensitivityResult,
    compute_empirical_marginals,
)


def _repeat_rows(values: torch.Tensor, repeats: int) -> torch.Tensor:
    """Repeat each row together, preserving factor/outcome pairing."""
    return values.repeat_interleave(repeats, dim=0)


def _continuous_dataset(repeats: int = 20) -> SensitivityDataset:
    """Four equally sampled bins with relative success ratios [0, 1, 2, 1]."""
    factors = [FactorSpec(name="offset", type="continuous", range=(-2.0, 2.0))]
    theta = _repeat_rows(torch.tensor([[-2.0], [-1.5], [-1.0], [-0.5], [0.0], [0.5], [1.0], [2.0]]), repeats)
    success = _repeat_rows(torch.tensor([[0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [1.0]]), repeats)
    return SensitivityDataset(factors, theta, success)


def _paired_bootstrap_dataset(repeats: int = 20) -> SensitivityDataset:
    """Two always-success and two always-failure bins for testing paired resampling."""
    factors = [FactorSpec(name="offset", type="continuous", range=(-2.0, 2.0))]
    theta = _repeat_rows(torch.tensor([[-2.0], [-1.5], [-1.0], [-0.5], [0.0], [0.5], [1.0], [2.0]]), repeats)
    success = _repeat_rows(torch.tensor([[0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0]]), repeats)
    return SensitivityDataset(factors, theta, success)


def test_exact_matching_uses_every_outcome_column():
    """Only episodes exactly matching every requested outcome contribute to the posterior."""
    repeats = 8
    dataset = SensitivityDataset(
        [FactorSpec(name="offset", type="continuous", range=(-2.0, 2.0))],
        _repeat_rows(torch.tensor([[-1.5], [-0.5], [0.5], [1.5]]), repeats),
        _repeat_rows(torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), repeats),
        outcome_names=["success", "collision_free"],
    )

    result = compute_empirical_marginals(
        dataset,
        observation=torch.tensor([1.0, 1.0]),
        num_bins=4,
        num_bootstrap_samples=32,
        seed=3,
    )

    assert isinstance(result, EmpiricalSensitivityResult)
    assert result.num_episodes == 4 * repeats
    assert result.num_matching_episodes == 2 * repeats
    np.testing.assert_array_equal(result.marginals[0].matching_counts, [repeats, 0, 0, repeats])
    np.testing.assert_allclose(result.observation, [1.0, 1.0])


def test_exact_matching_rejects_wrong_observation_shape():
    """An observation must provide exactly one value for every outcome column."""
    dataset = SensitivityDataset(
        [FactorSpec(name="offset", type="continuous", range=(-1.0, 1.0))],
        torch.tensor([[-0.5], [0.5]]),
        torch.tensor([[1.0, 1.0], [0.0, 0.0]]),
        outcome_names=["success", "collision_free"],
    )

    with pytest.raises(AssertionError, match="Observation shape"):
        compute_empirical_marginals(dataset, observation=torch.tensor([1.0]), num_bootstrap_samples=8)


def test_exact_matching_rejects_query_without_matches():
    """An empirical posterior cannot be reported when no episode exactly matches the query."""
    dataset = SensitivityDataset(
        [FactorSpec(name="offset", type="continuous", range=(-1.0, 1.0))],
        torch.tensor([[-0.5], [0.5]]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        outcome_names=["success", "collision_free"],
    )

    with pytest.raises(AssertionError, match="match"):
        compute_empirical_marginals(
            dataset,
            observation=torch.tensor([1.0, 1.0]),
            num_bootstrap_samples=8,
        )


def test_continuous_marginal_uses_declared_edges_for_both_populations():
    """Sampling and matching histograms share fixed declared bounds, including both endpoints."""
    repeats = 20
    result = compute_empirical_marginals(
        _continuous_dataset(repeats),
        num_bins=4,
        num_bootstrap_samples=32,
        seed=5,
    )
    marginal = result.marginals[0]

    assert isinstance(marginal, EmpiricalMarginal)
    assert marginal.bin_labels is None
    np.testing.assert_allclose(marginal.bin_edges, [-2.0, -1.0, 0.0, 1.0, 2.0])
    np.testing.assert_array_equal(marginal.sampled_counts, [2 * repeats] * 4)
    np.testing.assert_array_equal(marginal.matching_counts, [0, repeats, 2 * repeats, repeats])
    np.testing.assert_allclose(marginal.sampled_probabilities, [0.25] * 4)
    np.testing.assert_allclose(marginal.posterior_probabilities, [0.0, 0.25, 0.5, 0.25])


def test_continuous_marginal_rejects_values_outside_declared_range():
    """Declared bounds are enforced instead of silently clipping an invalid factor value."""
    dataset = SensitivityDataset(
        [FactorSpec(name="offset", type="continuous", range=(-1.0, 1.0))],
        torch.tensor([[-1.0], [0.0], [1.1]]),
        torch.tensor([[1.0], [0.0], [1.0]]),
    )

    with pytest.raises(AssertionError, match="outside its declared range"):
        compute_empirical_marginals(dataset, num_bootstrap_samples=8)


def test_probability_ratio_equals_bin_success_rate_relative_to_overall_success():
    """Posterior-to-sampling ratios equal per-bin match rates divided by the overall match rate."""
    result = compute_empirical_marginals(
        _continuous_dataset(),
        num_bins=4,
        num_bootstrap_samples=32,
        seed=7,
    )
    marginal = result.marginals[0]
    sampled_counts = np.asarray(marginal.sampled_counts, dtype=float)
    matching_counts = np.asarray(marginal.matching_counts, dtype=float)
    relative_match_rates = (matching_counts / sampled_counts) / (result.num_matching_episodes / result.num_episodes)

    np.testing.assert_allclose(marginal.posterior_to_sampling_ratio, [0.0, 1.0, 2.0, 1.0])
    np.testing.assert_allclose(marginal.posterior_to_sampling_ratio, relative_match_rates)


def test_bootstrap_resamples_episode_pairs_and_is_reproducible():
    """Paired resampling preserves deterministic bin outcomes and a seed reproduces its intervals."""
    dataset = _paired_bootstrap_dataset()
    arguments = {
        "num_bins": 4,
        "num_bootstrap_samples": 128,
        "confidence_level": 0.9,
        "seed": 11,
    }

    first = compute_empirical_marginals(dataset, **arguments).marginals[0]
    second = compute_empirical_marginals(dataset, **arguments).marginals[0]

    np.testing.assert_array_equal(first.ratio_confidence_low, second.ratio_confidence_low)
    np.testing.assert_array_equal(first.ratio_confidence_high, second.ratio_confidence_high)
    # In every paired resample, both always-success bins have ratio N / num_matches,
    # regardless of how often either bin was drawn. Their bootstrap distributions and
    # therefore their intervals must be identical.
    np.testing.assert_allclose(first.ratio_confidence_low[1], first.ratio_confidence_low[2])
    np.testing.assert_allclose(first.ratio_confidence_high[1], first.ratio_confidence_high[2])
    np.testing.assert_allclose(first.ratio_confidence_low[[0, 3]], 0.0)
    np.testing.assert_allclose(first.ratio_confidence_high[[0, 3]], 0.0)


def test_higher_confidence_level_does_not_narrow_bootstrap_intervals():
    """Using the same seeded bootstrap draws, a 95% interval contains the 80% interval."""
    dataset = _paired_bootstrap_dataset()
    common_arguments = {"num_bins": 4, "num_bootstrap_samples": 192, "seed": 13}

    interval_80 = compute_empirical_marginals(dataset, confidence_level=0.8, **common_arguments).marginals[0]
    interval_95 = compute_empirical_marginals(dataset, confidence_level=0.95, **common_arguments).marginals[0]

    assert np.all(np.asarray(interval_95.ratio_confidence_low) <= np.asarray(interval_80.ratio_confidence_low))
    assert np.all(np.asarray(interval_95.ratio_confidence_high) >= np.asarray(interval_80.ratio_confidence_high))


def test_categorical_ratio_corrects_uneven_sampling_and_keeps_zero_match_choice():
    """Choice ratios use the sampled baseline, and a sampled choice with no matches remains visible."""
    repeats = 20
    choices = ["a", "b", "c"]
    dataset = SensitivityDataset(
        [FactorSpec(name="background", type="categorical", choices=choices)],
        _repeat_rows(torch.tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [2.0], [2.0]]), repeats),
        _repeat_rows(torch.tensor([[1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [0.0], [0.0]]), repeats),
    )

    result = compute_empirical_marginals(
        dataset,
        num_bootstrap_samples=64,
        seed=17,
    )
    marginal = result.marginals[0]

    assert marginal.bin_edges is None
    assert list(marginal.bin_labels) == choices
    np.testing.assert_array_equal(marginal.sampled_counts, [6 * repeats, 2 * repeats, 2 * repeats])
    np.testing.assert_array_equal(marginal.matching_counts, [2 * repeats, 2 * repeats, 0])
    np.testing.assert_allclose(marginal.sampled_probabilities, [0.6, 0.2, 0.2])
    np.testing.assert_allclose(marginal.posterior_probabilities, [0.5, 0.5, 0.0])
    np.testing.assert_allclose(marginal.posterior_to_sampling_ratio, [5.0 / 6.0, 2.5, 0.0])

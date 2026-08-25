# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the sensitivity-analysis facade."""

from __future__ import annotations

import numpy as np
import torch

import pytest

import isaaclab_arena.analysis.sensitivity.analyzer as analyzer_module
from isaaclab_arena.analysis.sensitivity.analyzer import SensitivityAnalyzer
from isaaclab_arena.analysis.sensitivity.dataset import FactorSpec, SensitivityDataset
from isaaclab_arena.analysis.sensitivity.empirical import compute_empirical_marginals


def _continuous_dataset() -> SensitivityDataset:
    """Build a small continuous dataset with two successful episodes."""
    return SensitivityDataset(
        [FactorSpec(name="offset", type="continuous", range=(-1.0, 1.0))],
        torch.tensor([[-0.75], [-0.25], [0.25], [0.75]]),
        torch.tensor([[0.0], [1.0], [1.0], [0.0]]),
    )


def test_analyze_empirical_delegates_without_fitting(monkeypatch):
    """Empirical analysis delegates all options and does not touch fitted state."""
    dataset = _continuous_dataset()
    analyzer = SensitivityAnalyzer(dataset)
    expected_result = object()
    recorded_arguments = {}

    def record_empirical_call(
        received_dataset,
        observation,
        *,
        num_bins,
        num_bootstrap_samples,
        confidence_level,
        seed,
    ):
        recorded_arguments.update(
            dataset=received_dataset,
            observation=observation,
            num_bins=num_bins,
            num_bootstrap_samples=num_bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed,
        )
        return expected_result

    def fail_if_fitted(*args, **kwargs):
        raise AssertionError("Empirical analysis must not fit NPE or MNPE")

    monkeypatch.setattr(analyzer_module, "compute_empirical_marginals", record_empirical_call)
    monkeypatch.setattr(analyzer, "fit", fail_if_fitted)

    result = analyzer.analyze(
        method="empirical",
        observation=[1.0],
        seed=19,
        num_bins=3,
        num_bootstrap_samples=17,
        confidence_level=0.9,
    )

    assert result is expected_result
    assert recorded_arguments["dataset"] is dataset
    torch.testing.assert_close(recorded_arguments["observation"], torch.tensor([1.0]))
    assert recorded_arguments["num_bins"] == 3
    assert recorded_arguments["num_bootstrap_samples"] == 17
    assert recorded_arguments["confidence_level"] == 0.9
    assert recorded_arguments["seed"] == 19
    assert analyzer.posterior is None


def test_analyze_empirical_matches_direct_computation():
    """The facade preserves the pure empirical calculation and result type."""
    dataset = _continuous_dataset()
    arguments = {
        "observation": [1.0],
        "seed": 23,
        "num_bins": 2,
        "num_bootstrap_samples": 32,
        "confidence_level": 0.9,
    }

    direct_result = compute_empirical_marginals(dataset, **arguments)
    facade_result = SensitivityAnalyzer(dataset).analyze(method="empirical", **arguments)

    assert facade_result.num_episodes == direct_result.num_episodes
    assert facade_result.num_matching_episodes == direct_result.num_matching_episodes
    torch.testing.assert_close(facade_result.observation, direct_result.observation)
    for facade_marginal, direct_marginal in zip(facade_result.marginals, direct_result.marginals):
        np.testing.assert_array_equal(facade_marginal.sampled_counts, direct_marginal.sampled_counts)
        np.testing.assert_array_equal(facade_marginal.matching_counts, direct_marginal.matching_counts)
        np.testing.assert_allclose(
            facade_marginal.posterior_to_sampling_ratio,
            direct_marginal.posterior_to_sampling_ratio,
        )
        np.testing.assert_allclose(facade_marginal.ratio_confidence_low, direct_marginal.ratio_confidence_low)
        np.testing.assert_allclose(facade_marginal.ratio_confidence_high, direct_marginal.ratio_confidence_high)


def test_analyze_fitted_uses_existing_lifecycle(monkeypatch):
    """Fitted analysis trains and samples through the existing API."""
    dataset = _continuous_dataset()
    analyzer = SensitivityAnalyzer(dataset)
    fit_batch_sizes = []
    sampled_queries = []
    expected_samples = torch.tensor([[-0.4], [0.2], [0.8]])

    def record_fit(training_batch_size=50):
        fit_batch_sizes.append(training_batch_size)
        analyzer.posterior = object()
        return analyzer.posterior

    def record_sample(observation=None, num_samples=5000):
        sampled_queries.append((observation.clone(), num_samples))
        return expected_samples

    monkeypatch.setattr(analyzer, "fit", record_fit)
    monkeypatch.setattr(analyzer, "sample_posterior", record_sample)

    first_samples = analyzer.analyze(
        method="fitted",
        observation=[1.0],
        seed=None,
        training_batch_size=7,
        num_posterior_samples=3,
    )
    second_samples = analyzer.analyze(
        method="fitted",
        observation=[0.0],
        seed=None,
        training_batch_size=11,
        num_posterior_samples=3,
    )

    torch.testing.assert_close(first_samples, expected_samples)
    torch.testing.assert_close(second_samples, expected_samples)
    assert fit_batch_sizes == [7, 11]
    assert [num_samples for _, num_samples in sampled_queries] == [3, 3]
    torch.testing.assert_close(sampled_queries[1][0], torch.tensor([0.0]))


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"training_batch_size": 0}, "training_batch_size"),
        ({"num_posterior_samples": 0}, "num_posterior_samples"),
    ],
)
def test_analyze_fitted_rejects_non_positive_counts(arguments, message):
    """Fitted analysis rejects invalid training and sampling counts before fitting."""
    with pytest.raises(AssertionError, match=message):
        SensitivityAnalyzer(_continuous_dataset()).analyze(method="fitted", seed=None, **arguments)


def test_analyze_fitted_seed_does_not_change_global_random_state(monkeypatch):
    """A reproducible fitted call does not perturb random draws made by its caller."""
    analyzer = SensitivityAnalyzer(_continuous_dataset())

    def record_fit(training_batch_size):
        analyzer.posterior = object()

    def return_samples(observation, num_samples):
        return torch.zeros(num_samples, 1)

    monkeypatch.setattr(analyzer, "fit", record_fit)
    monkeypatch.setattr(analyzer, "sample_posterior", return_samples)

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(29)
        expected_next_draw = torch.rand(1)
        torch.random.default_generator.manual_seed(29)

        analyzer.analyze(method="fitted", seed=7, num_posterior_samples=2)

        torch.testing.assert_close(torch.rand(1), expected_next_draw)


def test_direct_fitted_lifecycle_rejects_non_positive_counts():
    """Direct fitted calls enforce the same count constraints as the facade."""
    analyzer = SensitivityAnalyzer(_continuous_dataset())

    with pytest.raises(AssertionError, match="training_batch_size"):
        analyzer.fit(training_batch_size=0)

    analyzer.posterior = object()
    with pytest.raises(AssertionError, match="num_samples"):
        analyzer.sample_posterior(num_samples=0)


def test_analyze_rejects_unknown_method():
    """Callers must choose one of the two supported analysis methods."""
    with pytest.raises(AssertionError, match="Unknown sensitivity method"):
        SensitivityAnalyzer(_continuous_dataset()).analyze(method="unknown")


@pytest.mark.parametrize("method", ["empirical", "fitted"])
def test_analyze_rejects_negative_seed(method):
    """Both analysis methods use the same non-negative seed contract."""
    with pytest.raises(AssertionError, match="seed"):
        SensitivityAnalyzer(_continuous_dataset()).analyze(method=method, seed=-1)


def test_prepare_observation_tensor_rejects_non_finite_values():
    """Observation preparation rejects a query that cannot represent an outcome value."""
    with pytest.raises(AssertionError, match="finite"):
        _continuous_dataset().prepare_observation_tensor([float("nan")])

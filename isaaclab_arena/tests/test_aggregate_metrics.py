# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import pytest

from isaaclab_arena.metrics.aggregate_metrics import aggregate_metrics
from isaaclab_arena.metrics.metric_data import MetricData, MetricsDataCollection
from isaaclab_arena.metrics.metric_term_cfg import MetricTermCfg


def _mean_metric(recorded_data: list[np.ndarray], **_) -> float:
    """Mean over the concatenated per-episode recorded data."""
    return float(np.mean(np.concatenate(recorded_data)))


def _scaled_count(recorded_data: list[np.ndarray], scale: float = 1.0) -> float:
    """Number of recorded episodes, scaled by ``scale`` (exercises params forwarding)."""
    return scale * len(recorded_data)


def _make_collection(
    num_episodes: int, success_data: list[np.ndarray], count_data: list[np.ndarray]
) -> MetricsDataCollection:
    success_cfg = MetricTermCfg(compute_metric_func=_mean_metric, params={}, recorder_term_name="success")
    count_cfg = MetricTermCfg(compute_metric_func=_scaled_count, params={"scale": 2.0}, recorder_term_name="count")
    return MetricsDataCollection(
        num_episodes=num_episodes,
        metric_data_entries={
            "success_rate": MetricData("success_rate", success_cfg, success_data, _mean_metric(success_data)),
            "count_metric": MetricData("count_metric", count_cfg, count_data, _scaled_count(count_data, scale=2.0)),
        },
    )


def test_aggregate_metrics_accumulates_and_recomputes():
    """Recorded data is concatenated across runs and metric values are recomputed from the whole."""
    # Run 1 success rate = mean([1, 0]) = 0.5, Run 2 success rate = mean([1, 1]) = 1.0.
    run1 = _make_collection(2, [np.array([1.0]), np.array([0.0])], [np.array([0]), np.array([1])])
    run2 = _make_collection(2, [np.array([1.0]), np.array([1.0])], [np.array([2])])

    aggregated = aggregate_metrics([run1, run2])

    # num_episodes is summed across runs.
    assert aggregated.num_episodes == 4

    entries = aggregated.metric_data_entries
    assert set(entries) == {"success_rate", "count_metric"}

    # Recorded per-episode data is concatenated across runs (data accumulation).
    success_entry = entries["success_rate"]
    assert len(success_entry.recorded_data) == 4
    np.testing.assert_array_equal(np.concatenate(success_entry.recorded_data), np.array([1.0, 0.0, 1.0, 1.0]))

    # Metric is recomputed from the full data (0.75), not averaged from the per-run values (0.5, 1.0).
    assert success_entry.metric_value == 0.75

    # Second metric: 2 + 1 = 3 episodes, with params (scale=2.0) forwarded -> 6.0.
    count_entry = entries["count_metric"]
    assert len(count_entry.recorded_data) == 3
    assert count_entry.metric_value == 6.0


def test_aggregate_metrics_single_run_is_identity():
    """A single run aggregates to the same episode count and recomputed values."""
    run = _make_collection(3, [np.array([1.0]), np.array([0.0]), np.array([1.0])], [np.array([0])])

    aggregated = aggregate_metrics([run])

    assert aggregated.num_episodes == 3
    entries = aggregated.metric_data_entries
    assert entries["success_rate"].metric_value == pytest.approx(2.0 / 3.0)


def test_aggregate_metrics_requires_consistent_metric_names():
    """Runs must expose the same set of metric names."""
    run1 = _make_collection(1, [np.array([1.0])], [np.array([0])])
    mismatched_cfg = MetricTermCfg(compute_metric_func=_mean_metric, params={}, recorder_term_name="other")
    run2 = MetricsDataCollection(
        num_episodes=1,
        metric_data_entries={"other_metric": MetricData("other_metric", mismatched_cfg, [np.array([1.0])], 1.0)},
    )

    with pytest.raises(AssertionError):
        aggregate_metrics([run1, run2])


def test_aggregate_metrics_requires_non_empty_input():
    """Aggregating zero runs is an error."""
    with pytest.raises(AssertionError):
        aggregate_metrics([])

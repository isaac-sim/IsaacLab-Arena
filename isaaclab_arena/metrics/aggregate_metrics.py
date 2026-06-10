# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
from typing import Any

from isaaclab_arena.metrics.metric_data import MetricData, MetricsDataCollection


def aggregate_metrics(metrics_per_run: list[MetricsDataCollection]) -> MetricsDataCollection:
    """Aggregate metrics across multiple runs into a single ``MetricsDataCollection``.

    Each run records per-episode data for the same set of metric terms. This function
    concatenates the recorded data across runs and recomputes each metric value from the
    combined data, so the aggregated metric reflects all episodes from all runs.

    Args:
        metrics_per_run: One ``MetricsDataCollection`` per run. Every run must expose the
            same set of metric ``term_name``s.

    Returns:
        A ``MetricsDataCollection`` whose ``num_episodes`` is the sum across runs and whose
        metric values are recomputed from the concatenated recorded data.
    """
    assert len(metrics_per_run) > 0, "metrics_per_run must contain at least one run"

    # All runs must expose the same set of metric names.
    metric_names = [metric_data.term_name for metric_data in metrics_per_run[0].metric_data_entries]
    for metrics_data_collection in metrics_per_run:
        for metric_data in metrics_data_collection.metric_data_entries:
            assert metric_data.term_name in metric_names, f"Metric {metric_data.term_name} not found in all runs"

    total_num_episodes = sum(metrics_data_collection.num_episodes for metrics_data_collection in metrics_per_run)

    # Concatenate the recorded per-episode data across runs, for each metric name.
    metric_name_to_aggregated_data: dict[str, list[np.ndarray]] = {}
    for metrics_data_collection in metrics_per_run:
        for metric_data in metrics_data_collection.metric_data_entries:
            metric_name_to_aggregated_data.setdefault(metric_data.term_name, []).extend(metric_data.recorded_data)

    # Recompute each metric value from the concatenated recorded data, reusing the term config.
    metric_cfgs = {
        metric_data.term_name: metric_data.term_cfg for metric_data in metrics_per_run[0].metric_data_entries
    }
    metric_name_to_aggregated_metric_value: dict[str, Any] = {}
    for metric_name, recorded_data in metric_name_to_aggregated_data.items():
        metric_cfg = metric_cfgs[metric_name]
        metric_name_to_aggregated_metric_value[metric_name] = metric_cfg.compute_metric_func(
            recorded_data, **metric_cfg.params
        )

    # Assemble a new MetricsDataCollection with the aggregated metric values.
    metric_data_entries: list[MetricData] = []
    for metric_name in metric_names:
        metric_data_entries.append(
            MetricData(
                term_name=metric_name,
                term_cfg=metric_cfgs[metric_name],
                recorded_data=metric_name_to_aggregated_data[metric_name],
                metric_value=metric_name_to_aggregated_metric_value[metric_name],
            )
        )

    return MetricsDataCollection(num_episodes=total_num_episodes, metric_data_entries=metric_data_entries)

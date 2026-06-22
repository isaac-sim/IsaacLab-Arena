# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from isaaclab_arena.metrics.metric_term_cfg import MetricTermCfg


@dataclass
class MetricData:
    """Data entry for a metric."""

    term_name: str
    """The name of the metric."""

    term_cfg: MetricTermCfg
    """The configuration for the metric."""

    recorded_data: list[np.ndarray]
    """The recorded data for the metric, one array per simulated episode."""

    metric_value: float | list[float]
    """The computed value of the metric."""


@dataclass
class MetricsDataCollection:
    """Collection of metric data entries."""

    num_episodes: int
    """The number of episodes in the collection."""

    metric_data_entries: dict[str, MetricData]
    """The metric data entries, keyed by metric term name."""

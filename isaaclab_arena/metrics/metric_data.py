# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from isaaclab_arena.metrics.metric_term_cfg import MetricTermCfg


@dataclass
class MetricData:
    term_name: str
    term_cfg: MetricTermCfg
    recorded_data: Any
    metric_value: Any


@dataclass
class MetricsDataCollection:
    num_episodes: int
    metric_data_entries: list[MetricData]

# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
import numpy as np

from isaaclab_arena.metrics.metric_data import MetricData, MetricsDataCollection
from isaaclab_arena.metrics.metric_term_cfg import MetricTermCfg
from isaaclab_arena.metrics.metrics_logger import MetricsLogger


def _unused_metric(_recorded_data):
    return 0.0


def test_save_metrics_to_file_creates_parent_and_serializes_metrics(tmp_path):
    """Metrics are persisted as plain JSON values in a newly created run directory."""
    metric_cfg = MetricTermCfg(compute_metric_func=_unused_metric, params={}, recorder_term_name="success")
    metrics = MetricsDataCollection(
        num_episodes=2,
        metric_data_entries={
            "success_rate": MetricData("success_rate", metric_cfg, [], np.float32(0.5)),
            "episode_lengths": MetricData("episode_lengths", metric_cfg, [], np.array([10, 20])),
        },
    )
    metrics_path = tmp_path / "2026-07-03_12-00-00" / "metrics.json"
    logger = MetricsLogger(metrics_path)

    logger.append_job_metrics("pick_and_place", metrics)
    logger.save_metrics_to_file()

    assert json.loads(metrics_path.read_text(encoding="utf-8")) == {
        "pick_and_place": {
            "num_episodes": 2,
            "success_rate": 0.5,
            "episode_lengths": [10, 20],
        }
    }

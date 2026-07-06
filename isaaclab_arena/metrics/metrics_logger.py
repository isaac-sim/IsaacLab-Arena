# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_arena.metrics.metric_data import MetricsDataCollection


def metrics_to_plain_python_types(metrics_data: MetricsDataCollection) -> dict[str, int | float | list]:
    """Convert numpy scalars/arrays in a metrics data to plain Python types."""
    sanitized = {
        "num_episodes": metrics_data.num_episodes,
    }
    for name, metric_data in metrics_data.metric_data_entries.items():
        value = metric_data.metric_value
        if isinstance(value, np.bool_):
            sanitized[name] = bool(value)
        elif isinstance(value, np.floating):
            sanitized[name] = float(value)
        elif isinstance(value, np.integer):
            sanitized[name] = int(value)
        elif isinstance(value, np.ndarray):
            sanitized[name] = value.tolist()
        else:
            sanitized[name] = value
    return sanitized


class MetricsLogger:
    def __init__(self, metrics_file: str = "metrics.json"):
        self.metrics_file = metrics_file
        self.metrics_data = {}

    def append_job_metrics(self, job_name: str, metrics_data: MetricsDataCollection):
        """Add or update metrics for a specific job.

        Args:
            job_name: Name of the job
            metrics_data: MetricsDataCollection instance containing the data for all metrics.
        """
        if job_name not in self.metrics_data:
            self.metrics_data[job_name] = {}
        self.metrics_data[job_name].update(metrics_to_plain_python_types(metrics_data))

    def save_metrics_to_file(self):
        """Save all metrics to JSON file."""
        with open(self.metrics_file, "w", encoding="utf-8") as f:
            json.dump(self.metrics_data, f, indent=2)

    def print_metrics(self):
        """Print all metrics in a clean, readable format."""
        if not self.metrics_data:
            print("No metrics to display")
            return

        print("\n" + "=" * 70)
        print("METRICS SUMMARY")
        print("=" * 70)

        for job_name, metrics in sorted(self.metrics_data.items()):
            print(f"\n{job_name}:")
            if metrics:
                for metric_name, value in sorted(metrics.items()):
                    if isinstance(value, float):
                        print(f"  {metric_name:<30} {value:>10.4f}")
                    else:
                        # Support list and other types by formatting their string representation
                        value_str = str(value) if not isinstance(value, str) else value
                        print(f"  {metric_name:<30} {value_str:>10}")
            else:
                print("  (no metrics available)")

        print("=" * 70 + "\n")

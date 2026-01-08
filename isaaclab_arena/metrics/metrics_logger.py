# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json

class MetricsLogger:
    def __init__(self, metrics_file: str = "metrics.json"):
        self.metrics_file = metrics_file
        self.metrics_data = {}

    def append_job_metrics(self, job_name: str, metrics: dict[str, float]):
        """Add or update metrics for a specific job.

        Args:
            job_name: Name of the job
            metrics: Dictionary of metric_name -> value
        """
        if job_name not in self.metrics_data:
            self.metrics_data[job_name] = {}
        self.metrics_data[job_name].update(metrics)

    def save_metrics_to_file(self):
        """Save all metrics to JSON file."""
        with open(self.metrics_file, "w", encoding="utf-8") as f:
            json.dump(self.metrics_data, f, indent=2)

    def print_metrics(self):
        """Print all metrics in a clean, readable format."""
        if not self.metrics_data:
            print("No metrics to display")
            return

        print("\n" + "="*70)
        print("METRICS SUMMARY")
        print("="*70)

        for job_name, metrics in sorted(self.metrics_data.items()):
            print(f"\n{job_name}:")
            if metrics:
                for metric_name, value in sorted(metrics.items()):
                    if isinstance(value, float):
                        print(f"  {metric_name:<30} {value:>10.4f}")
                    else:
                        print(f"  {metric_name:<30} {value:>10}")
            else:
                print("  (no metrics available)")

        print("="*70 + "\n")


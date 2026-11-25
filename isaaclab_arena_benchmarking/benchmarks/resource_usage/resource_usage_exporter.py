# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Export the resource usage of a finished Arena Experiment as InfluxDB KPI records.

The records file is ``kpis_resource_usage.json``. It holds one ``resource_usage`` record per tracked resource that was sampled,
with the statistics totalled over every Run, timestamped with the latest Run end and tagged with the Arena git commit
SHA, branch and benchmark id (the Experiment name).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from isaaclab_arena.evaluation.arena_experiment_result import (
    ARENA_AGGREGATED_RESOURCE_USAGE_FILENAME,
    ArenaExperimentResult,
)
from isaaclab_arena.utils.compute_resource.compute_resource_usage import RESOURCE_UNITS
from isaaclab_arena.utils.git_stamp import GitStamp
from isaaclab_arena_benchmarking.benchmarks.registry import register_kpi_exporter
from isaaclab_arena_benchmarking.utils.influxdb_kpi_records import InfluxdbKpiExporter, InfluxdbKpiRecord

RESOURCE_USAGE_RECORDS_FILENAME = "kpis_resource_usage.json"
"""Name of the resource KPI records file in an Experiment output directory."""


@dataclass(kw_only=True)
class MegabyteResourceUsageRecord(InfluxdbKpiRecord):
    """Statistics of one resource measured in megabytes, totalled over every Run."""

    record_type: ClassVar[str] = "resource_usage"

    name: str
    """Resource name."""

    count: int
    """Number of samples."""

    min_mb: float
    """Smallest sample in megabytes."""

    max_mb: float
    """Largest sample in megabytes."""

    mean_mb: float
    """Mean of the samples in megabytes."""

    p10_mb: float | None = None
    """Approximate 10th percentile in megabytes."""

    p50_mb: float | None = None
    """Approximate median in megabytes."""

    p90_mb: float | None = None
    """Approximate 90th percentile in megabytes."""


@dataclass(kw_only=True)
class PercentResourceUsageRecord(InfluxdbKpiRecord):
    """Statistics of one resource measured in percent, totalled over every Run."""

    record_type: ClassVar[str] = "resource_usage"

    name: str
    """Resource name."""

    count: int
    """Number of samples."""

    min_percent: float
    """Smallest sample in percent."""

    max_percent: float
    """Largest sample in percent."""

    mean_percent: float
    """Mean of the samples in percent."""

    p10_percent: float | None = None
    """Approximate 10th percentile in percent."""

    p50_percent: float | None = None
    """Approximate median in percent."""

    p90_percent: float | None = None
    """Approximate 90th percentile in percent."""


def build_resource_usage_record(
    total: dict[str, Any], *, timestamp_ns: int, git_stamp: GitStamp, benchmark_id: str
) -> InfluxdbKpiRecord:
    """Return the record of one Experiment resource usage total; the entry must have a nonzero count.

    Args:
        total: Resource statistics with ``name``, ``count`` and ``min_<u>``, ``max_<u>`` and ``mean_<u>``, and
            optionally ``p10_<u>``, ``p50_<u>`` and ``p90_<u>``, where ``<u>`` is the unit of the resource.
        timestamp_ns: Point time in Unix epoch nanoseconds (UTC).
        git_stamp: Arena git commit and branch the Experiment ran with.
        benchmark_id: Benchmark the Experiment belongs to.

    Returns:
        A megabyte or percent resource record, by the unit of the resource.
    """
    unit = RESOURCE_UNITS[total["name"]]
    record_type = MegabyteResourceUsageRecord if unit == "mb" else PercentResourceUsageRecord
    statistics = {
        f"{statistic}_{unit}": total.get(f"{statistic}_{unit}")
        for statistic in ("min", "max", "mean", "p10", "p50", "p90")
    }
    return record_type(
        timestamp_ns=timestamp_ns,
        git_sha=git_stamp.git_sha,
        git_branch=git_stamp.git_branch,
        benchmark_id=benchmark_id,
        name=total["name"],
        count=int(total["count"]),
        **statistics,
    )


@register_kpi_exporter("resource_usage")
class ResourceUsageKpiExporter(InfluxdbKpiExporter):
    """Exports the resource usage of a finished Arena Experiment output directory."""

    record_types: ClassVar[tuple[str, ...]] = (MegabyteResourceUsageRecord.record_type,)

    def get_records_filename(self) -> str:
        return RESOURCE_USAGE_RECORDS_FILENAME

    def build_records_from_output_dir(self, output_dir: Path) -> list[InfluxdbKpiRecord]:
        """Build the resource records of a finished Experiment output directory.

        Reads ``arena_experiment_result.json`` for the timestamp, git stamp and Experiment name and
        ``arena_experiment_resource_usage.json`` for the resource usage totals.

        Args:
            output_dir: Experiment output root holding the result file and, when tracked, the resources file.

        Returns:
            One record per resource that was sampled, from the resources totals; empty when the resources file is
            absent.
        """
        output_dir = Path(output_dir)
        resource_usage_path = output_dir / ARENA_AGGREGATED_RESOURCE_USAGE_FILENAME
        if not resource_usage_path.is_file():
            return []
        result = ArenaExperimentResult.from_experiment_output(output_dir)
        resources = json.loads(resource_usage_path.read_text(encoding="utf-8"))
        # Resources that were never sampled have no statistics.
        records: list[InfluxdbKpiRecord] = [
            build_resource_usage_record(
                total,
                timestamp_ns=result.latest_run_end_timestamp_ns,
                git_stamp=result.git_stamp,
                benchmark_id=result.experiment_name,
            )
            for total in resources["totals"]
            if total["count"] > 0
        ]
        return records

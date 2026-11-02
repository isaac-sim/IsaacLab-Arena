# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Export the timer statistics of a finished Arena Experiment as InfluxDB KPI records.

The records file is ``kpis_timing.json``. It holds one ``timing`` record per timer that ran, with the statistics
totalled over every Run, timestamped with the latest Run end and tagged with the Arena git commit SHA and branch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from isaaclab_arena.evaluation.arena_experiment_result import (
    ARENA_AGGREGATED_TIMINGS_FILENAME,
    get_git_stamp,
    get_latest_run_end_timestamp_ns,
    read_experiment_result,
)
from isaaclab_arena.utils.git_stamp import GitStamp
from isaaclab_arena_benchmarking.utils.influxdb_kpi_records import (
    InfluxdbKpiExporter,
    InfluxdbKpiRecord,
    assert_unique_tag_sets,
)

TIMING_RECORDS_FILENAME = "kpis_timing.json"
"""Name of the timing KPI records file in an Experiment output directory."""


@dataclass(kw_only=True)
class TotalTimingRecord(InfluxdbKpiRecord):
    """Statistics of one timer, totalled over every Run."""

    record_type: ClassVar[str] = "timing"

    name: str
    """Timer name."""

    mean_ms: float
    """Mean duration in milliseconds."""

    total_ms: float
    """Total duration in milliseconds."""

    min_ms: float
    """Shortest duration in milliseconds."""

    max_ms: float
    """Longest duration in milliseconds."""

    count: int
    """Number of timed intervals."""


def build_timing_record(timing: dict[str, Any], *, timestamp_ns: int, git_stamp: GitStamp) -> TotalTimingRecord:
    """Return the record of one Experiment timings total; the entry must have a nonzero count.

    Args:
        timing: Timer statistics with ``name``, ``count``, ``total_ms``, ``min_ms`` and ``max_ms``.
        timestamp_ns: Point time in Unix epoch nanoseconds (UTC).
        git_stamp: Arena git commit and branch the Experiment ran with.

    Returns:
        The timing record, with the mean computed from the total and count.
    """
    count = int(timing["count"])
    total_ms = float(timing["total_ms"])
    return TotalTimingRecord(
        timestamp_ns=timestamp_ns,
        git_sha=git_stamp.git_sha,
        git_branch=git_stamp.git_branch,
        name=timing["name"],
        mean_ms=total_ms / count,
        total_ms=total_ms,
        min_ms=float(timing["min_ms"]),
        max_ms=float(timing["max_ms"]),
        count=count,
    )


class TimingKpiExporter(InfluxdbKpiExporter):
    """Exports the timer statistics of a finished Arena Experiment output directory."""

    def get_records_filename(self) -> str:
        return TIMING_RECORDS_FILENAME

    def build_records_from_output_dir(self, output_dir: Path) -> list[InfluxdbKpiRecord]:
        """Build the timing records of a finished Experiment output directory.

        Reads ``arena_experiment_result.json`` for the timestamp and git stamp and
        ``arena_experiment_timings.json`` for the timers.

        Args:
            output_dir: Experiment output root holding the result file and, when timed, the timings file.

        Returns:
            One record per timer that ran, from the timings totals; empty when the timings file is absent.
        """
        output_dir = Path(output_dir)
        timings_path = output_dir / ARENA_AGGREGATED_TIMINGS_FILENAME
        if not timings_path.is_file():
            return []
        result = read_experiment_result(output_dir)
        timestamp_ns = get_latest_run_end_timestamp_ns(result)
        git_stamp = get_git_stamp(result)
        timings = json.loads(timings_path.read_text(encoding="utf-8"))
        # Timers that never ran have no statistics.
        records: list[InfluxdbKpiRecord] = [
            build_timing_record(timing, timestamp_ns=timestamp_ns, git_stamp=git_stamp)
            for timing in timings["totals"]
            if timing["count"] > 0
        ]
        assert_unique_tag_sets(records)
        return records

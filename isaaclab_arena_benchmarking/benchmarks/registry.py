# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Registry of the KPI exporters and the function that runs them on a finished Arena Experiment."""

from __future__ import annotations

import traceback
from collections.abc import Sequence
from pathlib import Path

from isaaclab_arena.evaluation.arena_experiment_result import ArenaExperimentResult
from isaaclab_arena_benchmarking.benchmarks.success_rate.success_rate_exporter import SuccessRateKpiExporter
from isaaclab_arena_benchmarking.benchmarks.timing.timing_exporter import TimingKpiExporter
from isaaclab_arena_benchmarking.utils.influxdb_kpi_records import InfluxdbKpiExporter

EXPORTERS_BY_NAME: dict[str, type[InfluxdbKpiExporter]] = {
    "success_rate": SuccessRateKpiExporter,
    "timing": TimingKpiExporter,
}
"""KPI exporter classes by the name used to select them."""

DEFAULT_EXPORTER_NAMES: tuple[str, ...] = ("success_rate", "timing")
"""Names of the exporters that run when no selection is given."""


def export_kpis(
    output_dir: Path,
    exporter_names: Sequence[str] = DEFAULT_EXPORTER_NAMES,
    *,
    raise_on_exporter_failure: bool = True,
) -> list[Path]:
    """Write the KPI records files of a finished Experiment, unless a Run failed.

    Every selected exporter runs even if an earlier one raises. When any Run failed, the reason is printed and no
    file is written.

    Args:
        output_dir: Experiment output root holding ``arena_experiment_result.json``.
        exporter_names: Names from ``EXPORTERS_BY_NAME`` of the exporters to run; no two may share a record type.
        raise_on_exporter_failure: Whether to re-raise the first exporter exception after every exporter ran. When
            False, each failure is printed with its exporter name instead.

    Returns:
        Paths of the records files actually written; empty when a Run failed. An exporter with no records writes none.
    """
    unknown_names = [name for name in exporter_names if name not in EXPORTERS_BY_NAME]
    assert not unknown_names, f"Unknown KPI exporters {unknown_names}; valid names are {sorted(EXPORTERS_BY_NAME)}"
    exporters = {name: EXPORTERS_BY_NAME[name]() for name in exporter_names}

    exporter_name_by_record_type: dict[str, str] = {}
    for name, exporter in exporters.items():
        for record_type in exporter.record_types:
            assert record_type not in exporter_name_by_record_type, (
                f"KPI exporters {exporter_name_by_record_type[record_type]!r} and {name!r} both use the record type"
                f" {record_type!r}"
            )
            exporter_name_by_record_type[record_type] = name

    output_dir = Path(output_dir)
    result = ArenaExperimentResult.read(output_dir)
    if not result.all_runs_completed:
        failed_run_names = [
            run_name for run_name, run_result in result.runs.items() if run_result["status"] != "completed"
        ]
        print(f"Skipping KPI export because these Runs did not complete: {', '.join(failed_run_names)}")
        return []

    written_paths: list[Path] = []
    failures: list[tuple[str, Exception]] = []
    for name, exporter in exporters.items():
        try:
            written_path = exporter.export_to_json(output_dir)
        except Exception as error:
            failures.append((name, error))
            continue
        if written_path is not None:
            written_paths.append(written_path)
    # Print every failure before any re-raise, so a later failure is not lost behind the first.
    for name, error in failures:
        print(f"KPI exporter {name!r} failed: {type(error).__name__}: {error}")
        if not raise_on_exporter_failure:
            traceback.print_exception(error)
    if failures and raise_on_exporter_failure:
        raise failures[0][1]
    return written_paths

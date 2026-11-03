# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Registry of the KPI exporters by name, and the function that runs them on a finished Arena Experiment.

Each exporter module registers its exporter class with the ``register_kpi_exporter`` decorator. Read the registered
exporters through ``get_exporters_by_name``, which imports the exporter modules first.
"""

from __future__ import annotations

import importlib
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path

from isaaclab_arena.evaluation.arena_experiment_result import ArenaExperimentResult
from isaaclab_arena_benchmarking.utils.influxdb_kpi_records import InfluxdbKpiExporter

_EXPORTER_MODULE_NAMES = (
    "isaaclab_arena_benchmarking.benchmarks.success_rate.success_rate_exporter",
    "isaaclab_arena_benchmarking.benchmarks.timing.timing_exporter",
)
"""Modules whose import registers the KPI exporters; add new exporter modules here."""

_EXPORTERS_BY_NAME: dict[str, type[InfluxdbKpiExporter]] = {}


def register_kpi_exporter(name: str) -> Callable[[type[InfluxdbKpiExporter]], type[InfluxdbKpiExporter]]:
    """Return a class decorator that registers a KPI exporter under ``name``.

    Args:
        name: Name that selects the exporter; no other exporter may use it.
    """

    def decorator(exporter_type: type[InfluxdbKpiExporter]) -> type[InfluxdbKpiExporter]:
        assert name not in _EXPORTERS_BY_NAME, (
            f"KPI exporter name {name!r} is already registered by {_EXPORTERS_BY_NAME[name].__name__}, cannot"
            f" register {exporter_type.__name__}"
        )
        _EXPORTERS_BY_NAME[name] = exporter_type
        return exporter_type

    return decorator


def ensure_kpi_exporters_registered() -> None:
    """Import every exporter module, so that its exporter is registered; repeated calls do nothing."""
    for module_name in _EXPORTER_MODULE_NAMES:
        importlib.import_module(module_name)


def get_exporters_by_name() -> dict[str, type[InfluxdbKpiExporter]]:
    """Return the KPI exporter classes by the name used to select them, with every exporter module imported."""
    ensure_kpi_exporters_registered()
    return _EXPORTERS_BY_NAME


def assert_valid_kpi_exporter_names(exporter_names: Sequence[str]) -> None:
    """Assert that every name is a registered KPI exporter and that no name appears twice.

    Args:
        exporter_names: Names of registered KPI exporters, as returned by ``get_exporters_by_name``.
    """
    exporters_by_name = get_exporters_by_name()
    unknown_names = [name for name in exporter_names if name not in exporters_by_name]
    assert not unknown_names, f"Unknown KPI exporters {unknown_names}; valid names are {sorted(exporters_by_name)}"
    duplicate_names = sorted({name for name in exporter_names if list(exporter_names).count(name) > 1})
    assert not duplicate_names, f"Select each KPI exporter once; duplicate KPI exporters {duplicate_names}"


def export_kpis(
    output_dir: Path,
    exporter_names: Sequence[str],
    *,
    raise_on_exporter_failure: bool = True,
) -> list[Path]:
    """Write the KPI records files of a finished Experiment, unless any of the Runs in the Experiment failed.

    Every selected exporter runs even if an earlier one raises. When any of the Runs in the Experiment failed, the
    failed Runs are printed and no KPI file is written. With no exporter selected, nothing is read or written.

    Args:
        output_dir: Experiment output root holding ``arena_experiment_result.json``.
        exporter_names: Names of the registered exporters to run, each at most once; no two may share
            a record type. Empty to export nothing.
        raise_on_exporter_failure: Whether to re-raise the first exporter exception after every exporter ran. When
            False, each failure is printed with its exporter name instead.

    Returns:
        Paths of the records files actually written; empty when any Run failed. An exporter with no records writes none.
    """
    assert_valid_kpi_exporter_names(exporter_names)
    if not exporter_names:
        return []
    exporters = {name: get_exporters_by_name()[name]() for name in exporter_names}

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

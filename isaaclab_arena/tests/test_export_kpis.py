# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the KPI exporter registry and ``export_kpis`` in ``isaaclab_arena_benchmarking/benchmarks``."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import ClassVar

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab_arena.evaluation.arena_experiment_result import (  # noqa: E402
    ARENA_AGGREGATED_TIMINGS_FILENAME,
    ARENA_EXPERIMENT_RESULT_FILENAME,
)
from isaaclab_arena.tests.utils.influxdb_kpi_records_validation import (  # noqa: E402
    assert_influxdb_kpi_records_are_valid,
)
from isaaclab_arena_benchmarking.benchmarks.registry import (  # noqa: E402
    DEFAULT_EXPORTER_NAMES,
    EXPORTERS_BY_NAME,
    export_kpis,
)
from isaaclab_arena_benchmarking.benchmarks.success_rate.success_rate_exporter import PerExperimentRecord  # noqa: E402
from isaaclab_arena_benchmarking.benchmarks.timing.timing_exporter import TimingRecord  # noqa: E402
from isaaclab_arena_benchmarking.utils import kpi_exporter_registry  # noqa: E402
from isaaclab_arena_benchmarking.utils.influxdb_kpi_records import InfluxdbKpiExporter, InfluxdbKpiRecord  # noqa: E402

START_TIMESTAMP_NS = 1792799000000000000
"""Before every Run end timestamp, in Unix epoch nanoseconds."""

END_TIMESTAMP_NS = 1792800000000000000
"""2026-10-23 UTC, in Unix epoch nanoseconds."""

GIT_STAMP = {"git_sha": "1a2b3c4", "git_branch": "main"}

EXPERIMENT_NAME = "pnp_benchmark"

SHARED_VALUES = {"timestamp_ns": END_TIMESTAMP_NS, **GIT_STAMP, "benchmark_id": EXPERIMENT_NAME}

KPI_FILENAMES = {"kpis_success_rates.json", "kpis_timing.json"}


def _write_experiment(output_dir: Path, run_statuses: dict[str, str]) -> Path:
    """Write a result file with one single-episode Run per status, plus a timings file."""
    result_runs = {}
    for run_name, status in run_statuses.items():
        episodes = [{"env_id": 0, "episode_in_env": 0, "success": True, "episode_length": 10}]
        run_dir = output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "episode_results.jsonl").write_text(json.dumps(episodes[0]) + "\n", encoding="utf-8")
        result_runs[run_name] = {
            "environment": {"name": f"{run_name}_env", "definition": "unused"},
            "policy_variant": "pi05",
            "status": status,
            "start_timestamp_ns": START_TIMESTAMP_NS,
            "end_timestamp_ns": END_TIMESTAMP_NS,
            "rebuilds": [{"index": 0, "episodes": episodes}],
        }
    (output_dir / ARENA_EXPERIMENT_RESULT_FILENAME).write_text(
        json.dumps({"git_stamp": GIT_STAMP, "experiment_name": EXPERIMENT_NAME, "runs": result_runs}), encoding="utf-8"
    )
    timings = {"totals": [{"name": "step", "count": 2, "total_ms": 3.0, "min_ms": 1.0, "max_ms": 2.0}], "runs": []}
    (output_dir / ARENA_AGGREGATED_TIMINGS_FILENAME).write_text(json.dumps(timings), encoding="utf-8")
    return output_dir


def _kpi_filenames(output_dir: Path) -> set[str]:
    return {path.name for path in output_dir.glob("kpis_*.json")}


class _FailingExporter(InfluxdbKpiExporter):
    record_types: ClassVar[tuple[str, ...]] = ("failing",)

    def get_records_filename(self) -> str:
        return "kpis_failing.json"

    def build_records_from_output_dir(self, output_dir: Path) -> list[InfluxdbKpiRecord]:
        raise RuntimeError("exporter broke")


class _SecondFailingExporter(_FailingExporter):
    record_types: ClassVar[tuple[str, ...]] = ("failing_too",)

    def build_records_from_output_dir(self, output_dir: Path) -> list[InfluxdbKpiRecord]:
        raise ValueError("second exporter broke")


def test_registry_lists_the_default_exporters():
    assert set(EXPORTERS_BY_NAME) == {"success_rate", "timing"}
    assert DEFAULT_EXPORTER_NAMES == ("success_rate", "timing")


def test_completed_experiment_writes_every_default_file(tmp_path: Path):
    output_dir = _write_experiment(tmp_path / "experiment", {"run_a": "completed", "run_b": "completed"})

    written_paths = export_kpis(output_dir)

    assert {path.name for path in written_paths} == KPI_FILENAMES
    assert _kpi_filenames(output_dir) == KPI_FILENAMES
    kpis = []
    for path in written_paths:
        file_kpis = json.loads(path.read_text(encoding="utf-8"))
        assert_influxdb_kpi_records_are_valid(file_kpis)
        kpis.extend(file_kpis)
    assert {record["type"] for record in kpis} == {"per_experiment", "timing"}


def test_every_record_has_benchmark_id_tag_from_the_experiment_name(tmp_path: Path):
    output_dir = _write_experiment(tmp_path / "experiment", {"run_a": "completed"})

    written_paths = export_kpis(output_dir)

    kpis = [record for path in written_paths for record in json.loads(path.read_text(encoding="utf-8"))]
    assert {record["type"] for record in kpis} == {"per_experiment", "timing"}
    assert all(record["benchmark_id"] == EXPERIMENT_NAME for record in kpis)


def test_result_without_experiment_name_fails_kpi_export(tmp_path: Path):
    output_dir = _write_experiment(tmp_path / "experiment", {"run_a": "completed"})
    result_path = output_dir / ARENA_EXPERIMENT_RESULT_FILENAME
    result = json.loads(result_path.read_text(encoding="utf-8"))
    del result["experiment_name"]
    result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(AssertionError, match="experiment_name"):
        export_kpis(output_dir)

    assert _kpi_filenames(output_dir) == set()


def test_experiment_without_timings_file_writes_only_the_success_rate_file(tmp_path: Path):
    output_dir = _write_experiment(tmp_path / "experiment", {"run_a": "completed"})
    (output_dir / ARENA_AGGREGATED_TIMINGS_FILENAME).unlink()

    written_paths = export_kpis(output_dir)

    assert [path.name for path in written_paths] == ["kpis_success_rates.json"]
    assert _kpi_filenames(output_dir) == {"kpis_success_rates.json"}


def test_failed_run_writes_no_files(tmp_path: Path, capsys):
    output_dir = _write_experiment(tmp_path / "experiment", {"good_run": "completed", "bad_run": "failed"})

    assert export_kpis(output_dir) == []

    assert _kpi_filenames(output_dir) == set()
    printed = capsys.readouterr().out
    assert "bad_run" in printed
    assert "good_run" not in printed


def test_one_exporter_failing_still_writes_the_other(tmp_path: Path, monkeypatch):
    output_dir = _write_experiment(tmp_path / "experiment", {"run_a": "completed"})
    monkeypatch.setitem(EXPORTERS_BY_NAME, "failing", _FailingExporter)

    with pytest.raises(RuntimeError, match="exporter broke"):
        export_kpis(output_dir, ("failing", "success_rate"))

    assert _kpi_filenames(output_dir) == {"kpis_success_rates.json"}


def test_first_exception_is_raised_after_every_exporter_ran(tmp_path: Path, monkeypatch, capsys):
    output_dir = _write_experiment(tmp_path / "experiment", {"run_a": "completed"})
    monkeypatch.setitem(EXPORTERS_BY_NAME, "failing", _FailingExporter)
    monkeypatch.setitem(EXPORTERS_BY_NAME, "failing_too", _SecondFailingExporter)

    with pytest.raises(RuntimeError, match="exporter broke"):
        export_kpis(output_dir, ("failing", "failing_too", "timing"))

    assert _kpi_filenames(output_dir) == {"kpis_timing.json"}
    printed = capsys.readouterr().out
    assert "exporter broke" in printed
    assert "second exporter broke" in printed


def test_raise_on_exporter_failure_false_does_not_raise(tmp_path: Path, monkeypatch, capsys):
    output_dir = _write_experiment(tmp_path / "experiment", {"run_a": "completed"})
    monkeypatch.setitem(EXPORTERS_BY_NAME, "failing", _FailingExporter)
    monkeypatch.setitem(EXPORTERS_BY_NAME, "failing_too", _SecondFailingExporter)

    written_paths = export_kpis(output_dir, ("failing", "success_rate", "failing_too"), raise_on_exporter_failure=False)

    assert [path.name for path in written_paths] == ["kpis_success_rates.json"]
    printed = capsys.readouterr().out
    assert "failing" in printed and "exporter broke" in printed
    assert "failing_too" in printed and "second exporter broke" in printed


def test_registering_a_second_exporter_under_a_taken_name_is_rejected(monkeypatch):
    monkeypatch.setattr(kpi_exporter_registry, "EXPORTERS_BY_NAME", dict(EXPORTERS_BY_NAME))

    with pytest.raises(AssertionError, match="'timing' is already registered"):
        kpi_exporter_registry.register_kpi_exporter("timing")(_FailingExporter)


def test_register_kpi_exporter_adds_the_class_under_its_name(monkeypatch):
    monkeypatch.setattr(kpi_exporter_registry, "EXPORTERS_BY_NAME", {})

    decorated_type = kpi_exporter_registry.register_kpi_exporter("failing")(_FailingExporter)

    assert decorated_type is _FailingExporter
    assert kpi_exporter_registry.EXPORTERS_BY_NAME == {"failing": _FailingExporter}


def test_unknown_exporter_name_is_rejected(tmp_path: Path):
    output_dir = _write_experiment(tmp_path / "experiment", {"run_a": "completed"})

    with pytest.raises(AssertionError, match="no_such_exporter") as error_info:
        export_kpis(output_dir, ("no_such_exporter",))

    assert "success_rate" in str(error_info.value) and "timing" in str(error_info.value)
    assert _kpi_filenames(output_dir) == set()


def test_exporters_with_the_same_type_are_rejected(tmp_path: Path, monkeypatch):
    output_dir = _write_experiment(tmp_path / "experiment", {"run_a": "completed"})

    class _DuplicateTimingExporter(_FailingExporter):
        record_types: ClassVar[tuple[str, ...]] = ("timing",)

    monkeypatch.setitem(EXPORTERS_BY_NAME, "duplicate_timing", _DuplicateTimingExporter)

    with pytest.raises(AssertionError, match="timing"):
        export_kpis(output_dir, ("timing", "duplicate_timing"))

    assert _kpi_filenames(output_dir) == set()


def test_exporters_declare_the_record_types_they_build():
    assert EXPORTERS_BY_NAME["success_rate"].record_types == (PerExperimentRecord.record_type,)
    assert EXPORTERS_BY_NAME["timing"].record_types == (TimingRecord.record_type,)


def _valid_records() -> list[dict]:
    """Return the flat dicts of a valid per-Experiment and timing record."""
    per_experiment = PerExperimentRecord(**SHARED_VALUES, num_episodes=1, num_runs=1)
    timing = TimingRecord(**SHARED_VALUES, name="a", mean_ms=1.0, total_ms=1.0, min_ms=1.0, max_ms=1.0, count=1)
    return [per_experiment.to_flat_dict(), timing.to_flat_dict()]


def _set_record_value(kpis: list[dict], key: str, value: object) -> None:
    kpis[1][key] = value


def _duplicate_record(kpis: list[dict]) -> None:
    kpis.append({**kpis[1], "count": 2})


def _keep_only_timestamp_ns(kpis: list[dict]) -> None:
    kpis[1] = {key: value for key, value in kpis[1].items() if not isinstance(value, (int, float))}
    kpis[1]["timestamp_ns"] = END_TIMESTAMP_NS


def test_records_are_valid_influxdb_points():
    assert_influxdb_kpi_records_are_valid(_valid_records())


@pytest.mark.parametrize(
    "make_invalid",
    [
        pytest.param(lambda kpis: kpis.clear(), id="empty_list"),
        pytest.param(lambda kpis: kpis[1].pop("timestamp_ns"), id="missing_timestamp_ns"),
        pytest.param(lambda kpis: _set_record_value(kpis, "timestamp_ns", 1.8e18), id="float_timestamp_ns"),
        pytest.param(lambda kpis: _set_record_value(kpis, "timestamp_ns", True), id="bool_timestamp_ns"),
        pytest.param(lambda kpis: _set_record_value(kpis, "timestamp_ns", 1767225599999999999), id="before_2026"),
        pytest.param(lambda kpis: _set_record_value(kpis, "timestamp_ns", 2**62), id="future_timestamp_ns"),
        pytest.param(_keep_only_timestamp_ns, id="timestamp_ns_is_only_number"),
        pytest.param(lambda kpis: _set_record_value(kpis, "git_sha", ""), id="empty_git_sha"),
        pytest.param(lambda kpis: _set_record_value(kpis, "nested", {"a": 1}), id="nested_value"),
        pytest.param(lambda kpis: _set_record_value(kpis, "missing", None), id="null_value"),
        pytest.param(lambda kpis: _set_record_value(kpis, "flag", True), id="bool_value"),
        pytest.param(lambda kpis: _set_record_value(kpis, "rate", math.inf), id="infinite_value"),
        pytest.param(lambda kpis: _set_record_value(kpis, "env", "a\nb"), id="line_break_value"),
        pytest.param(_duplicate_record, id="duplicate_tag_set"),
        pytest.param(lambda kpis: _set_record_value(kpis, "_x", 1), id="underscore_prefixed_key"),
        pytest.param(lambda kpis: _set_record_value(kpis, "time", 1), id="time_key"),
    ],
)
def test_validation_helper_rejects_invalid_records(make_invalid):
    """The validation helper must catch invalid records, or tests that rely on it would pass vacuously."""
    kpis = _valid_records()
    make_invalid(kpis)

    with pytest.raises(AssertionError):
        assert_influxdb_kpi_records_are_valid(kpis)


def test_validation_helper_rejects_non_list_contents():
    with pytest.raises(AssertionError):
        assert_influxdb_kpi_records_are_valid({"records": _valid_records()})

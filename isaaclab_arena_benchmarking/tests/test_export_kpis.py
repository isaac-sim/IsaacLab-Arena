# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the KPI exporter registry and ``export_kpis`` in ``isaaclab_arena_benchmarking/benchmarks``."""

from __future__ import annotations

import json
import subprocess
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
from isaaclab_arena_benchmarking.benchmarks import registry  # noqa: E402
from isaaclab_arena_benchmarking.benchmarks.registry import export_kpis, get_exporters_by_name  # noqa: E402
from isaaclab_arena_benchmarking.benchmarks.success_rate.success_rate_exporter import PerExperimentRecord  # noqa: E402
from isaaclab_arena_benchmarking.benchmarks.timing.timing_exporter import TimingRecord  # noqa: E402
from isaaclab_arena_benchmarking.tests.utils.influxdb_kpi_records_validation import (  # noqa: E402
    assert_influxdb_kpi_records_are_valid,
)
from isaaclab_arena_benchmarking.utils.influxdb_kpi_records import InfluxdbKpiExporter, InfluxdbKpiRecord  # noqa: E402

START_TIMESTAMP_NS = 1792799000000000000
"""Before every Run end timestamp, in Unix epoch nanoseconds."""

END_TIMESTAMP_NS = 1792800000000000000
"""2026-10-23 UTC, in Unix epoch nanoseconds."""

GIT_STAMP = {"git_sha": "1a2b3c4", "git_branch": "main"}

EXPERIMENT_NAME = "pnp_benchmark"

SHARED_VALUES = {"timestamp_ns": END_TIMESTAMP_NS, **GIT_STAMP, "benchmark_id": EXPERIMENT_NAME}

KPI_FILENAMES = {"kpis_success_rates.json", "kpis_timing.json"}

ALL_EXPORTER_NAMES = ["success_rate", "timing"]


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
    assert set(get_exporters_by_name()) == {"placement_success", "success_rate", "timing"}


def test_registry_lists_the_exporters_in_a_fresh_process_without_importing_torch_or_isaaclab():
    """Read the registry before any exporter module is imported, in a fresh process."""
    code = (
        "import sys\n"
        "from isaaclab_arena_benchmarking.benchmarks.registry import get_exporters_by_name\n"
        "print(sorted(get_exporters_by_name()))\n"
        "heavy_modules = sorted(name for name in ('torch', 'isaaclab', 'isaacsim') if name in sys.modules)\n"
        "print(heavy_modules)\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO_ROOT, timeout=120
    )

    assert result.stdout.splitlines()[-2:] == ["['placement_success', 'success_rate', 'timing']", "[]"]


def test_completed_experiment_writes_every_default_file(tmp_path: Path):
    output_dir = _write_experiment(tmp_path / "experiment", {"run_a": "completed", "run_b": "completed"})

    written_paths = export_kpis(output_dir, ALL_EXPORTER_NAMES)

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

    written_paths = export_kpis(output_dir, ALL_EXPORTER_NAMES)

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
        export_kpis(output_dir, ALL_EXPORTER_NAMES)

    assert _kpi_filenames(output_dir) == set()


def test_experiment_without_timings_file_writes_only_the_success_rate_file(tmp_path: Path):
    output_dir = _write_experiment(tmp_path / "experiment", {"run_a": "completed"})
    (output_dir / ARENA_AGGREGATED_TIMINGS_FILENAME).unlink()

    written_paths = export_kpis(output_dir, ALL_EXPORTER_NAMES)

    assert [path.name for path in written_paths] == ["kpis_success_rates.json"]
    assert _kpi_filenames(output_dir) == {"kpis_success_rates.json"}


def test_failed_run_writes_no_files(tmp_path: Path, capsys):
    output_dir = _write_experiment(tmp_path / "experiment", {"good_run": "completed", "bad_run": "failed"})

    assert export_kpis(output_dir, ALL_EXPORTER_NAMES) == []

    assert _kpi_filenames(output_dir) == set()
    printed = capsys.readouterr().out
    assert "bad_run" in printed
    assert "good_run" not in printed


def test_one_exporter_failing_still_writes_the_other(tmp_path: Path, monkeypatch):
    output_dir = _write_experiment(tmp_path / "experiment", {"run_a": "completed"})
    monkeypatch.setitem(get_exporters_by_name(), "failing", _FailingExporter)

    with pytest.raises(RuntimeError, match="exporter broke"):
        export_kpis(output_dir, ("failing", "success_rate"))

    assert _kpi_filenames(output_dir) == {"kpis_success_rates.json"}


def test_first_exception_is_raised_after_every_exporter_ran(tmp_path: Path, monkeypatch, capsys):
    output_dir = _write_experiment(tmp_path / "experiment", {"run_a": "completed"})
    monkeypatch.setitem(get_exporters_by_name(), "failing", _FailingExporter)
    monkeypatch.setitem(get_exporters_by_name(), "failing_too", _SecondFailingExporter)

    with pytest.raises(RuntimeError, match="exporter broke"):
        export_kpis(output_dir, ("failing", "failing_too", "timing"))

    assert _kpi_filenames(output_dir) == {"kpis_timing.json"}
    printed = capsys.readouterr().out
    assert "exporter broke" in printed
    assert "second exporter broke" in printed


def test_raise_on_exporter_failure_false_does_not_raise(tmp_path: Path, monkeypatch, capsys):
    output_dir = _write_experiment(tmp_path / "experiment", {"run_a": "completed"})
    monkeypatch.setitem(get_exporters_by_name(), "failing", _FailingExporter)
    monkeypatch.setitem(get_exporters_by_name(), "failing_too", _SecondFailingExporter)

    written_paths = export_kpis(output_dir, ("failing", "success_rate", "failing_too"), raise_on_exporter_failure=False)

    assert [path.name for path in written_paths] == ["kpis_success_rates.json"]
    printed = capsys.readouterr().out
    assert "failing" in printed and "exporter broke" in printed
    assert "failing_too" in printed and "second exporter broke" in printed


def test_registering_a_second_exporter_under_a_taken_name_is_rejected(monkeypatch):
    with pytest.raises(AssertionError, match="'timing' is already registered"):
        registry.register_kpi_exporter("timing")(_FailingExporter)


def test_register_kpi_exporter_adds_the_class_under_its_name(monkeypatch):
    monkeypatch.setattr(registry, "_EXPORTERS_BY_NAME", {})

    decorated_type = registry.register_kpi_exporter("failing")(_FailingExporter)

    assert decorated_type is _FailingExporter
    assert registry._EXPORTERS_BY_NAME == {"failing": _FailingExporter}


def test_unknown_exporter_name_is_rejected(tmp_path: Path):
    output_dir = _write_experiment(tmp_path / "experiment", {"run_a": "completed"})

    with pytest.raises(AssertionError, match="no_such_exporter") as error_info:
        export_kpis(output_dir, ("no_such_exporter",))

    assert "success_rate" in str(error_info.value) and "timing" in str(error_info.value)
    assert _kpi_filenames(output_dir) == set()


def test_duplicate_exporter_name_is_rejected(tmp_path: Path):
    output_dir = _write_experiment(tmp_path / "experiment", {"run_a": "completed"})

    with pytest.raises(AssertionError, match=r"duplicate KPI exporters \['timing'\]"):
        export_kpis(output_dir, ["timing", "timing"])

    assert _kpi_filenames(output_dir) == set()


def test_no_exporters_returns_without_reading_the_result_or_reporting_failed_runs(tmp_path: Path, capsys):
    output_dir = _write_experiment(tmp_path / "experiment", {"good_run": "completed", "bad_run": "failed"})
    (output_dir / ARENA_EXPERIMENT_RESULT_FILENAME).unlink()

    assert export_kpis(output_dir, []) == []

    assert capsys.readouterr().out == ""


def test_exporters_with_the_same_type_are_rejected(tmp_path: Path, monkeypatch):
    output_dir = _write_experiment(tmp_path / "experiment", {"run_a": "completed"})

    class _DuplicateTimingExporter(_FailingExporter):
        record_types: ClassVar[tuple[str, ...]] = ("timing",)

    monkeypatch.setitem(get_exporters_by_name(), "duplicate_timing", _DuplicateTimingExporter)

    with pytest.raises(AssertionError, match="timing"):
        export_kpis(output_dir, ("timing", "duplicate_timing"))

    assert _kpi_filenames(output_dir) == set()


def test_exporters_declare_the_record_types_they_build():
    assert get_exporters_by_name()["success_rate"].record_types == (PerExperimentRecord.record_type,)
    assert get_exporters_by_name()["timing"].record_types == (TimingRecord.record_type,)


def _valid_records() -> list[dict]:
    """Return the flat dicts of a valid per-Experiment and timing record."""
    per_experiment = PerExperimentRecord(**SHARED_VALUES, num_episodes=1, num_runs=1)
    timing = TimingRecord(**SHARED_VALUES, name="a", mean_ms=1.0, total_ms=1.0, min_ms=1.0, max_ms=1.0, count=1)
    return [per_experiment.to_flat_dict(), timing.to_flat_dict()]


def test_records_are_valid_influxdb_points():
    assert_influxdb_kpi_records_are_valid(_valid_records())

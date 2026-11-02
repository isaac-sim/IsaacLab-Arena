# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the generic InfluxDB KPI records and exporter in ``isaaclab_arena_benchmarking/utils``."""

from __future__ import annotations

import json
import math
import sys
import textwrap
import types
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab_arena_benchmarking.utils.influxdb_kpi_records import (  # noqa: E402
    InfluxdbKpiExporter,
    InfluxdbKpiRecord,
    assert_unique_tag_sets,
)

TIMESTAMP_NS = 1792800000000000000
"""2026-10-23 UTC, in Unix epoch nanoseconds."""

SHARED_VALUES = {"timestamp_ns": TIMESTAMP_NS, "git_sha": "1a2b3c4", "git_branch": "main"}

TEST_RECORDS_FILENAME = "test_records.json"


@dataclass(kw_only=True)
class _TestRecord(InfluxdbKpiRecord):
    record_type: ClassVar[str] = "test"

    name: str
    score: float | None = None
    count: int


def _test_record(**overrides) -> _TestRecord:
    return _TestRecord(**{**SHARED_VALUES, "name": "a", "count": 1, **overrides})


class _TestExporter(InfluxdbKpiExporter):
    """Exporter that builds one record per line of ``names.txt`` in the output directory."""

    record_types: ClassVar[tuple[str, ...]] = ("test",)

    def get_records_filename(self) -> str:
        return TEST_RECORDS_FILENAME

    def build_records_from_output_dir(self, output_dir: Path) -> list[InfluxdbKpiRecord]:
        names = (output_dir / "names.txt").read_text(encoding="utf-8").split()
        return [_test_record(name=name, score=0.5) for name in names]


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"timestamp_ns": 1.5e18}, id="float_timestamp_ns"),
        pytest.param({"timestamp_ns": True}, id="bool_timestamp_ns"),
        pytest.param({"timestamp_ns": 1_000}, id="tiny_timestamp_ns"),
        pytest.param({"timestamp_ns": 1767225599999999999}, id="timestamp_ns_before_2026"),
        pytest.param({"git_sha": ""}, id="empty_git_sha"),
        pytest.param({"git_branch": "a\nb"}, id="line_break_git_branch"),
        pytest.param({"name": ""}, id="empty_tag"),
        pytest.param({"name": "a\nb"}, id="line_break_tag"),
        pytest.param({"count": True}, id="bool_field"),
        pytest.param({"score": math.nan}, id="nan_field"),
        pytest.param({"score": math.inf}, id="inf_field"),
        pytest.param({"score": "1"}, id="string_field"),
    ],
)
def test_record_rejects_invalid_values(overrides):
    with pytest.raises(AssertionError):
        _test_record(**overrides)


def test_timestamp_ns_does_not_count_as_a_field():
    with pytest.raises(AssertionError, match="at least one numeric field"):
        _test_record(count=None)


def test_record_omits_missing_fields_from_flat_dict():
    record = _test_record()

    assert list(record.to_flat_dict().items()) == [
        ("type", "test"),
        ("timestamp_ns", TIMESTAMP_NS),
        ("git_sha", "1a2b3c4"),
        ("git_branch", "main"),
        ("name", "a"),
        ("count", 1),
    ]


def test_tag_set_holds_only_string_values():
    assert _test_record().tag_set() == {("type", "test"), ("git_sha", "1a2b3c4"), ("git_branch", "main"), ("name", "a")}


def test_unique_tag_sets_reject_duplicate_records():
    assert_unique_tag_sets([_test_record(), _test_record(name="b")])
    with pytest.raises(AssertionError, match="record 1 has the same tags as record 0"):
        assert_unique_tag_sets([_test_record(), _test_record(count=2)])


def test_tags_are_detected_without_future_annotations(monkeypatch):
    # In a module without ``from __future__ import annotations`` the field types are classes, not strings.
    source = textwrap.dedent("""
        from dataclasses import dataclass
        from typing import ClassVar

        from isaaclab_arena_benchmarking.utils.influxdb_kpi_records import InfluxdbKpiRecord


        @dataclass(kw_only=True)
        class EagerRecord(InfluxdbKpiRecord):
            record_type: ClassVar[str] = "eager"
            name: str
            count: int
        """)
    module = types.ModuleType("eager_annotations_module")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    namespace = module.__dict__
    exec(compile(source, "eager_annotations_module.py", "exec", dont_inherit=True), namespace)
    eager_record_class = namespace["EagerRecord"]

    assert [field.type for field in eager_record_class.__dataclass_fields__.values()][-2:] == [str, int]
    eager_record_class(**SHARED_VALUES, name="a", count=1)
    with pytest.raises(AssertionError, match="tag 'name'"):
        eager_record_class(**SHARED_VALUES, name="", count=1)
    with pytest.raises(AssertionError, match="tag 'name'"):
        eager_record_class(**SHARED_VALUES, name="a\nb", count=1)
    with pytest.raises(AssertionError, match="field 'count'"):
        eager_record_class(**SHARED_VALUES, name="a", count="1")


def test_export_to_json_writes_the_records_file(tmp_path):
    (tmp_path / "names.txt").write_text("a b\n", encoding="utf-8")

    records_path = _TestExporter().export_to_json(tmp_path)

    assert records_path == tmp_path / TEST_RECORDS_FILENAME
    text = records_path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert json.loads(text) == [
        {"type": "test", **SHARED_VALUES, "name": "a", "score": 0.5, "count": 1},
        {"type": "test", **SHARED_VALUES, "name": "b", "score": 0.5, "count": 1},
    ]
    assert text == json.dumps(json.loads(text), indent=2) + "\n"


def test_export_to_json_prints_the_written_path(tmp_path, capsys):
    (tmp_path / "names.txt").write_text("a\n", encoding="utf-8")

    records_path = _TestExporter().export_to_json(tmp_path)

    assert f"Wrote InfluxDB KPI records to: {records_path}" in capsys.readouterr().out


def test_export_to_json_rejects_duplicate_tag_sets_without_writing(tmp_path):
    (tmp_path / "names.txt").write_text("a a\n", encoding="utf-8")

    with pytest.raises(AssertionError, match="record 1 has the same tags as record 0"):
        _TestExporter().export_to_json(tmp_path)

    assert not (tmp_path / TEST_RECORDS_FILENAME).exists()


def test_exporter_cannot_be_instantiated_without_its_abstract_methods():
    with pytest.raises(TypeError):
        InfluxdbKpiExporter()


def test_exporter_declares_the_record_types_it_builds():
    assert typing.get_type_hints(InfluxdbKpiExporter)["record_types"] == typing.ClassVar[tuple[str, ...]]
    assert _TestExporter.record_types == (_TestRecord.record_type,)


def test_export_to_json_writes_no_file_without_records(tmp_path, capsys):
    (tmp_path / "names.txt").write_text("", encoding="utf-8")

    assert _TestExporter().export_to_json(tmp_path) is None

    assert not (tmp_path / TEST_RECORDS_FILENAME).exists()
    assert TEST_RECORDS_FILENAME in capsys.readouterr().out

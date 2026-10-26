# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test helper that checks KPI records file contents map to valid InfluxDB points."""

from __future__ import annotations

import math
import time

MIN_TIMESTAMP_NS = 1767225600 * 10**9
"""2026-01-01T00:00:00Z in Unix epoch nanoseconds; earlier timestamps are rejected."""

TIMESTAMP_NS_KEY = "timestamp_ns"
"""Key of a record's point time, the only numeric value that is not a field."""


def _is_number(value: object) -> bool:
    """Return whether a value is an int or float; bool is not a number."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite_number(value: int | float) -> bool:
    """Return whether a number converts to a finite float."""
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _assert_single_line_text(text: object, description: str) -> None:
    """Assert that a value is a string usable as a line-protocol key or tag value."""
    assert isinstance(text, str), f"{description} must be a string, got {text!r}"
    assert text, f"{description} must not be empty"
    assert "\n" not in text and "\r" not in text, f"{description} must not contain a line break"


def _assert_record_is_valid_point(record_index: int, record: object) -> frozenset:
    """Assert that one record maps to an InfluxDB point and return its tag set: the record's string-valued items."""
    assert isinstance(record, dict), f"record {record_index} must be a JSON object"
    assert isinstance(record.get("type"), str), f"record {record_index} must have a string type"

    timestamp_ns = record.get(TIMESTAMP_NS_KEY)
    assert isinstance(timestamp_ns, int) and not isinstance(
        timestamp_ns, bool
    ), f"record {record_index}: timestamp_ns must be an integer, got {timestamp_ns!r}"
    assert (
        timestamp_ns >= MIN_TIMESTAMP_NS
    ), f"record {record_index}: timestamp_ns {timestamp_ns} is before 2026-01-01T00:00:00Z"
    assert timestamp_ns <= time.time_ns(), f"record {record_index}: timestamp_ns {timestamp_ns} is in the future"

    has_field = False
    for key, value in record.items():
        _assert_single_line_text(key, f"record {record_index} key {key!r}")
        assert not key.startswith("_"), f"record {record_index}: key {key!r} must not start with '_'"
        assert key != "time", f"record {record_index}: key 'time' is reserved by InfluxDB"
        if key == TIMESTAMP_NS_KEY:
            continue
        if _is_number(value):
            assert _is_finite_number(value), f"record {record_index}: value for {key!r} must be finite"
            has_field = True
        else:
            _assert_single_line_text(value, f"record {record_index} value for {key!r}")
    assert has_field, f"record {record_index} must have at least one numeric value other than timestamp_ns"
    return frozenset((key, value) for key, value in record.items() if isinstance(value, str))


def assert_influxdb_kpi_records_are_valid(contents: object) -> None:
    """Assert that parsed KPI records file contents map to InfluxDB points without changes.

    Checks for a non-empty list of flat records, each with an integer ``timestamp_ns`` from 2026 that is not in
    the future, keys that InfluxDB does not reserve (no leading ``_``, not ``time``), values that are finite numbers or single-line strings, at least one
    numeric value other than ``timestamp_ns``, and a tag set unique within the list.

    Args:
        contents: Parsed JSON of an ``arena_experiment_influxdb_kpi_records.json`` file.
    """
    assert isinstance(contents, list) and contents, "KPI records file contents must be a non-empty JSON list"
    record_index_by_tag_set: dict[frozenset, int] = {}
    for record_index, record in enumerate(contents):
        tag_set = _assert_record_is_valid_point(record_index, record)
        assert (
            tag_set not in record_index_by_tag_set
        ), f"record {record_index} has the same tag set as record {record_index_by_tag_set.get(tag_set)}"
        record_index_by_tag_set[tag_set] = record_index

# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test helper that checks KPI records file contents against the rules an InfluxDB ingestion tool relies on."""

from __future__ import annotations

import math
import time

SUPPORTED_SCHEMA_VERSION = 1
"""Only schema version an ingestion tool accepts."""

MIN_TIMESTAMP_NS = 1767225600 * 10**9
"""2026-01-01T00:00:00Z in Unix epoch nanoseconds; earlier timestamps are rejected."""

RESERVED_RECORD_KEYS = frozenset({"benchmark", "launch_time", "git_sha", "branch"})
"""Tag keys an ingestion tool adds to every point, so records must not use them."""

PROVENANCE_KEYS = ("git_sha", "git_branch")
"""Keys of the provenance object."""


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


def _assert_record_follows_contract(record_index: int, record: object) -> frozenset:
    """Assert that one record follows the contract and return its tag set: the record's string-valued items."""
    assert isinstance(record, dict), f"record {record_index} must be a JSON object"
    assert isinstance(record.get("type"), str), f"record {record_index} must have a string type"
    has_number = False
    for key, value in record.items():
        _assert_single_line_text(key, f"record {record_index} key {key!r}")
        assert not key.startswith("_"), f"record {record_index}: key {key!r} must not start with '_'"
        assert key != "time", f"record {record_index}: key 'time' is reserved by InfluxDB"
        assert key not in RESERVED_RECORD_KEYS, f"record {record_index}: key {key!r} is reserved"
        if _is_number(value):
            assert _is_finite_number(value), f"record {record_index}: value for {key!r} must be finite"
            has_number = True
        else:
            _assert_single_line_text(value, f"record {record_index} value for {key!r}")
    assert has_number, f"record {record_index} must have at least one numeric value"
    return frozenset((key, value) for key, value in record.items() if isinstance(value, str))


def assert_follows_collector_contract(contents: object) -> None:
    """Assert that parsed KPI records file contents can be ingested into InfluxDB without changes.

    Checks the schema version, a timestamp from 2026 that is not in the future, single-line provenance strings,
    and flat records whose keys are not reserved, whose values are finite numbers or single-line strings,
    which have at least one numeric value, and whose tag sets are unique.

    Args:
        contents: Parsed JSON of an ``arena_experiment_influxdb_kpi_records.json`` file.
    """
    assert isinstance(contents, dict), "KPI records file contents must be a JSON object"

    schema_version = contents.get("schema_version")
    assert (
        _is_number(schema_version) and schema_version == SUPPORTED_SCHEMA_VERSION
    ), f"schema_version must be {SUPPORTED_SCHEMA_VERSION}, got {schema_version!r}"

    timestamp_ns = contents.get("timestamp_ns")
    assert isinstance(timestamp_ns, int) and not isinstance(
        timestamp_ns, bool
    ), f"timestamp_ns must be an integer, got {timestamp_ns!r}"
    assert timestamp_ns >= MIN_TIMESTAMP_NS, f"timestamp_ns {timestamp_ns} is before 2026-01-01T00:00:00Z"
    assert timestamp_ns <= time.time_ns(), f"timestamp_ns {timestamp_ns} is in the future"

    provenance = contents.get("provenance")
    assert isinstance(provenance, dict), "provenance must be a JSON object"
    assert set(provenance) == set(PROVENANCE_KEYS), f"provenance must have exactly the keys {PROVENANCE_KEYS}"
    for key in PROVENANCE_KEYS:
        _assert_single_line_text(provenance[key], f"provenance.{key}")

    records = contents.get("records")
    assert isinstance(records, list) and records, "records must be a non-empty list"
    record_index_by_tag_set: dict[frozenset, int] = {}
    for record_index, record in enumerate(records):
        tag_set = _assert_record_follows_contract(record_index, record)
        assert (
            tag_set not in record_index_by_tag_set
        ), f"record {record_index} has the same tag set as record {record_index_by_tag_set.get(tag_set)}"
        record_index_by_tag_set[tag_set] = record_index

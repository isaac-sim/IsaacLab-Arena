# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Generic InfluxDB KPI records and a base class that writes them to a JSON file.

This module only formats data for InfluxDB. Ingesting the file into a database is not part of this repository.
It depends on the standard library only, so it can be imported without Arena or Isaac Sim.

InfluxDB tags and fields
------------------------
InfluxDB stores time series of records. A record has tags, which are indexed strings that identify the series it
belongs to and that queries filter and group by, and fields, which are the measured values (here always numbers).
A record is identified by its measurement, tag set and time, so a second record with the same identity overwrites
the first. A record is always flat and can only store scalars, i.e. no nested lists or dicts.

A records file is a JSON list of flat records. Fields appearing in every record:

- ``timestamp_ns``: Record time in Unix epoch nanoseconds (UTC).
- ``type``: Record type.
- ``git_sha``: Git commit SHA the records were produced with.
- ``git_branch``: Git branch the records were produced with.
"""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from pathlib import Path
from typing import ClassVar

MIN_TIMESTAMP_NS = 1767225600 * 10**9
"""2026-01-01T00:00:00Z in Unix epoch nanoseconds.

No KPI records were written before this date, so an earlier timestamp is malformed, for example one in seconds
instead of nanoseconds.
"""

RECORD_TYPE_KEY = "type"
"""Key of the tag that holds a record's type."""

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


def _is_single_line_text(value: object) -> bool:
    """Return whether a value is a non-empty string without line breaks."""
    return isinstance(value, str) and bool(value) and "\n" not in value and "\r" not in value


@dataclass(kw_only=True)
class InfluxdbKpiRecord(ABC):
    """Abstract base class for KPI records.

    Each record is written as a flat JSON object for one InfluxDB point. Subclasses set ``record_type`` and add
    their own tags and fields. Attributes annotated as ``str`` are tags; the others, except ``timestamp_ns``, are
    numeric fields, left out when set to None.
    """

    timestamp_ns: int
    """Point time in Unix epoch nanoseconds (UTC); not a field."""

    git_sha: str
    """Git commit SHA the records were produced with."""

    git_branch: str
    """Git branch the records were produced with."""

    @property
    @abstractmethod
    def record_type(self) -> str:
        """Value of the record's ``type`` tag."""

    def __post_init__(self) -> None:
        # Check that the record can be written as one valid InfluxDB point: a plausible nanosecond timestamp,
        # non-empty single-line tags, finite numeric fields and at least one field.
        assert (
            isinstance(self.timestamp_ns, int)
            and not isinstance(self.timestamp_ns, bool)
            and self.timestamp_ns >= MIN_TIMESTAMP_NS
        ), (
            f"{self.record_type} record: timestamp_ns must be an integer no earlier than 2026-01-01T00:00:00Z,"
            f" got {self.timestamp_ns!r}"
        )
        has_field = False
        for record_field in fields(self):
            if record_field.name == TIMESTAMP_NS_KEY:
                continue
            value = getattr(self, record_field.name)
            # Annotations are strings under ``from __future__ import annotations`` and classes otherwise.
            if record_field.type in ("str", str):
                assert _is_single_line_text(value), (
                    f"{self.record_type} record: tag {record_field.name!r} must be a non-empty single-line string,"
                    f" got {value!r}"
                )
            elif value is not None:
                assert _is_number(value) and _is_finite_number(
                    value
                ), f"{self.record_type} record: field {record_field.name!r} must be a finite number, got {value!r}"
                has_field = True
        assert has_field, f"{self.record_type} record must have at least one numeric field"

    def to_flat_dict(self) -> dict[str, str | int | float]:
        """Return the record as a flat JSON object.

        The keys are ``type``, ``timestamp_ns``, ``git_sha`` and ``git_branch``, then every other attribute that is
        not None.
        """
        flat_dict: dict[str, str | int | float] = {RECORD_TYPE_KEY: self.record_type}
        for record_field in fields(self):
            value = getattr(self, record_field.name)
            if value is not None:
                flat_dict[record_field.name] = value
        return flat_dict

    def tag_set(self) -> frozenset[tuple[str, str]]:
        """Return the record's tags as ``(key, value)`` pairs, which identify its InfluxDB series."""
        return frozenset((key, value) for key, value in self.to_flat_dict().items() if isinstance(value, str))


def assert_unique_tag_sets(records: list[InfluxdbKpiRecord]) -> None:
    """Assert that no two records have the same tag set, so each record is a distinct InfluxDB point.

    Args:
        records: KPI records that share one timestamp.
    """
    record_index_by_tag_set: dict[frozenset[tuple[str, str]], int] = {}
    for record_index, record in enumerate(records):
        tag_set = record.tag_set()
        assert (
            tag_set not in record_index_by_tag_set
        ), f"record {record_index} has the same tags as record {record_index_by_tag_set.get(tag_set)}"
        record_index_by_tag_set[tag_set] = record_index


@dataclass(kw_only=True)
class TimingRecord(InfluxdbKpiRecord):
    """Statistics of one timer, for one run or summed over every run."""

    record_type: ClassVar[str] = "timing"

    name: str
    """Timer name."""

    run_name: str
    """Name of the run, or ``"all"`` for totals over every run."""

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


class InfluxdbKpiExporter(ABC):
    """Abstract base class for exporters that build KPI records from an output directory and write them to JSON."""

    @abstractmethod
    def get_records_filename(self) -> str:
        """Return the name of the records file written into the output directory."""

    @abstractmethod
    def build_records_from_output_dir(self, output_dir: Path) -> list[InfluxdbKpiRecord]:
        """Build the KPI records of a finished output directory.

        Args:
            output_dir: Directory holding the results to build records from.

        Returns:
            The KPI records of the output directory.
        """

    def export_to_json(self, output_dir: Path) -> Path:
        """Build the records of an output directory and write them to the records file in that directory.

        Args:
            output_dir: Directory holding the results to build records from; the records file is written into it.

        Returns:
            Path to the written records file.
        """
        output_dir = Path(output_dir)
        records = self.build_records_from_output_dir(output_dir)
        assert_unique_tag_sets(records)
        records_path = output_dir / self.get_records_filename()
        records_json = json.dumps([record.to_flat_dict() for record in records], allow_nan=False, indent=2)
        records_path.write_text(records_json + "\n", encoding="utf-8")
        print(f"Wrote InfluxDB KPI records to: {records_path}")
        return records_path

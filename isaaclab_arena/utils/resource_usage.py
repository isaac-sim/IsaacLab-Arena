# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Statistics of the memory and utilization of a process, and their JSON records."""

from __future__ import annotations

from collections.abc import Iterable, MutableMapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from isaaclab_arena.utils.sample_statistics import SAMPLE_DECIMALS, SampleStatistics, merge_statistics_json


@dataclass
class MegabyteStatistics(SampleStatistics):
    """Statistics of measurements in megabytes (2**20 bytes)."""

    unit: ClassVar[str] = "mb"


@dataclass
class PercentStatistics(SampleStatistics):
    """Statistics of measurements in percent."""

    unit: ClassVar[str] = "percent"


RESOURCE_NAMES = ("vram", "ram", "cpu_util", "gpu_util")
"""Names of the tracked resources, in the order their records are written."""

RESOURCE_UNITS: dict[str, str] = {"vram": "mb", "ram": "mb", "cpu_util": "percent", "gpu_util": "percent"}
"""Unit of each tracked resource."""


@dataclass
class ResourceUsage:
    """Statistics of the resources used during a Run."""

    vram: MegabyteStatistics = field(default_factory=MegabyteStatistics)
    """Device-wide GPU memory in use."""

    ram: MegabyteStatistics = field(default_factory=MegabyteStatistics)
    """Resident memory of the process."""

    cpu_util: PercentStatistics = field(default_factory=PercentStatistics)
    """CPU utilization of the process."""

    gpu_util: PercentStatistics = field(default_factory=PercentStatistics)
    """GPU utilization of the device."""

    def to_json_records(self, app_name: str) -> list[dict]:
        """Return one JSON record per resource that has measurements, in RESOURCE_NAMES order.

        Args:
            app_name: Name of the application the measurements belong to.

        Returns:
            The records, each with a type, name, app_name and the unit-suffixed statistics fields.
        """
        records = []
        for name in RESOURCE_NAMES:
            statistics: SampleStatistics = getattr(self, name)
            if statistics.count == 0:
                continue
            records.append({
                "type": "resource",
                "name": name,
                "app_name": app_name,
                **statistics.to_json_fields(SAMPLE_DECIMALS),
            })
        return records


def merge_resource_records(
    records: Iterable[MutableMapping[str, Any]], remove_samples_from_input_records: bool = False
) -> list[dict]:
    """Combine resource records that share a name into one record per name.

    Args:
        records: Records produced by ResourceUsage.to_json_records.
        remove_samples_from_input_records: Whether to delete the samples from each input record once read.

    Returns:
        One merged record per resource name, sorted by name.
    """
    records_by_name: dict[str, list[MutableMapping[str, Any]]] = {}
    for record in records:
        records_by_name.setdefault(record["name"], []).append(record)

    merged_records = []
    for name, named_records in records_by_name.items():
        merged_records.extend(
            merge_statistics_json(
                named_records, RESOURCE_UNITS[name], remove_samples_from_input_records=remove_samples_from_input_records
            )
        )
    return sorted(merged_records, key=lambda record: record["name"])

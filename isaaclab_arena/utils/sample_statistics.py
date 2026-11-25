# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Running statistics of a stream of measurements, with approximate percentiles and a cross-record merge.

Subclass SampleStatistics and implement get_unit and get_num_decimals for the measured quantity, for example
"ms" with 3 decimals or "percent" with 0. The unit suffixes the serialized keys, as in "mean_ms" and "samples_percent".
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Iterable, MutableMapping
from dataclasses import dataclass, field, replace
from typing import Any, Self

# Keep reservoir sampling independent from application randomness.
_reservoir_random_generator = random.Random()

PERCENTILES = (10, 50, 90)
"""Percentiles written to a record and estimated when records are merged."""


def weighted_percentile(weighted_samples: list[tuple[float, float]], p: float) -> float:
    """Return the p-th percentile (0-100) of (value, weight) pairs: the smallest value whose cumulative weight reaches p%."""
    assert 0 <= p <= 100, f"Percentile must be between 0 and 100, got {p}"
    ordered = sorted(weighted_samples)
    threshold = p / 100.0 * sum(weight for _, weight in ordered)
    cumulative_weight = 0.0
    for value, weight in ordered:
        cumulative_weight += weight
        if cumulative_weight >= threshold:
            return value
    return ordered[-1][0]


@dataclass
class SampleStatistics(ABC):
    """Accumulated statistics of measurements in one unit.

    Maintains a reservoir sample for approximate percentile estimation.
    """

    percentile_approximation_reservoir_size: int = 1024
    count: int = 0
    total: float = 0.0
    minimum: float = field(default=float("inf"))
    maximum: float = field(default=float("-inf"))
    _reservoir: list[float] = field(default_factory=list)

    @classmethod
    @abstractmethod
    def get_unit(cls) -> str:
        """Return the unit of the measurements, used as the suffix of the serialized keys."""

    @classmethod
    @abstractmethod
    def get_num_decimals(cls) -> int:
        """Return the number of decimals each reservoir sample is rounded to when written."""

    def get_mean(self) -> float:
        """Return the mean of the measurements, or 0.0 when there are none."""
        if self.count == 0:
            return 0.0
        return self.total / self.count

    def get_samples(self) -> list[float]:
        """Return a copy of the reservoir sample of measurements."""
        return list(self._reservoir)

    def copy(self) -> Self:
        """Return an independent copy that later measurements do not change."""
        return replace(self, _reservoir=list(self._reservoir))

    def update(self, value: float) -> None:
        """Record a new measurement."""
        self.count += 1
        self.total += value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

        # Reservoir sampling (Algorithm R). Maintain a small representative sample and compute
        # percentiles on it as a proxy, avoiding unbounded memory growth.
        if len(self._reservoir) < self.percentile_approximation_reservoir_size:
            self._reservoir.append(value)
        else:
            replacement_index = _reservoir_random_generator.randint(0, self.count - 1)
            if replacement_index < self.percentile_approximation_reservoir_size:
                self._reservoir[replacement_index] = value

    def percentile(self, p: float) -> float | None:
        """Return the approximate p-th percentile (0-100), or None if nothing was recorded.

        Args:
            p: Percentile to estimate, between 0 and 100.

        Returns:
            The estimated percentile, or None when no measurements exist.
        """
        assert 0 <= p <= 100, f"Percentile must be between 0 and 100, got {p}"
        if not self._reservoir:
            return None
        return weighted_percentile([(sample, 1.0) for sample in self._reservoir], p)

    def to_json_fields(self) -> dict[str, float | int | list[float] | None]:
        """Return the statistics as unit-suffixed JSON fields.

        Returns:
            The count, mean, total, min, max, one field per entry of PERCENTILES and the rounded reservoir samples.
        """
        unit = self.get_unit()
        num_decimals = self.get_num_decimals()
        fields: dict[str, float | int | list[float] | None] = {
            "count": self.count,
            f"mean_{unit}": self.get_mean(),
            f"total_{unit}": self.total,
            f"min_{unit}": self.minimum,
            f"max_{unit}": self.maximum,
        }
        for percentile in PERCENTILES:
            fields[f"p{percentile}_{unit}"] = self.percentile(percentile)
        fields[f"samples_{unit}"] = [round(sample, num_decimals) for sample in self._reservoir]
        return fields


def merge_statistics_json(
    records: Iterable[MutableMapping[str, Any]], unit: str, remove_samples_from_input_records: bool = False
) -> list[dict]:
    """Combine statistics records that share a name into one total per name, sorted by name.

    Use this to summarize records produced by several processes, such as one Experiment Runner per Run.
    When every record of a name carries "samples_<unit>", the merged record also holds approximate p10, p50
    and p90, estimated from the pooled samples with each sample weighted by the number of measurements it
    stands for. Otherwise the percentiles are left out, because they cannot be recovered from summaries.

    Args:
        records: Records with the fields of SampleStatistics.to_json_fields plus a "name".
        unit: Unit suffix of the records' keys, for example "ms".
        remove_samples_from_input_records: Whether to delete "samples_<unit>" from each input record once it
            is read, for callers that keep the records and do not want the samples in them.

    Returns:
        One record per distinct name, holding the combined count, total, mean, min, and max, plus the
        percentiles when available. The samples are not carried over.
    """
    name_key = "name"
    count_key = "count"
    total_key = f"total_{unit}"
    min_key = f"min_{unit}"
    max_key = f"max_{unit}"
    mean_key = f"mean_{unit}"
    samples_key = f"samples_{unit}"

    merged_by_name: dict[str, dict] = {}
    weighted_samples_by_name: dict[str, list[tuple[float, float]] | None] = {}
    for record in records:
        name = record[name_key]
        if remove_samples_from_input_records:
            samples = record.pop(samples_key, None)
        else:
            samples = record.get(samples_key)
        merged = merged_by_name.get(name)
        # Add the record's count, total, min and max to the running totals of its name.
        if merged is None:
            merged_by_name[name] = {
                name_key: name,
                count_key: record[count_key],
                total_key: record[total_key],
                min_key: record[min_key],
                max_key: record[max_key],
            }
            weighted_samples_by_name[name] = []
        else:
            merged[count_key] += record[count_key]
            merged[total_key] += record[total_key]
            merged[min_key] = min(merged[min_key], record[min_key])
            merged[max_key] = max(merged[max_key], record[max_key])

        # Collect the record's samples, each weighted by the measurements it stands for. A record without
        # samples makes the percentiles of its name unavailable.
        if samples is None:
            weighted_samples_by_name[name] = None
        elif weighted_samples_by_name[name] is not None and samples:
            weight = record[count_key] / len(samples)
            weighted_samples_by_name[name].extend((sample, weight) for sample in samples)

    results = []
    for name, merged in sorted(merged_by_name.items()):
        result = {**merged, mean_key: merged[total_key] / merged[count_key] if merged[count_key] else 0.0}
        weighted_samples = weighted_samples_by_name[name]
        if weighted_samples:
            for percentile in PERCENTILES:
                result[f"p{percentile}_{unit}"] = weighted_percentile(weighted_samples, percentile)
        results.append(result)
    return results

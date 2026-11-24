# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SampleStatistics and merge_statistics_json."""

import random
from dataclasses import dataclass

import pytest

from isaaclab_arena.utils.sample_statistics import SampleStatistics, merge_statistics_json


@dataclass
class _MillisecondStatistics(SampleStatistics):
    @classmethod
    def get_unit(cls) -> str:
        return "ms"

    @classmethod
    def get_num_decimals(cls) -> int:
        return 3


@dataclass
class _PercentStatistics(SampleStatistics):
    @classmethod
    def get_unit(cls) -> str:
        return "percent"

    @classmethod
    def get_num_decimals(cls) -> int:
        return 0


def test_initial_state() -> None:
    stats = _MillisecondStatistics()
    assert stats.count == 0
    assert stats.total == 0.0
    assert stats.minimum == float("inf")
    assert stats.maximum == float("-inf")
    assert stats.get_mean() == 0.0
    assert stats.get_samples() == []
    assert stats.percentile(50) is None


def test_accumulation() -> None:
    stats = _MillisecondStatistics()
    for value in (10.0, 20.0, 30.0):
        stats.update(value)
    assert stats.count == 3
    assert stats.total == 60.0
    assert stats.minimum == 10.0
    assert stats.maximum == 30.0
    assert stats.get_mean() == 20.0


def test_percentiles() -> None:
    stats = _MillisecondStatistics()
    for value in range(1, 101):
        stats.update(float(value))
    assert stats.percentile(10) == 10
    assert stats.percentile(50) == 50
    assert stats.percentile(90) == 90


def test_reservoir_is_bounded_and_leaves_global_random_state_alone() -> None:
    stats = _MillisecondStatistics(percentile_approximation_reservoir_size=5)
    state_before = random.getstate()
    for value in range(100):
        stats.update(float(value))
    assert random.getstate() == state_before
    assert len(stats.get_samples()) == 5
    assert stats.count == 100


def test_samples_is_a_copy() -> None:
    stats = _MillisecondStatistics()
    stats.update(1.0)
    stats.get_samples().append(2.0)
    assert stats.get_samples() == [1.0]


def test_to_json_fields_keys_follow_unit() -> None:
    for statistics_class, unit in ((_MillisecondStatistics, "ms"), (_PercentStatistics, "percent")):
        stats = statistics_class()
        for value in (1.23456, 2.0):
            stats.update(value)
        fields = stats.to_json_fields()
        assert list(fields) == [
            "count",
            f"mean_{unit}",
            f"total_{unit}",
            f"min_{unit}",
            f"max_{unit}",
            f"p10_{unit}",
            f"p50_{unit}",
            f"p90_{unit}",
            f"samples_{unit}",
        ]
        assert fields["count"] == 2
        assert fields[f"samples_{unit}"] == ([1.235, 2.0] if unit == "ms" else [1.0, 2.0])


def test_to_json_fields_of_empty_statistics() -> None:
    fields = _PercentStatistics().to_json_fields()
    assert fields["p10_percent"] is None
    assert fields["p50_percent"] is None
    assert fields["p90_percent"] is None
    assert fields["samples_percent"] == []


def _record(name: str, count: int, total: float, minimum: float, maximum: float, unit: str = "percent") -> dict:
    return {
        "name": name,
        "count": count,
        f"total_{unit}": total,
        f"min_{unit}": minimum,
        f"max_{unit}": maximum,
    }


def test_merge_empty_input() -> None:
    assert merge_statistics_json([], unit="ms") == []


def test_merge_combines_records_and_sorts_by_name() -> None:
    merged = merge_statistics_json(
        [
            _record("b", 1, 2.0, 2.0, 2.0),
            _record("a", 2, 10.0, 3.0, 7.0),
            _record("a", 3, 30.0, 1.0, 20.0),
        ],
        unit="percent",
    )
    assert merged == [
        {"name": "a", "count": 5, "total_percent": 40.0, "min_percent": 1.0, "max_percent": 20.0, "mean_percent": 8.0},
        {"name": "b", "count": 1, "total_percent": 2.0, "min_percent": 2.0, "max_percent": 2.0, "mean_percent": 2.0},
    ]


def test_merge_leaves_percentiles_out_when_a_record_lacks_samples() -> None:
    with_samples = {**_record("a", 2, 3.0, 1.0, 2.0), "samples_percent": [1.0, 2.0]}
    without_samples = _record("a", 1, 3.0, 3.0, 3.0)
    merged = merge_statistics_json([with_samples, without_samples], unit="percent")
    assert set(merged[0]) == {"name", "count", "total_percent", "min_percent", "max_percent", "mean_percent"}


def test_merge_pools_samples_weighted_by_count() -> None:
    many = {**_record("s", 1000, 1000.0, 1.0, 1.0, "ms"), "samples_ms": [1.0] * 2}
    few = {**_record("s", 50, 5000.0, 100.0, 100.0, "ms"), "samples_ms": [100.0] * 50}
    merged = merge_statistics_json([many, few], unit="ms")
    assert merged[0]["p50_ms"] == 1.0
    assert merged[0]["p90_ms"] == 1.0
    assert "samples_ms" not in merged[0]


def test_merge_pools_samples_of_equal_weight() -> None:
    first = {**_record("s", 5, 15.0, 1.0, 5.0), "samples_percent": [1, 2, 3, 4, 5]}
    second = {**_record("s", 5, 40.0, 6.0, 10.0), "samples_percent": [6, 7, 8, 9, 10]}
    merged = merge_statistics_json([first, second], unit="percent")
    assert (merged[0]["p10_percent"], merged[0]["p50_percent"], merged[0]["p90_percent"]) == (1, 5, 9)


def test_merge_removes_samples_from_inputs_only_when_asked() -> None:
    record = {**_record("s", 1, 1.0, 1.0, 1.0), "samples_percent": [1.0]}
    merge_statistics_json([record], unit="percent")
    assert record["samples_percent"] == [1.0]
    merge_statistics_json([record], unit="percent", remove_samples_from_input_records=True)
    assert "samples_percent" not in record


def test_subclass_missing_get_unit_cannot_be_instantiated() -> None:
    @dataclass
    class _MissingUnit(SampleStatistics):
        @classmethod
        def get_num_decimals(cls) -> int:
            return 0

    with pytest.raises(TypeError, match="get_unit"):
        _MissingUnit()


def test_subclass_missing_get_num_decimals_cannot_be_instantiated() -> None:
    @dataclass
    class _MissingDecimals(SampleStatistics):
        @classmethod
        def get_unit(cls) -> str:
            return "ms"

    with pytest.raises(TypeError, match="get_num_decimals"):
        _MissingDecimals()


def test_num_decimals_rounds_samples_but_not_summary_fields() -> None:
    @dataclass
    class _TwoDecimals(_MillisecondStatistics):
        @classmethod
        def get_num_decimals(cls) -> int:
            return 2

    stats = _TwoDecimals()
    for value in (1.23456, 2.34567):
        stats.update(value)
    fields = stats.to_json_fields()
    assert fields["samples_ms"] == [1.23, 2.35]
    assert fields["min_ms"] == 1.23456
    assert fields["total_ms"] == pytest.approx(3.58023)


def test_zero_decimal_samples_serialize_and_merge() -> None:
    stats = _PercentStatistics()
    for value in (0.2, 49.6, 100.4):
        stats.update(value)
    record = {"name": "u", **stats.to_json_fields()}
    assert record["samples_percent"] == [0.0, 50.0, 100.0]
    merged = merge_statistics_json([record], unit="percent")
    assert merged[0]["p50_percent"] == 50.0
    assert merged[0]["max_percent"] == 100.4

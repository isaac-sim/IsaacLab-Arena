# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ResourceUsage and its JSON records."""

import json

from isaaclab_arena.utils.resource_usage import (
    RESOURCE_NAMES,
    RESOURCE_UNITS,
    MegabyteStatistics,
    PercentStatistics,
    ResourceUsage,
    merge_resource_records,
)


def _usage(vram=(), ram=(), cpu_util=(), gpu_util=()) -> ResourceUsage:
    usage = ResourceUsage()
    for value in vram:
        usage.vram.update(value)
    for value in ram:
        usage.ram.update(value)
    for value in cpu_util:
        usage.cpu_util.update(value)
    for value in gpu_util:
        usage.gpu_util.update(value)
    return usage


def test_constants() -> None:
    assert RESOURCE_NAMES == ("vram", "ram", "cpu_util", "gpu_util")
    assert RESOURCE_UNITS == {"vram": "mb", "ram": "mb", "cpu_util": "percent", "gpu_util": "percent"}


def test_statistics_classes_define_unit_and_decimals() -> None:
    assert (MegabyteStatistics.get_unit(), MegabyteStatistics.get_num_decimals()) == ("mb", 0)
    assert (PercentStatistics.get_unit(), PercentStatistics.get_num_decimals()) == ("percent", 0)


def test_default_instances_are_independent() -> None:
    first = ResourceUsage()
    second = ResourceUsage()
    first.vram.update(1.0)
    assert second.vram.count == 0


def test_records_only_for_resources_with_samples() -> None:
    records = _usage(vram=[100.0, 300.0], gpu_util=[50.0]).to_json_records("app")
    assert [record["name"] for record in records] == ["vram", "gpu_util"]


def test_empty_usage_has_no_records() -> None:
    assert ResourceUsage().to_json_records("app") == []


def test_record_keys_follow_unit_and_order_follows_resource_names() -> None:
    usage = _usage(vram=[1.0], ram=[2.0], cpu_util=[3.0], gpu_util=[4.0])
    records = usage.to_json_records("my_app")
    assert [record["name"] for record in records] == list(RESOURCE_NAMES)
    vram, _, cpu, _ = records
    assert vram["type"] == "resource_usage" and vram["app_name"] == "my_app"
    assert "max_mb" in vram and "samples_mb" in vram
    assert "mean_percent" in cpu and "p90_percent" in cpu and "max_mb" not in cpu


def test_merge_takes_max_of_maxima_and_count_weighted_mean() -> None:
    run_a = _usage(vram=[100.0, 200.0]).to_json_records("a")
    run_b = _usage(vram=[400.0, 600.0, 800.0, 1000.0]).to_json_records("a")
    merged = merge_resource_records(run_a + run_b)
    assert len(merged) == 1
    record = merged[0]
    assert record["name"] == "vram"
    assert record["count"] == 6
    assert record["max_mb"] == 1000.0
    assert record["min_mb"] == 100.0
    assert record["mean_mb"] == (300.0 + 2800.0) / 6


def test_merge_handles_mixed_units_in_one_call() -> None:
    records = _usage(vram=[10.0], cpu_util=[50.0]).to_json_records("a")
    merged = merge_resource_records(records)
    by_name = {record["name"]: record for record in merged}
    assert by_name["vram"]["max_mb"] == 10.0
    assert by_name["cpu_util"]["max_percent"] == 50.0


def test_merge_can_remove_samples_from_input() -> None:
    records = _usage(ram=[1.0, 2.0]).to_json_records("a")
    merge_resource_records(records, remove_samples_from_input_records=True)
    assert "samples_mb" not in records[0]


def test_records_dump_with_allow_nan_false() -> None:
    usage = _usage(vram=[1.0, 2.0], ram=[3.0], cpu_util=[0.0], gpu_util=[99.5])
    json.dumps(usage.to_json_records("a"), allow_nan=False)
    json.dumps(merge_resource_records(usage.to_json_records("a")), allow_nan=False)

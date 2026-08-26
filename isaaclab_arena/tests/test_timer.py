# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Timer context manager and TimerStats."""

import json
import random

import pytest

from isaaclab_arena.utils.timer import (
    Timer,
    TimerStats,
    get_timer_stats,
    get_timer_stats_json,
    merge_timer_stats_json,
    print_timer_stats,
    reset_timer_stats,
    write_timer_stats_json,
)


class TestTimerStats:
    """Tests for the TimerStats dataclass."""

    def test_initial_state(self) -> None:
        """Verify default state of a fresh TimerStats."""
        stats = TimerStats()
        assert stats.count == 0
        assert stats.total_ms == 0.0
        assert stats.min_ms == float("inf")
        assert stats.max_ms == float("-inf")
        assert stats.mean_ms == 0.0

    def test_accumulation(self) -> None:
        """Verify min/max/mean/total after multiple measurements."""
        stats = TimerStats()
        stats.update(10.0)
        stats.update(20.0)
        stats.update(30.0)
        assert stats.count == 3
        assert stats.total_ms == 60.0
        assert stats.min_ms == 10.0
        assert stats.max_ms == 30.0
        assert stats.mean_ms == pytest.approx(20.0)

    def test_percentile_empty(self) -> None:
        """Verify percentile returns None when no data recorded."""
        stats = TimerStats()
        assert stats.percentile(50) is None

    def test_percentile_values(self) -> None:
        """Verify approximate percentiles on a known distribution."""
        stats = TimerStats()
        for i in range(1, 101):
            stats.update(float(i))
        assert stats.percentile(10) == 10.0
        assert stats.percentile(50) == 50.0
        assert stats.percentile(90) == 90.0

    def test_reservoir_accuracy(self) -> None:
        """Verify reservoir sampling percentiles track the exact ones on skewed data."""
        seed = 32
        # Seed both the module RNG used by TimerStats reservoir sampling and the local RNG
        # used to generate this test's synthetic timing data.
        random.seed(seed)
        rng = random.Random(seed)
        num_samples = 10_000

        values = [rng.gauss(20.0, 5.0) for _ in range(num_samples)]
        # Sprinkle in ~1% outliers.
        for _ in range(num_samples // 100):
            values.append(rng.uniform(80.0, 200.0))

        exact = TimerStats(reservoir_size=len(values))
        approx = TimerStats()
        for value in values:
            exact.update(value)
            approx.update(value)

        for p in (10, 50, 90):
            assert approx.percentile(p) == pytest.approx(exact.percentile(p), rel=0.05)


class TestTimer:
    """Tests for the Timer context manager."""

    def setup_method(self) -> None:
        """Reset timer registry before each test."""
        reset_timer_stats()

    def test_basic_timing(self) -> None:
        """Verify a single timer records one measurement."""
        with Timer("test_op"):
            pass

        stats = get_timer_stats()
        assert "test_op" in stats
        assert stats["test_op"].count == 1
        assert stats["test_op"].total_ms >= 0.0

    def test_multiple_names(self) -> None:
        """Verify distinct timer names produce separate stats."""
        with Timer("op_a"):
            pass
        with Timer("op_b"):
            pass

        stats = get_timer_stats()
        assert stats["op_a"].count == 1
        assert stats["op_b"].count == 1

    def test_accumulation(self) -> None:
        """Verify repeated use of the same timer name accumulates."""
        for _ in range(5):
            with Timer("repeated"):
                pass

        assert get_timer_stats()["repeated"].count == 5

    def test_nesting(self) -> None:
        """Verify nested timers both record and outer >= inner."""
        with Timer("outer"):
            with Timer("inner"):
                pass

        stats = get_timer_stats()
        assert stats["outer"].total_ms >= stats["inner"].total_ms

    def test_exception_propagation(self) -> None:
        """Verify exceptions propagate and stats are still recorded."""
        with pytest.raises(ValueError, match="test error"):
            with Timer("failing_op"):
                raise ValueError("test error")

        assert get_timer_stats()["failing_op"].count == 1

    def test_timer_returns_self(self) -> None:
        """Verify the context manager yields the Timer instance."""
        with Timer("self_test") as timer:
            assert timer.name == "self_test"


class TestPrintTimerStats:
    """Tests for print_timer_stats output formatting."""

    def setup_method(self) -> None:
        """Reset timer registry before each test."""
        reset_timer_stats()

    def test_empty_stats(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify output when no timers have been recorded."""
        print_timer_stats()
        assert "No timer stats recorded." in capsys.readouterr().out

    def test_output_format(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify the printed table has header, separator, and data row."""
        with Timer("my_operation"):
            pass

        print_timer_stats()
        lines = capsys.readouterr().out.strip().split("\n")
        assert len(lines) == 4  # units line, header, separator, one data row
        assert "Name" in lines[1]
        assert "p50" in lines[1]
        assert "my_operation" in lines[3]


class TestGetTimerStatsJson:
    """Tests for get_timer_stats_json."""

    def setup_method(self) -> None:
        """Reset timer registry before each test."""
        reset_timer_stats()

    def test_records(self) -> None:
        """Verify one record per timer, each tagged with the app name and expected keys."""
        with Timer("op_a"):
            pass
        with Timer("op_b"):
            pass

        records = get_timer_stats_json(app_name="test_app")
        assert len(records) == 2
        assert all(record["app_name"] == "test_app" for record in records)
        assert all(record["type"] == "timing" for record in records)
        assert set(records[0].keys()) == {
            "type",
            "name",
            "app_name",
            "count",
            "mean_ms",
            "total_ms",
            "min_ms",
            "max_ms",
            "p10_ms",
            "p50_ms",
            "p90_ms",
        }


class TestMergeTimerStatsJson:
    """Tests for merge_timer_stats_json."""

    @staticmethod
    def _record(name: str, count: int, total_ms: float, min_ms: float, max_ms: float) -> dict:
        return {
            "type": "timing",
            "name": name,
            "app_name": "some_app",
            "count": count,
            "mean_ms": total_ms / count,
            "total_ms": total_ms,
            "min_ms": min_ms,
            "max_ms": max_ms,
            "p50_ms": min_ms,
        }

    def test_empty_input(self) -> None:
        """Verify merging nothing produces nothing."""
        assert merge_timer_stats_json([]) == []

    def test_combines_records_that_share_a_name(self) -> None:
        """Verify counts and totals add while min and max span every record."""
        merged = merge_timer_stats_json([
            self._record("step", count=2, total_ms=10.0, min_ms=3.0, max_ms=7.0),
            self._record("step", count=3, total_ms=30.0, min_ms=1.0, max_ms=20.0),
        ])

        assert merged == [{"name": "step", "count": 5, "total_ms": 40.0, "min_ms": 1.0, "max_ms": 20.0, "mean_ms": 8.0}]

    def test_separate_names_sorted(self) -> None:
        """Verify distinct names stay separate and come back sorted by name."""
        merged = merge_timer_stats_json([
            self._record("b_op", count=1, total_ms=2.0, min_ms=2.0, max_ms=2.0),
            self._record("a_op", count=1, total_ms=1.0, min_ms=1.0, max_ms=1.0),
        ])

        assert [record["name"] for record in merged] == ["a_op", "b_op"]

    def test_percentiles_are_dropped(self) -> None:
        """Verify percentile fields are not carried into the merged record."""
        merged = merge_timer_stats_json([self._record("step", count=1, total_ms=5.0, min_ms=5.0, max_ms=5.0)])

        assert set(merged[0]) == {"name", "count", "total_ms", "min_ms", "max_ms", "mean_ms"}

    def test_merges_recorded_stats(self) -> None:
        """Verify merging one process's own records reproduces its totals."""
        reset_timer_stats()
        for _ in range(3):
            with Timer("op"):
                pass

        records = get_timer_stats_json(app_name="test_app")
        merged = merge_timer_stats_json(records)
        assert merged[0]["count"] == 3
        assert merged[0]["total_ms"] == pytest.approx(records[0]["total_ms"])


class TestWriteTimerStatsJson:
    """Tests for write_timer_stats_json."""

    def setup_method(self) -> None:
        """Reset timer registry before each test."""
        reset_timer_stats()

    def test_written_file_round_trips(self, tmp_path) -> None:
        """Verify the written file parses back to the in-memory records."""
        with Timer("op_a"):
            pass

        output_path = write_timer_stats_json(tmp_path / "timings.json", app_name="test_app")
        written_records = json.loads(output_path.read_text(encoding="utf-8"))
        assert written_records == get_timer_stats_json(app_name="test_app")
        assert written_records[0]["name"] == "op_a"

    def test_empty_registry_writes_empty_list(self, tmp_path) -> None:
        """Verify a file is still written when no timers were recorded."""
        output_path = write_timer_stats_json(tmp_path / "timings.json", app_name="test_app")
        assert json.loads(output_path.read_text(encoding="utf-8")) == []


class TestResetTimerStats:
    """Tests for reset_timer_stats."""

    def test_reset_clears_all(self) -> None:
        """Verify reset removes all recorded stats and new timers start fresh."""
        with Timer("first_run"):
            pass

        assert len(get_timer_stats()) > 0
        reset_timer_stats()
        assert len(get_timer_stats()) == 0

        with Timer("second_run"):
            pass

        stats = get_timer_stats()
        assert "first_run" not in stats
        assert stats["second_run"].count == 1

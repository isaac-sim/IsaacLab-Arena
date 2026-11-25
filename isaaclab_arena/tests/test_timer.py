# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Timer context manager and TimerStats."""

import json
import threading
from unittest.mock import patch

import pytest

import isaaclab_arena.utils.timer as timer_module
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

    def test_unit_and_decimals(self) -> None:
        """Verify timings are in milliseconds with microsecond sample resolution."""
        assert TimerStats.get_unit() == "ms"
        assert TimerStats.get_num_decimals() == 3


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
        assert stats["test_op"].total >= 0.0

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
        assert stats["outer"].total >= stats["outer/inner"].total

    def test_nested_name_is_qualified_by_its_enclosing_timers(self) -> None:
        """Verify each nesting level appends its name to the enclosing name."""
        with Timer("a"):
            with Timer("b"):
                with Timer("c"):
                    pass

        assert set(get_timer_stats()) == {"a", "a/b", "a/b/c"}

    def test_same_name_records_separately_per_nesting_position(self) -> None:
        """Verify a name used at two depths does not collapse into one entry."""
        with Timer("step"):
            pass
        with Timer("outer"):
            with Timer("step"):
                pass

        stats = get_timer_stats()
        assert stats["step"].count == 1
        assert stats["outer/step"].count == 1

    def test_siblings_share_the_enclosing_name(self) -> None:
        """Verify timers nested side by side each hang off the same enclosing name."""
        with Timer("step"):
            with Timer("policy"):
                pass
            with Timer("env"):
                pass

        assert set(get_timer_stats()) == {"step", "step/policy", "step/env"}

    def test_nesting_unwinds_when_an_inner_timer_raises(self) -> None:
        """Verify a timer that exits on an exception still leaves the enclosing name correct."""
        with Timer("outer"):
            with pytest.raises(ValueError, match="inner error"):
                with Timer("failing"):
                    raise ValueError("inner error")
            with Timer("after"):
                pass

        assert set(get_timer_stats()) == {"outer", "outer/failing", "outer/after"}

    def test_name_returns_to_top_level_after_the_enclosing_timer_exits(self) -> None:
        """Verify the enclosing name is not leaked into timers entered after it closes."""
        with Timer("outer"):
            with Timer("inner"):
                pass

        with Timer("later"):
            pass

        assert "later" in get_timer_stats()

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

    def test_timers_on_several_threads_record_every_measurement(self) -> None:
        """Verify concurrent worker-thread timers record under their own names without losing measurements."""
        num_threads = 8
        num_timings_per_thread = 200

        def time_blocks() -> None:
            for _ in range(num_timings_per_thread):
                with Timer("worker"):
                    pass

        with Timer("main"):
            threads = [threading.Thread(target=time_blocks) for _ in range(num_threads)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        stats = get_timer_stats()
        assert set(stats) == {"main", "worker"}
        assert stats["worker"].count == num_threads * num_timings_per_thread

    def test_stats_snapshot_does_not_change_with_later_timings(self) -> None:
        """Verify get_timer_stats returns copies that later measurements leave unchanged."""
        with Timer("op"):
            pass
        snapshot = get_timer_stats()
        with Timer("op"):
            pass

        assert snapshot["op"].count == 1
        assert len(snapshot["op"].get_samples()) == 1
        assert get_timer_stats()["op"].count == 2


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
            "samples_ms",
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

    def test_single_record_percentiles_match_the_timer_percentiles(self) -> None:
        """Verify merging one record gives the percentiles the timer reported for it."""
        timer_stats = TimerStats()
        for elapsed_ms in (1.0, 2.0, 100.0):
            timer_stats.update(elapsed_ms)
        with patch("isaaclab_arena.utils.timer.get_timer_stats", return_value={"step": timer_stats}):
            records = get_timer_stats_json(app_name="test_app")

        merged = merge_timer_stats_json(records)

        for key, percentile in (("p10_ms", 10), ("p50_ms", 50), ("p90_ms", 90)):
            assert merged[0][key] == records[0][key] == timer_stats.percentile(percentile)

    def test_percentiles_are_left_out_when_one_record_has_no_samples(self) -> None:
        """Verify a record without samples makes the pooled percentiles unavailable for its name."""
        with_samples = {**self._record("step", count=1, total_ms=1.0, min_ms=1.0, max_ms=1.0), "samples_ms": [1.0]}
        without_samples = self._record("step", count=1, total_ms=2.0, min_ms=2.0, max_ms=2.0)

        # The order matters: a record without samples may come before or after the ones that have them.
        for records in ([with_samples, without_samples], [without_samples, with_samples]):
            merged = merge_timer_stats_json(records)

            assert "p50_ms" not in merged[0]
            assert merged[0]["count"] == 2

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
        """Verify the written file parses back to the in-memory records, samples included."""
        with Timer("op_a"):
            pass

        output_path = write_timer_stats_json(tmp_path / "timings.json", app_name="test_app")
        written_records = json.loads(output_path.read_text(encoding="utf-8"))
        assert written_records == get_timer_stats_json(app_name="test_app")
        assert written_records[0]["name"] == "op_a"
        assert len(written_records[0]["samples_ms"]) == 1

    def test_samples_are_rounded_and_on_one_line(self, tmp_path) -> None:
        """Verify each samples list is one line of values rounded to microseconds."""
        timer_stats = TimerStats()
        for elapsed_ms in (1.23456789, 2.0, 3.5):
            timer_stats.update(elapsed_ms)

        with patch("isaaclab_arena.utils.timer.get_timer_stats", return_value={"op_a": timer_stats}):
            output_path = write_timer_stats_json(tmp_path / "timings.json", app_name="test_app")

        text = output_path.read_text(encoding="utf-8")
        assert '"samples_ms": [1.235, 2.0, 3.5]' in text
        assert json.loads(text)[0]["samples_ms"] == [1.235, 2.0, 3.5]

    def test_written_file_merges_to_percentiles(self, tmp_path) -> None:
        """Verify records read back from the written file merge into percentiles."""
        for elapsed_ms in range(1, 11):
            timer_module._update_stats("op_a", float(elapsed_ms))

        output_path = write_timer_stats_json(tmp_path / "timings.json", app_name="test_app")
        merged = merge_timer_stats_json(json.loads(output_path.read_text(encoding="utf-8")))

        assert merged[0]["p50_ms"] == 5.0
        assert merged[0]["p90_ms"] == 9.0

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

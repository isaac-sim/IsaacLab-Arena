# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the rollout timing plot. No Isaac Sim or GPU required."""

import json
from pathlib import Path

import pytest

from isaaclab_arena.visualization.plot_timings import (
    build_run_timings,
    collect_run_timings,
    main,
    plot_run_timings,
    read_run_timer_records,
)


def _timer_record(name: str, count: int, total_ms: float, with_percentiles: bool = True) -> dict:
    """Build one record in the shape get_timer_stats_json writes."""
    mean_ms = total_ms / count
    record = {
        "type": "timing",
        "name": name,
        "app_name": "experiment_runner",
        "count": count,
        "mean_ms": mean_ms,
        "total_ms": total_ms,
        "min_ms": mean_ms,
        "max_ms": mean_ms,
    }
    if with_percentiles:
        record.update({"p10_ms": mean_ms, "p50_ms": mean_ms, "p90_ms": mean_ms * 2})
    return record


def _nested_records(steps: int = 10, with_percentiles: bool = True) -> list[dict]:
    """Records for a step that splits into policy inference and an env step with two children."""
    return [
        _timer_record("step", steps, 100.0 * steps, with_percentiles),
        _timer_record("step/policy_inference", steps, 20.0 * steps, with_percentiles),
        _timer_record("step/env_step", steps, 70.0 * steps, with_percentiles),
        _timer_record("step/env_step/sim_step", steps, 40.0 * steps, with_percentiles),
        _timer_record("step/env_step/record_camera_frames", steps, 25.0 * steps, with_percentiles),
    ]


def _write_timings(path: Path, records: list[dict], run_names: list[str] | None = None) -> Path:
    """Write either Experiment Runner shape (a bare list) or collected shape (runs keyed by name)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if run_names is None:
        document: object = records
    else:
        document = {
            "totals": [],
            "runs": [{"run_name": run_name, **record} for run_name in run_names for record in records],
        }
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def test_reads_the_experiment_runner_shape_and_names_the_run_after_its_directory(tmp_path):
    timings_path = _write_timings(tmp_path / "banana_in_bowl_pi0/timings.json", _nested_records())

    records_by_run_name = read_run_timer_records(timings_path)

    assert list(records_by_run_name) == ["banana_in_bowl_pi0"]
    assert len(records_by_run_name["banana_in_bowl_pi0"]) == 5


def test_reads_the_collected_shape_and_groups_by_run_name(tmp_path):
    timings_path = _write_timings(tmp_path / "timings.json", _nested_records(), run_names=["first", "second"])

    records_by_run_name = read_run_timer_records(timings_path)

    assert sorted(records_by_run_name) == ["first", "second"]
    assert len(records_by_run_name["first"]) == 5


def test_splits_the_root_timer_into_children_then_its_own_leftover():
    """Each parent contributes its children first, then whatever it did not attribute to them."""
    run_timings = build_run_timings("run", _nested_records(steps=10), "step")

    assert [segment.label for segment in run_timings.segments] == [
        "policy_inference",
        "env_step/sim_step",
        "env_step/record_camera_frames",
        "env_step (other)",
        "step (other)",
    ]


def test_normalizes_each_slice_to_one_root_timer_call():
    """Slices are per-step figures, and together they add back up to the mean step."""
    run_timings = build_run_timings("run", _nested_records(steps=10), "step")

    slice_ms_by_label = {segment.label: segment.ms_per_call for segment in run_timings.segments}
    assert slice_ms_by_label["policy_inference"] == pytest.approx(20.0)
    assert slice_ms_by_label["env_step/sim_step"] == pytest.approx(40.0)
    assert slice_ms_by_label["env_step/record_camera_frames"] == pytest.approx(25.0)
    # env_step is 70 with 65 attributed to its children, and the step is 100 with 90 attributed.
    assert slice_ms_by_label["env_step (other)"] == pytest.approx(5.0)
    assert slice_ms_by_label["step (other)"] == pytest.approx(10.0)
    assert sum(slice_ms_by_label.values()) == pytest.approx(run_timings.mean_ms)
    assert run_timings.call_count == 10


def test_a_timer_below_the_root_can_be_used_as_the_root():
    """Passing a nested timer plots just that subtree, with labels relative to it."""
    run_timings = build_run_timings("run", _nested_records(steps=10), "step/env_step")

    assert [segment.label for segment in run_timings.segments] == [
        "sim_step",
        "record_camera_frames",
        "step/env_step (other)",
    ]
    assert run_timings.mean_ms == pytest.approx(70.0)


def test_rejects_a_file_whose_children_outweigh_their_parent():
    records = _nested_records(steps=10)
    records[3]["total_ms"] = 10_000.0  # sim_step alone now exceeds its parent env_step

    with pytest.raises(ValueError, match="inconsistent"):
        build_run_timings("run", records, "step")


def test_rejects_a_run_without_the_root_timer():
    records = [record for record in _nested_records() if record["name"] != "step"]

    with pytest.raises(ValueError, match="has no 'step' timer"):
        build_run_timings("run", records, "step")


def test_rejects_a_file_that_is_neither_shape(tmp_path):
    timings_path = tmp_path / "timings.json"
    timings_path.write_text(json.dumps({"totals": []}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no 'runs' entry"):
        read_run_timer_records(timings_path)


def test_orders_runs_by_name(tmp_path):
    timings_path = _write_timings(tmp_path / "timings.json", _nested_records(), run_names=["zulu", "alpha"])

    run_timings = collect_run_timings(timings_path, "step")

    assert [run.run_name for run in run_timings] == ["alpha", "zulu"]


def test_writes_a_plot_without_percentiles(tmp_path):
    """The spread panel needs percentiles, which merged timings drop, but the breakdown still plots."""
    records = _nested_records(with_percentiles=False)
    run_timings = [build_run_timings("run", records, "step")]
    assert run_timings[0].p50_ms is None

    output_path = plot_run_timings(run_timings, tmp_path / "nested/timings.png", "Title")

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_main_writes_the_plot_beside_the_input(tmp_path, capsys):
    timings_path = _write_timings(tmp_path / "timings.json", _nested_records(steps=7), run_names=["first", "second"])

    return_code = main([str(timings_path)])

    assert return_code == 0
    assert (tmp_path / "timings.png").is_file()
    printed = capsys.readouterr().out
    assert "14 'step' calls across 2 run(s)" in printed

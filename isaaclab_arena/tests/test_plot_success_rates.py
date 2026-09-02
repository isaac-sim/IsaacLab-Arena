# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
import tarfile
from pathlib import Path

import pytest

from isaaclab_arena.visualization.plot_success_rates import (
    AverageProgressResult,
    SuccessRateResult,
    collect_average_progress,
    collect_success_rates,
    main,
    plot_average_progress,
    plot_success_rates,
)


def _write_episode_results(results_path: Path, job_name: str, successes: list[bool | None]) -> None:
    """Write minimal episode records for one Run."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    records = [json.dumps({"job_name": job_name, "success": success}) for success in successes]
    results_path.write_text("\n".join(records) + "\n", encoding="utf-8")


def _write_progress_results(results_path: Path, job_name: str, overall_scores: list[float | None]) -> None:
    """Write episode records carrying a ``progress`` block for one Run.

    Each element is either None (progress not recorded) or the episode's ``overall_score`` (already
    normalized to [0, 1] by the progress tracker).
    """
    results_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for overall_score in overall_scores:
        value = None if overall_score is None else {"overall_score": overall_score}
        records.append(json.dumps({"job_name": job_name, "success": None, "progress": value}))
    results_path.write_text("\n".join(records) + "\n", encoding="utf-8")


def test_collect_success_rates_aggregates_rebuilds_by_episode_count(tmp_path):
    results_directory = tmp_path / "results"
    _write_episode_results(
        results_directory / "task_alpha_pi0/episode_results_rebuild0.jsonl", "task_alpha_pi0", [True, False]
    )
    _write_episode_results(
        results_directory / "task_alpha_pi0/episode_results_rebuild1.jsonl", "task_alpha_pi0", [True, None]
    )
    _write_episode_results(
        results_directory / "task_alpha_cosmos/episode_results_rebuild0.jsonl", "task_alpha_cosmos", [False, False]
    )

    results = collect_success_rates(results_directory, ["pi0", "cosmos"])

    assert results[("task_alpha", "pi0")] == SuccessRateResult(successful_episodes=2, total_episodes=3)
    assert results[("task_alpha", "cosmos")] == SuccessRateResult(successful_episodes=0, total_episodes=2)


def test_collect_success_rates_reads_tar_archive(tmp_path):
    archive_contents = tmp_path / "archive_contents"
    _write_episode_results(
        archive_contents / "task_alpha_pi0/episode_results_rebuild0.jsonl", "task_alpha_pi0", [True, False]
    )
    _write_episode_results(
        archive_contents / "task_alpha_cosmos/episode_results_rebuild0.jsonl", "task_alpha_cosmos", [True]
    )
    archive_path = tmp_path / "results.tar.xz"
    with tarfile.open(archive_path, mode="w:xz") as archive:
        archive.add(archive_contents, arcname="results")

    results = collect_success_rates(archive_path, ["cosmos", "pi0"])

    assert results[("task_alpha", "pi0")].success_rate == pytest.approx(0.5)
    assert results[("task_alpha", "cosmos")].success_rate == pytest.approx(1.0)


def test_collect_success_rates_uses_most_recent_dated_experiment_directory(tmp_path):
    output_directory = tmp_path / "output"
    _write_episode_results(
        output_directory / "2026-08-06_10-00-00/task_alpha_pi0/episode_results.jsonl", "task_alpha_pi0", [False]
    )
    _write_episode_results(
        output_directory / "2026-08-07_10-00-00/task_alpha_pi0/episode_results.jsonl", "task_alpha_pi0", [True]
    )

    results = collect_success_rates(output_directory, ["pi0"])

    assert results[("task_alpha", "pi0")].success_rate == pytest.approx(1.0)


def test_collect_success_rates_uses_most_recent_dated_experiment_in_archive(tmp_path):
    archive_contents = tmp_path / "archive_contents"
    _write_episode_results(
        archive_contents / "2026-08-06_10-00-00/task_alpha_pi0/episode_results.jsonl", "task_alpha_pi0", [False]
    )
    _write_episode_results(
        archive_contents / "2026-08-07_10-00-00/task_alpha_pi0/episode_results.jsonl", "task_alpha_pi0", [True]
    )
    archive_path = tmp_path / "results.tar.xz"
    with tarfile.open(archive_path, mode="w:xz") as archive:
        archive.add(archive_contents, arcname="results")

    results = collect_success_rates(archive_path, ["pi0"])

    assert results[("task_alpha", "pi0")].success_rate == pytest.approx(1.0)


def test_collect_success_rates_preserves_task_with_only_unscored_episodes(tmp_path):
    results_directory = tmp_path / "results"
    _write_episode_results(results_directory / "task_alpha_pi0/episode_results.jsonl", "task_alpha_pi0", [True])
    _write_episode_results(results_directory / "task_beta_pi0/episode_results.jsonl", "task_beta_pi0", [None])

    results = collect_success_rates(results_directory, ["pi0"])

    assert results[("task_beta", "pi0")] == SuccessRateResult(successful_episodes=0, total_episodes=0)
    assert results[("task_beta", "pi0")].success_rate is None


def test_collect_success_rates_rejects_non_boolean_success(tmp_path):
    results_path = tmp_path / "episode_results.jsonl"
    results_path.write_text(json.dumps({"job_name": "task_alpha_pi0", "success": "false"}), encoding="utf-8")

    with pytest.raises(ValueError, match="not a boolean or null"):
        collect_success_rates(results_path, ["pi0"])


def test_collect_success_rates_rejects_run_without_requested_policy_suffix(tmp_path):
    results_path = tmp_path / "episode_results.jsonl"
    _write_episode_results(results_path, "task_alpha_cosmos", [True])

    with pytest.raises(ValueError, match="does not end with a requested policy suffix"):
        collect_success_rates(results_path, ["pi0"])


def test_plot_success_rates_writes_png_with_missing_policy_pair(tmp_path):
    results = {
        ("task_alpha", "pi0"): SuccessRateResult(successful_episodes=1, total_episodes=2),
        ("task_alpha", "cosmos"): SuccessRateResult(successful_episodes=2, total_episodes=2),
        ("task_beta", "pi0"): SuccessRateResult(successful_episodes=0, total_episodes=1),
    }
    output_path = tmp_path / "success_rates.png"

    plot_success_rates(results, ["pi0", "cosmos"], output_path)

    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_main_skips_metric_without_data_and_reports_unavailable_pair(tmp_path, capsys):
    results_directory = tmp_path / "results"
    # Only success is recorded, so the average-progress metric is skipped (not an error).
    _write_episode_results(results_directory / "task_alpha_pi0/episode_results.jsonl", "task_alpha_pi0", [True])

    return_code = main([str(results_directory), "--policies", "pi0", "cosmos"])

    captured_output = capsys.readouterr()
    assert return_code == 0
    assert (results_directory / "success_rates.png").is_file()
    assert not (results_directory / "average_progress.png").exists()
    assert "[success_rate] Read 1 scored episodes across 1 tasks and 2 policies." in captured_output.out
    assert "[average_progress] skipped" in captured_output.err
    assert "Unavailable results (bars marked N/A): task_alpha/cosmos" in captured_output.err


def test_collect_average_progress_averages_recorded_overall_score(tmp_path):
    results_directory = tmp_path / "results"
    _write_progress_results(
        results_directory / "task_alpha_pi0/episode_results_rebuild0.jsonl",
        "task_alpha_pi0",
        [1.0, 0.5],
    )
    _write_progress_results(
        results_directory / "task_alpha_pi0/episode_results_rebuild1.jsonl",
        "task_alpha_pi0",
        [0.0, None],  # one scored-at-zero episode and one unrecorded episode
    )
    _write_progress_results(
        results_directory / "task_alpha_cosmos/episode_results_rebuild0.jsonl",
        "task_alpha_cosmos",
        [0.25],
    )

    results = collect_average_progress(results_directory, ["pi0", "cosmos"])

    assert results[("task_alpha", "pi0")] == AverageProgressResult(total_progress=1.5, scored_episodes=3)
    assert results[("task_alpha", "pi0")].average_progress == pytest.approx(0.5)
    assert results[("task_alpha", "cosmos")].average_progress == pytest.approx(0.25)


def test_collect_average_progress_preserves_task_with_only_unscored_episodes(tmp_path):
    results_directory = tmp_path / "results"
    _write_progress_results(results_directory / "task_alpha_pi0/episode_results.jsonl", "task_alpha_pi0", [1.0])
    _write_progress_results(results_directory / "task_beta_pi0/episode_results.jsonl", "task_beta_pi0", [None])

    results = collect_average_progress(results_directory, ["pi0"])

    assert results[("task_beta", "pi0")] == AverageProgressResult(total_progress=0.0, scored_episodes=0)
    assert results[("task_beta", "pi0")].average_progress is None


def test_collect_average_progress_rejects_out_of_range_score(tmp_path):
    results_path = tmp_path / "episode_results.jsonl"
    # overall_score is recorded already normalized to [0, 1]; an out-of-range value is an error.
    _write_progress_results(results_path, "task_alpha_pi0", [1.5])

    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        collect_average_progress(results_path, ["pi0"])


def test_plot_average_progress_writes_png_with_missing_policy_pair(tmp_path):
    results = {
        ("task_alpha", "pi0"): AverageProgressResult(total_progress=1.0, scored_episodes=2),
        ("task_alpha", "cosmos"): AverageProgressResult(total_progress=2.0, scored_episodes=2),
        ("task_beta", "pi0"): AverageProgressResult(total_progress=0.0, scored_episodes=0),
    }
    output_path = tmp_path / "average_progress.png"

    plot_average_progress(results, ["pi0", "cosmos"], output_path)

    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_main_writes_both_plots_by_default(tmp_path, capsys):
    results_directory = tmp_path / "results"
    results_directory.mkdir()
    (results_directory / "task_alpha_pi0").mkdir()
    (results_directory / "task_alpha_pi0" / "episode_results.jsonl").write_text(
        json.dumps({"job_name": "task_alpha_pi0", "success": True, "progress": {"overall_score": 0.5}}) + "\n",
        encoding="utf-8",
    )

    return_code = main([str(results_directory), "--policies", "pi0"])

    captured_output = capsys.readouterr()
    assert return_code == 0
    assert (results_directory / "success_rates.png").is_file()
    assert (results_directory / "average_progress.png").is_file()
    assert "[success_rate] Read 1 scored episodes across 1 tasks and 1 policies." in captured_output.out
    assert "[average_progress] Read 1 scored episodes across 1 tasks and 1 policies." in captured_output.out


def test_main_writes_both_plots_into_output_dir(tmp_path):
    results_directory = tmp_path / "results"
    results_directory.mkdir()
    (results_directory / "task_alpha_pi0").mkdir()
    (results_directory / "task_alpha_pi0" / "episode_results.jsonl").write_text(
        json.dumps({"job_name": "task_alpha_pi0", "success": True, "progress": {"overall_score": 0.5}}) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "plots"

    return_code = main([str(results_directory), "--policies", "pi0", "--output-dir", str(output_dir)])

    assert return_code == 0
    assert (output_dir / "success_rates.png").is_file()
    assert (output_dir / "average_progress.png").is_file()

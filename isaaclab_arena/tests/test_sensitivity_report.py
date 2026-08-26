# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Report dispatch and rendering tests for sensitivity analysis."""

from __future__ import annotations

import json
import torch

import isaaclab_arena.analysis.sensitivity.generate_report as report_module


def _write_episode_results(path) -> None:
    """Write a small varied dataset with both successful and failed episodes."""
    rows = [
        {"success": success, "variations": {"offset": offset}}
        for offset, success in zip(range(6), [False, True, True, False, True, False])
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_empirical_report_writes_png_without_fitting(tmp_path, monkeypatch, capsys):
    """Empirical dispatch renders exact matches without constructing a fitted posterior."""
    episode_results_path = tmp_path / "episode_results.jsonl"
    output_path = tmp_path / "empirical.png"
    _write_episode_results(episode_results_path)

    def fail_if_called(_analyzer):
        raise AssertionError("Empirical reports must not fit NPE or MNPE")

    monkeypatch.setattr(report_module.SensitivityAnalyzer, "fit", fail_if_called)

    returned_path = report_module.generate_report(
        episode_results_path,
        output_path,
        method="empirical",
        num_bins=3,
        num_bootstrap_samples=32,
        seed=19,
    )

    assert returned_path == output_path
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert "3 of 6 episodes exactly match" in capsys.readouterr().out


def test_report_defaults_to_existing_fitted_path(tmp_path, monkeypatch):
    """Omitting method preserves the existing fitted report behavior."""
    episode_results_path = tmp_path / "episode_results.jsonl"
    _write_episode_results(episode_results_path)
    calls = {"analyze": None, "analyzer_dataset": None, "plot": None}

    class _FakeSensitivityAnalyzer:
        def __init__(self, dataset):
            self.dataset = dataset
            calls["analyzer_dataset"] = dataset

        def analyze(self, **kwargs):
            calls["analyze"] = kwargs
            return self.dataset.theta

    def record_plot(samples, dataset, observation, output_path):
        calls["plot"] = {
            "samples": samples,
            "dataset": dataset,
            "observation": observation,
            "output_path": output_path,
        }

    monkeypatch.setattr(report_module, "SensitivityAnalyzer", _FakeSensitivityAnalyzer)
    monkeypatch.setattr(report_module, "plot_marginals", record_plot)

    report_module.generate_report(episode_results_path, tmp_path / "fitted.png")

    assert calls["analyze"]["method"] == "fitted"
    torch.testing.assert_close(calls["analyze"]["observation"], torch.tensor([1.0]))
    assert calls["analyze"]["seed"] == 0
    assert calls["plot"]["samples"] is calls["analyzer_dataset"].theta
    assert calls["plot"]["dataset"] is calls["analyzer_dataset"]
    assert calls["plot"]["observation"].tolist() == [1.0]
    assert calls["plot"]["output_path"] == str(tmp_path / "fitted.png")

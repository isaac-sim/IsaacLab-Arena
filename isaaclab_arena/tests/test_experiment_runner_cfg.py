# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify the typed-only Experiment Runner YAML contract."""

from pathlib import Path

import pytest

from isaaclab_arena.evaluation.experiment_runner_cfg import load_experiment_runner_cfg


def test_loads_checked_in_openpi_evaluation_config():
    config_path = (
        Path(__file__).resolve().parents[2] / "isaaclab_arena_environments" / "evaluation_configs" / "openpi.yaml"
    )

    cfg = load_experiment_runner_cfg(config_path)

    assert Path(cfg.experiment_config).name == "openpi_experiment.yaml"
    assert cfg.output_base_dir == "/eval/output"
    assert not cfg.record_camera_video


def test_load_experiment_runner_cfg_resolves_experiment_relative_to_config(tmp_path):
    experiment_path = tmp_path / "experiments" / "example.yaml"
    experiment_path.parent.mkdir()
    experiment_path.write_text("runs: []\n", encoding="utf-8")
    config_path = tmp_path / "configs" / "evaluation.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        """experiment_config: ../experiments/example.yaml
experiment_overrides:
  - runs.baseline.rollout_limit.num_steps=4
record_camera_video: true
continue_on_error: true
evaluation_report_port: 9000
""",
        encoding="utf-8",
    )

    cfg = load_experiment_runner_cfg(config_path)

    assert Path(cfg.experiment_config) == experiment_path.resolve()
    assert cfg.experiment_overrides == ["runs.baseline.rollout_limit.num_steps=4"]
    assert cfg.output_base_dir == "/eval/output"
    assert not cfg.record_viewport_video
    assert cfg.record_camera_video
    assert cfg.continue_on_error
    assert not cfg.serve_evaluation_report
    assert cfg.evaluation_report_port == 9000


def test_load_experiment_runner_cfg_rejects_legacy_json_experiment(tmp_path):
    experiment_path = tmp_path / "legacy.json"
    experiment_path.write_text('{"jobs": []}\n', encoding="utf-8")
    config_path = tmp_path / "evaluation.yaml"
    config_path.write_text("experiment_config: legacy.json\n", encoding="utf-8")

    with pytest.raises(AssertionError, match="must be YAML"):
        load_experiment_runner_cfg(config_path)


def test_load_experiment_runner_cfg_rejects_unknown_fields(tmp_path):
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text("runs: []\n", encoding="utf-8")
    config_path = tmp_path / "evaluation.yaml"
    config_path.write_text(
        "experiment_config: experiment.yaml\nchunk_size: 2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="chunk_size"):
        load_experiment_runner_cfg(config_path)

# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Typed configuration for executing a YAML Arena Experiment."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from isaaclab_arena.hydra.typed_yaml import load_typed_yaml_cfg


@dataclass
class ExperimentRunnerCfg:
    """Configure execution of one typed Arena Experiment."""

    experiment_config: str
    """Typed Experiment YAML path, relative to this configuration when not absolute."""

    experiment_overrides: list[str] = field(default_factory=list)
    """Hydra overrides applied to the typed Experiment after its YAML values."""

    output_base_dir: str = "/eval/output"
    """Base directory receiving the timestamped Experiment output directory."""

    record_viewport_video: bool = False
    """Whether to record the rendered viewport for each Run."""

    record_camera_video: bool = False
    """Whether to record observation-camera videos for each Run."""

    continue_on_error: bool = False
    """Whether to continue with later Runs after a Run fails."""

    serve_evaluation_report: bool = False
    """Whether to serve the generated evaluation report after execution."""

    evaluation_report_port: int = 8000
    """Port used when serving the generated evaluation report."""

    def __post_init__(self) -> None:
        assert self.experiment_config, "experiment_config must not be empty"
        assert self.output_base_dir, "output_base_dir must not be empty"
        assert 0 < self.evaluation_report_port < 65536, "evaluation_report_port must be between 1 and 65535"


def load_experiment_runner_cfg(config_path: str | Path) -> ExperimentRunnerCfg:
    """Load an Experiment Runner configuration and resolve its Experiment path."""
    path = Path(config_path).expanduser().resolve()
    cfg = load_typed_yaml_cfg(path, ExperimentRunnerCfg, config_name="Experiment Runner")

    experiment_config_path = Path(cfg.experiment_config).expanduser()
    if not experiment_config_path.is_absolute():
        experiment_config_path = path.parent / experiment_config_path
    experiment_config_path = experiment_config_path.resolve()

    assert experiment_config_path.is_file(), f"Experiment config does not exist: '{experiment_config_path}'"
    assert experiment_config_path.suffix.lower() in {
        ".yaml",
        ".yml",
    }, f"Typed Experiment config must be YAML, got '{experiment_config_path}'"
    return replace(cfg, experiment_config=str(experiment_config_path))

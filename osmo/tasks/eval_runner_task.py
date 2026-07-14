# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""OSMO task that executes a complete Arena Experiment through ``eval_runner.py``."""

from __future__ import annotations

import shlex
import yaml
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from isaaclab_arena.assets.registries import EnvironmentRegistry, PolicyRegistry
from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentDefinitionCfg
from isaaclab_arena.evaluation.arena_run import ArenaRunCfg
from osmo.tasks.base_task import BaseTask, TaskCfg
from osmo.workflows.utils.yaml_utils import block_literal_str
from osmo.workflows.workflow_constants import DATASET_SWIFT_URL, OSMO_TASK_OUTPUT_DIR

EVAL_RUNNER_SCRIPT = "isaaclab_arena/evaluation/eval_runner.py"
DEFAULT_EVAL_RUNNER_IMAGE = "nvcr.io/nvstaging/isaac-amr/isaaclab_arena:latest"
REMOTE_EXPERIMENT_PATH = "/tmp/arena_experiment.yaml"


@dataclass
class EvalRunnerTaskCfg(TaskCfg):
    """Configuration for an OSMO eval-runner task."""

    image: str = DEFAULT_EVAL_RUNNER_IMAGE
    """Container image that runs the Arena Experiment."""


class EvalRunnerTask(BaseTask):
    """Lead OSMO task that runs every Run in one effective Arena Experiment."""

    def __init__(
        self,
        task_cfg: EvalRunnerTaskCfg,
        experiment_definition: ArenaExperimentDefinitionCfg,
        lead: bool | None = None,
    ) -> None:
        super().__init__(task_cfg=task_cfg, lead=lead)
        assert isinstance(experiment_definition, ArenaExperimentDefinitionCfg)
        self.experiment_definition = deepcopy(experiment_definition)

    @staticmethod
    def get_task_name() -> str:
        return "eval_runner"

    def _get_image(self) -> str:
        return self.task_cfg.image

    def _get_inputs(self) -> list[dict[str, Any]]:
        return []

    def _get_outputs(self) -> list[dict[str, Any]]:
        return [{"url": DATASET_SWIFT_URL}]

    def _get_files(self) -> list[dict[str, Any]]:
        """Embed the effective Experiment at the path consumed by ``eval_runner.py``."""
        experiment_yaml = yaml.safe_dump(self._get_experiment_definition_yaml_values(), sort_keys=False)
        return [
            *super()._get_files(),
            {"path": REMOTE_EXPERIMENT_PATH, "contents": block_literal_str(experiment_yaml)},
        ]

    def _get_experiment_definition_yaml_values(self) -> dict[str, Any]:
        """Restore YAML selectors around the effective typed Run configs."""
        environment_registry = EnvironmentRegistry()
        policy_registry = PolicyRegistry()
        run_values_by_name = {}
        for run_name, run_cfg in self.experiment_definition.runs.items():
            assert isinstance(run_cfg, ArenaRunCfg)
            run_values = OmegaConf.to_container(
                OmegaConf.structured(run_cfg),
                resolve=True,
                enum_to_str=True,
            )
            assert isinstance(run_values, dict)
            assert run_values.pop("name") == run_name

            environment_type = environment_registry.get_factory_type_for_cfg(run_cfg.environment)
            policy_type = policy_registry.get_policy_type_for_cfg(run_cfg.policy)
            run_values["environment"] = {"type": environment_type.name, **run_values["environment"]}
            policy_selector = policy_type.name
            if not policy_type.__module__.startswith("isaaclab_arena.policy."):
                policy_selector = f"{policy_type.__module__}.{policy_type.__qualname__}"
            run_values["policy"] = {"type": policy_selector, **run_values["policy"]}
            run_values_by_name[run_name] = _to_yaml_values(run_values)
        return {"runs": run_values_by_name}

    def _get_run_script(self) -> str:
        command = [
            "/isaac-sim/python.sh",
            EVAL_RUNNER_SCRIPT,
            "--experiment_config",
            REMOTE_EXPERIMENT_PATH,
            "--output_base_dir",
            OSMO_TASK_OUTPUT_DIR,
            "--viz",
            "none",
            "--enable_cameras",
        ]
        return f"set -euo pipefail\n{shlex.join(command)}\n"


def _to_yaml_values(value: Any) -> Any:
    """Convert structured-config leaf values into safe YAML primitives."""
    if isinstance(value, dict):
        return {key: _to_yaml_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_yaml_values(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    return value

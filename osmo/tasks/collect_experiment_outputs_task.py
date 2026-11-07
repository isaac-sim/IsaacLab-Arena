# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""OSMO task that collects Experiment Runner outputs into one published Arena Experiment output."""

from __future__ import annotations

import json
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentCfg
from isaaclab_arena.hydra.typed_experiment_serializer import serialize_arena_experiment_to_yaml
from osmo.scripts.build_experiment_output import ARENA_EXPERIMENT_DEFINITION_FILENAME
from osmo.tasks.base_task import BaseTask
from osmo.workflows.utils.yaml_utils import block_literal_str
from osmo.workflows.workflow_constants import DATASET_SWIFT_URL, OSMO_TASK_OUTPUT_DIR

_LOCAL_BUILD_EXPERIMENT_OUTPUT_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "build_experiment_output.py"
_REMOTE_BUILD_EXPERIMENT_OUTPUT_SCRIPT_PATH = "/tmp/arena_build_experiment_output.py"
_REMOTE_EXPERIMENT_RUNNER_OUTPUT_DIRECTORIES_FILE_PATH = "/tmp/arena_experiment_runner_output_directories.json"
_REMOTE_EXPERIMENT_DEFINITION_FILE_PATH = "/tmp/arena_experiment_definition.yaml"


def experiment_runner_output_directory_input_token(experiment_runner_task_name: str) -> str:
    """Return the OSMO input token that resolves to an Experiment Runner task's output directory."""
    return "{{input:" + experiment_runner_task_name + "}}"


class CollectExperimentOutputsTask(BaseTask):
    """Collect and publish one Experiment output from the Experiment Runner task outputs.

    Copies each completed Run's output, the Experiment result, timings and report, and the Experiment Definition
    (with its overrides applied) into one output directory. Failed Runs are listed without their partial artifacts.
    Only this copied output is published to Swift.
    """

    def __init__(
        self,
        image: str,
        experiment_runner_task_names_by_run_name: Mapping[str, str],
        experiment_cfg: ArenaExperimentCfg,
        lead: bool | None = None,
        resource: str | None = None,
        output_url: str = DATASET_SWIFT_URL,
        *,
        task_name: str,
    ) -> None:
        assert experiment_runner_task_names_by_run_name, "Experiment output requires at least one Run task"
        super().__init__(task_name=task_name, lead=lead, resource=resource)
        self.experiment_cfg = experiment_cfg
        self.image = image
        self.experiment_runner_task_names_by_run_name = dict(experiment_runner_task_names_by_run_name)
        self.output_url = output_url

    def _get_image(self) -> str:
        return self.image

    def _get_inputs(self) -> list[dict[str, Any]]:
        """Make every Experiment Runner task's workflow-local output available to this task."""
        return [
            {"task": experiment_runner_task_name}
            for experiment_runner_task_name in self.experiment_runner_task_names_by_run_name.values()
        ]

    def _get_outputs(self) -> list[dict[str, Any]]:
        """Publish the final Experiment directory, including its canonical result and report."""
        return [{"url": self.output_url}]

    def _get_files_to_create(self) -> list[dict[str, Any]]:
        """Embed the output builder, its JSON input and the Experiment Definition."""
        experiment_runner_output_directory_tokens_by_run_name = {
            run_name: experiment_runner_output_directory_input_token(experiment_runner_task_name)
            for run_name, experiment_runner_task_name in self.experiment_runner_task_names_by_run_name.items()
        }
        return [
            *super()._get_files_to_create(),
            {
                "path": _REMOTE_BUILD_EXPERIMENT_OUTPUT_SCRIPT_PATH,
                "contents": block_literal_str(_LOCAL_BUILD_EXPERIMENT_OUTPUT_SCRIPT_PATH.read_text(encoding="utf-8")),
            },
            {
                "path": _REMOTE_EXPERIMENT_RUNNER_OUTPUT_DIRECTORIES_FILE_PATH,
                "contents": block_literal_str(
                    json.dumps(experiment_runner_output_directory_tokens_by_run_name, indent=2)
                ),
            },
            {
                "path": _REMOTE_EXPERIMENT_DEFINITION_FILE_PATH,
                "contents": block_literal_str(serialize_arena_experiment_to_yaml(self.experiment_cfg)),
            },
        ]

    def _get_run_script(self) -> str:
        build_experiment_output_command = shlex.join([
            "python",
            _REMOTE_BUILD_EXPERIMENT_OUTPUT_SCRIPT_PATH,
            "--experiment-runner-output-directories-file",
            _REMOTE_EXPERIMENT_RUNNER_OUTPUT_DIRECTORIES_FILE_PATH,
            "--experiment-output-directory",
            OSMO_TASK_OUTPUT_DIR,
        ])
        # The builder reads the Experiment Definition back from the Experiment output, which then publishes it.
        copy_experiment_definition_command = shlex.join([
            "cp",
            _REMOTE_EXPERIMENT_DEFINITION_FILE_PATH,
            f"{OSMO_TASK_OUTPUT_DIR}/{ARENA_EXPERIMENT_DEFINITION_FILENAME}",
        ])
        return (
            f"set -euo pipefail\nmkdir -p {shlex.quote(OSMO_TASK_OUTPUT_DIR)}\n"
            f"{copy_experiment_definition_command}\n{build_experiment_output_command}\n"
        )

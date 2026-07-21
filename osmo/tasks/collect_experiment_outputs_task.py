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

from osmo.tasks.base_task import BaseTask
from osmo.workflows.utils.yaml_utils import block_literal_str
from osmo.workflows.workflow_constants import DATASET_SWIFT_URL, OSMO_TASK_OUTPUT_DIR

_LOCAL_BUILD_EXPERIMENT_OUTPUT_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "build_experiment_output.py"
_REMOTE_BUILD_EXPERIMENT_OUTPUT_SCRIPT_PATH = "/tmp/arena_build_experiment_output.py"
_REMOTE_EXPERIMENT_RUNNER_OUTPUT_DIRECTORIES_FILE_PATH = "/tmp/arena_experiment_runner_output_directories.json"


def experiment_runner_output_directory_input_token(experiment_runner_task_name: str) -> str:
    """Return the OSMO input token that resolves to an Experiment Runner task's output directory."""
    return "{{input:" + experiment_runner_task_name + "}}"


class CollectExperimentOutputsTask(BaseTask):
    """Collect and publish one Experiment output from the Experiment Runner task outputs.

    For each Run, OSMO exposes its Experiment Runner task output at ``{{input:<runner-task>}}``. The embedded script
    copies ``{{input:<runner-task>}}/<run-name>/...`` to ``{{output}}/<run-name>/...`` and writes
    ``{{output}}/index.html``. Only this final task output is published to Swift; the Experiment Runner task outputs
    remain workflow-local.
    """

    def __init__(
        self,
        image: str,
        experiment_runner_task_names_by_run_name: Mapping[str, str],
        lead: bool | None = None,
        resource: str | None = None,
        *,
        task_name: str,
    ) -> None:
        assert experiment_runner_task_names_by_run_name, "Experiment output requires at least one Run task"
        super().__init__(task_name=task_name, lead=lead, resource=resource)
        self.image = image
        self.experiment_runner_task_names_by_run_name = dict(experiment_runner_task_names_by_run_name)

    def _get_image(self) -> str:
        return self.image

    def _get_inputs(self) -> list[dict[str, Any]]:
        """Make every Experiment Runner task's workflow-local output available to this task."""
        return [
            {"task": experiment_runner_task_name}
            for experiment_runner_task_name in self.experiment_runner_task_names_by_run_name.values()
        ]

    def _get_outputs(self) -> list[dict[str, Any]]:
        """Publish the final Experiment directory, including all Runs and ``index.html``."""
        return [{"url": DATASET_SWIFT_URL}]

    def _get_files_to_create(self) -> list[dict[str, Any]]:
        """Embed the output-building script and its ``run-name -> Experiment Runner output`` JSON input."""
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
        ]

    def _get_run_script(self) -> str:
        build_experiment_output_command = shlex.join([
            "/isaac-sim/python.sh",
            _REMOTE_BUILD_EXPERIMENT_OUTPUT_SCRIPT_PATH,
            "--experiment-runner-output-directories-file",
            _REMOTE_EXPERIMENT_RUNNER_OUTPUT_DIRECTORIES_FILE_PATH,
            "--experiment-output-directory",
            OSMO_TASK_OUTPUT_DIR,
        ])
        return f"set -euo pipefail\n{build_experiment_output_command}\n"

# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""OSMO task that stages per-Run task outputs and publishes one Arena Experiment output."""

from __future__ import annotations

import json
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from osmo.tasks.base_task import BaseTask
from osmo.workflows.utils.yaml_utils import block_literal_str
from osmo.workflows.workflow_constants import DATASET_SWIFT_URL, OSMO_TASK_OUTPUT_DIR

_LOCAL_AGGREGATE_EXPERIMENT_RESULTS_SCRIPT_PATH = (
    Path(__file__).parents[1] / "scripts" / "aggregate_experiment_results.py"
)
REMOTE_AGGREGATE_EXPERIMENT_RESULTS_SCRIPT_PATH = "/tmp/arena_aggregate_experiment_results.py"
REMOTE_STAGED_EXPERIMENT_RUNNER_OUTPUT_DIRECTORIES_FILE_PATH = (
    "/tmp/arena_staged_experiment_runner_output_directories.json"
)


def staged_task_output_directory_token(upstream_task_name: str) -> str:
    """Return the OSMO input token resolving to a staged upstream task output directory."""
    return "{{input:" + upstream_task_name + "}}"


class ExperimentResultsTask(BaseTask):
    """Publish one Experiment output from workflow-local outputs produced per Run.

    For each Run, OSMO stages its exact single-Run Experiment directory at ``{{input:<runner-task>}}``. The
    embedded script copies ``{{input:<runner-task>}}/<run-name>/...`` to ``{{output}}/<run-name>/...`` and writes
    ``{{output}}/index.html``. Only this final task output is published to Swift; the upstream runner outputs remain
    workflow-local.
    """

    def __init__(
        self,
        image: str,
        experiment_runner_task_names_by_run_name: Mapping[str, str],
        lead: bool | None = None,
        resource: str | None = None,
    ) -> None:
        assert experiment_runner_task_names_by_run_name, "Experiment output requires at least one Run task"
        super().__init__(lead=lead, resource=resource)
        self.image = image
        self.experiment_runner_task_names_by_run_name = dict(experiment_runner_task_names_by_run_name)

    @staticmethod
    def get_task_name() -> str:
        return "aggregate-results"

    def _get_image(self) -> str:
        return self.image

    def _get_inputs(self) -> list[dict[str, Any]]:
        """Stage the workflow-local output directory from every Run's Experiment Runner task."""
        return [
            {"task": experiment_runner_task_name}
            for experiment_runner_task_name in self.experiment_runner_task_names_by_run_name.values()
        ]

    def _get_outputs(self) -> list[dict[str, Any]]:
        """Publish the final Experiment directory, including all Runs and ``index.html``."""
        return [{"url": DATASET_SWIFT_URL}]

    def _get_files_to_create(self) -> list[dict[str, Any]]:
        """Embed the builder and its ``run-name -> staged task output root`` JSON input."""
        staged_experiment_runner_output_directory_tokens_by_run_name = {
            run_name: staged_task_output_directory_token(experiment_runner_task_name)
            for run_name, experiment_runner_task_name in self.experiment_runner_task_names_by_run_name.items()
        }
        return [
            *super()._get_files_to_create(),
            {
                "path": REMOTE_AGGREGATE_EXPERIMENT_RESULTS_SCRIPT_PATH,
                "contents": block_literal_str(
                    _LOCAL_AGGREGATE_EXPERIMENT_RESULTS_SCRIPT_PATH.read_text(encoding="utf-8")
                ),
            },
            {
                "path": REMOTE_STAGED_EXPERIMENT_RUNNER_OUTPUT_DIRECTORIES_FILE_PATH,
                "contents": block_literal_str(
                    json.dumps(staged_experiment_runner_output_directory_tokens_by_run_name, indent=2)
                ),
            },
        ]

    def _get_run_script(self) -> str:
        build_experiment_output_command = shlex.join([
            "/isaac-sim/python.sh",
            REMOTE_AGGREGATE_EXPERIMENT_RESULTS_SCRIPT_PATH,
            "--staged-experiment-runner-output-directories-file",
            REMOTE_STAGED_EXPERIMENT_RUNNER_OUTPUT_DIRECTORIES_FILE_PATH,
            "--combined-experiment-output-directory",
            OSMO_TASK_OUTPUT_DIR,
        ])
        return f"set -euo pipefail\n{build_experiment_output_command}\n"

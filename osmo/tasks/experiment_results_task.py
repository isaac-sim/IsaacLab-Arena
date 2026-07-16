# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""OSMO task that combines independently executed Arena Run outputs."""

from __future__ import annotations

import json
import shlex
from collections.abc import Mapping
from typing import Any

from osmo.tasks.base_task import BaseTask
from osmo.workflows.utils.yaml_utils import block_literal_str
from osmo.workflows.workflow_constants import DATASET_SWIFT_URL, OSMO_TASK_OUTPUT_DIR

AGGREGATE_EXPERIMENT_OUTPUTS_SCRIPT = "isaaclab_arena/evaluation/aggregate_experiment_outputs.py"
REMOTE_RUN_INPUTS_PATH = "/tmp/arena_run_inputs.json"


def task_input_token(task_name: str) -> str:
    """Return the OSMO token for a task's staged output directory."""
    return "{{input:" + task_name + "}}"


class ExperimentResultsTask(BaseTask):
    """Lead CPU task that builds the combined Arena Experiment output."""

    def __init__(
        self,
        image: str,
        run_task_names: Mapping[str, str],
        lead: bool | None = None,
        resource: str | None = None,
    ) -> None:
        assert run_task_names, "Experiment result aggregation requires at least one Run task"
        super().__init__(lead=lead, resource=resource)
        self.image = image
        self.run_task_names = dict(run_task_names)

    @staticmethod
    def get_task_name() -> str:
        return "aggregate-results"

    def _get_image(self) -> str:
        return self.image

    def _get_inputs(self) -> list[dict[str, Any]]:
        return [{"task": task_name} for task_name in self.run_task_names.values()]

    def _get_outputs(self) -> list[dict[str, Any]]:
        return [{"url": DATASET_SWIFT_URL}]

    def _get_files_to_create(self) -> list[dict[str, Any]]:
        run_input_dirs = {run_name: task_input_token(task_name) for run_name, task_name in self.run_task_names.items()}
        return [
            *super()._get_files_to_create(),
            {
                "path": REMOTE_RUN_INPUTS_PATH,
                "contents": block_literal_str(json.dumps(run_input_dirs, indent=2)),
            },
        ]

    def _get_run_script(self) -> str:
        command = shlex.join([
            "/isaac-sim/python.sh",
            AGGREGATE_EXPERIMENT_OUTPUTS_SCRIPT,
            "--run-inputs",
            REMOTE_RUN_INPUTS_PATH,
            "--output-dir",
            OSMO_TASK_OUTPUT_DIR,
        ])
        return f"set -euo pipefail\n{command}\n"

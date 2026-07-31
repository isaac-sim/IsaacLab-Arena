# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""OSMO task that coordinates one report from all successful Experiment Runs."""

from __future__ import annotations

import json
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from osmo.tasks.base_task import BaseTask
from osmo.workflows.utils.yaml_utils import block_literal_str
from osmo.workflows.workflow_constants import OSMO_TASK_OUTPUT_DIR

_LOCAL_COORDINATE_EXPERIMENT_OUTPUT_SCRIPT_PATH = (
    Path(__file__).parents[1] / "scripts" / "coordinate_experiment_output.py"
)
_LOCAL_BUILD_EXPERIMENT_OUTPUT_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "build_experiment_output.py"
_REMOTE_COORDINATE_EXPERIMENT_OUTPUT_SCRIPT_PATH = "/tmp/arena_coordinate_experiment_output.py"
_REMOTE_BUILD_EXPERIMENT_OUTPUT_SCRIPT_PATH = "/tmp/arena_build_experiment_output.py"
_REMOTE_COORDINATOR_SETTINGS_FILE_PATH = "/tmp/arena_experiment_output_coordinator_settings.json"


class ExperimentOutputCoordinatorTask(BaseTask):
    """Wait for every Run and build one report from the successful outputs."""

    def __init__(
        self,
        image: str,
        source_workflow_id: str,
        experiment_runner_task_names_by_run_name: Mapping[str, str],
        published_output_url: str,
        poll_interval_seconds: int = 30,
        lead: bool | None = None,
        resource: str | None = None,
        *,
        task_name: str,
    ) -> None:
        assert source_workflow_id, "Experiment output coordinator requires a source workflow ID"
        assert experiment_runner_task_names_by_run_name, "Experiment output coordinator requires at least one Run task"
        assert published_output_url, "Experiment output coordinator requires a published output URL"
        assert poll_interval_seconds > 0, "Experiment output coordinator poll interval must be positive"
        super().__init__(task_name=task_name, lead=lead, resource=resource)
        self.image = image
        self.source_workflow_id = source_workflow_id
        self.experiment_runner_task_names_by_run_name = dict(experiment_runner_task_names_by_run_name)
        self.published_output_url = published_output_url
        self.poll_interval_seconds = poll_interval_seconds

    def _get_image(self) -> str:
        return self.image

    def _get_inputs(self) -> list[dict[str, Any]]:
        """Start independently so a failed Run cannot cause ``FAILED_UPSTREAM``."""
        return []

    def _get_outputs(self) -> list[dict[str, Any]]:
        """Publish the report under the source Experiment workflow ID."""
        return [{"url": self.published_output_url}]

    def _get_files_to_create(self) -> list[dict[str, Any]]:
        """Embed the self-contained coordinator, collector builder, and settings."""
        settings = {
            "experiment_runner_task_names_by_run_name": self.experiment_runner_task_names_by_run_name,
            "poll_interval_seconds": self.poll_interval_seconds,
        }
        return [
            *super()._get_files_to_create(),
            {
                "path": _REMOTE_COORDINATE_EXPERIMENT_OUTPUT_SCRIPT_PATH,
                "contents": block_literal_str(
                    _LOCAL_COORDINATE_EXPERIMENT_OUTPUT_SCRIPT_PATH.read_text(encoding="utf-8")
                ),
            },
            {
                "path": _REMOTE_BUILD_EXPERIMENT_OUTPUT_SCRIPT_PATH,
                "contents": block_literal_str(_LOCAL_BUILD_EXPERIMENT_OUTPUT_SCRIPT_PATH.read_text(encoding="utf-8")),
            },
            {
                "path": _REMOTE_COORDINATOR_SETTINGS_FILE_PATH,
                "contents": block_literal_str(json.dumps(settings, indent=2)),
            },
        ]

    def _get_run_script(self) -> str:
        coordinate_experiment_output_command = shlex.join([
            "/isaac-sim/python.sh",
            _REMOTE_COORDINATE_EXPERIMENT_OUTPUT_SCRIPT_PATH,
            "--source-workflow-id",
            self.source_workflow_id,
            "--settings-file",
            _REMOTE_COORDINATOR_SETTINGS_FILE_PATH,
            "--build-experiment-output-script",
            _REMOTE_BUILD_EXPERIMENT_OUTPUT_SCRIPT_PATH,
            "--experiment-output-directory",
            OSMO_TASK_OUTPUT_DIR,
        ])
        return f"set -euo pipefail\n{coordinate_experiment_output_command}\n"

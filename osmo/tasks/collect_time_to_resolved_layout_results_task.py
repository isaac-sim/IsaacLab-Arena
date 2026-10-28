# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""OSMO task that combines time-to-resolved-layout benchmark task outputs into one published result."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

from osmo.tasks.base_task import BaseTask
from osmo.workflows.utils.yaml_utils import block_literal_str
from osmo.workflows.workflow_constants import DATASET_SWIFT_URL, OSMO_TASK_OUTPUT_DIR

_LOCAL_COMBINE_RESULTS_SCRIPT_PATH = (
    Path(__file__).parents[1] / "scripts" / "combine_time_to_resolved_layout_results.py"
)
_REMOTE_COMBINE_RESULTS_SCRIPT_PATH = "/tmp/arena_combine_time_to_resolved_layout_results.py"
_REMOTE_BENCHMARK_FILE_PATH = "/tmp/arena_time_to_resolved_layout_benchmark.json"


class CollectTimeToResolvedLayoutResultsTask(BaseTask):
    """Collect every benchmark task output and publish them with a combined ``all_results.json``.

    For each configuration, OSMO exposes its benchmark task output at ``{{input:<benchmark-task>}}``. The embedded
    script copies those files to ``{{output}}`` and writes ``{{output}}/all_results.json`` in the batch runner's format.
    """

    def __init__(
        self,
        image: str,
        benchmark: dict[str, Any],
        benchmark_task_names_by_configuration_name: dict[str, str],
        lead: bool | None = None,
        resource: str | None = None,
        output_url: str = DATASET_SWIFT_URL,
        *,
        task_name: str,
    ) -> None:
        """Create the task.

        Args:
            image: Container image with Arena installed.
            benchmark: Benchmark settings and configurations, each with ``name``, ``case``, and ``num_envs``.
            benchmark_task_names_by_configuration_name: OSMO benchmark task that measured each configuration.
            lead: Whether this is the lead task of its OSMO group.
            resource: Named OSMO resource for the task.
            output_url: URL the combined result is published to.
            task_name: OSMO task name.
        """
        assert benchmark_task_names_by_configuration_name, "Combined result requires at least one benchmark task"
        super().__init__(task_name=task_name, lead=lead, resource=resource)
        self.image = image
        self.benchmark = benchmark
        self.benchmark_task_names_by_configuration_name = dict(benchmark_task_names_by_configuration_name)
        self.output_url = output_url

    def _get_image(self) -> str:
        return self.image

    def _get_inputs(self) -> list[dict[str, Any]]:
        """Make every benchmark task's workflow-local output available to this task."""
        return [{"task": task_name} for task_name in self.benchmark_task_names_by_configuration_name.values()]

    def _get_outputs(self) -> list[dict[str, Any]]:
        return [{"url": self.output_url}]

    def _get_files_to_create(self) -> list[dict[str, Any]]:
        """Embed the combine script and the benchmark settings with each task's output directory."""
        configurations = []
        for configuration in self.benchmark["configurations"]:
            task_name = self.benchmark_task_names_by_configuration_name[configuration["name"]]
            configurations.append({
                **configuration,
                # OSMO resolves this token to the benchmark task's output directory.
                "task_output_directory": "{{input:" + task_name + "}}",
            })
        return [
            *super()._get_files_to_create(),
            {
                "path": _REMOTE_COMBINE_RESULTS_SCRIPT_PATH,
                "contents": block_literal_str(_LOCAL_COMBINE_RESULTS_SCRIPT_PATH.read_text(encoding="utf-8")),
            },
            {
                "path": _REMOTE_BENCHMARK_FILE_PATH,
                "contents": block_literal_str(
                    json.dumps({**self.benchmark, "configurations": configurations}, indent=2)
                ),
            },
        ]

    def _get_run_script(self) -> str:
        combine_results_command = shlex.join([
            "python",
            _REMOTE_COMBINE_RESULTS_SCRIPT_PATH,
            "--benchmark-file",
            _REMOTE_BENCHMARK_FILE_PATH,
            "--output-directory",
            OSMO_TASK_OUTPUT_DIR,
        ])
        return f"set -euo pipefail\n{combine_results_command}\n"

# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Workflow class for Isaac Lab Arena OSMO workflows.

Modeled after ``mindmap_osmo.workflow_utils.workflow.Workflow``. Wraps task
classes and task arguments into an OSMO workflow dict using Arena's
``version: 2`` schema.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
import yaml
from pathlib import Path
from typing import Any

from tasks.base_task import BaseTask
from workflows.utils.workflow_types import WorkflowType
from workflows.utils.yaml_utils import block_literal_str  # noqa: F401  (registers representer)


class Workflow:
    """Builds, renders, and submits an Arena OSMO workflow."""

    def __init__(
        self,
        workflow_type: WorkflowType,
        workflow_args: argparse.Namespace,
        task_cls_list: list[type[BaseTask]],
        task_args_list: list[argparse.Namespace],
        group_name: str = "arena",
    ) -> None:
        assert len(task_cls_list) > 0, "Workflow requires at least one task"
        assert len(task_cls_list) == len(task_args_list), "Each task requires one task args object"
        self.workflow_type = workflow_type
        self.workflow_args = workflow_args
        self.task_cls_list = task_cls_list
        self.task_args_list = task_args_list
        self.group_name = group_name

    def generate_workflow(self) -> dict[str, Any]:
        """Create and return the workflow dictionary."""
        return self.create_workflow_dict()

    def create_workflow_dict(self) -> dict[str, Any]:
        """Build the full OSMO workflow dict."""
        return {
            "version": 2,
            "workflow": {
                "name": self.workflow_args.workflow_name,
                "groups": [{
                    "name": self.group_name,
                    "tasks": [task.create_task_dict() for task in self._get_tasks()],
                }],
                "resources": {"default": self._create_resource_dict()},
                "timeout": {
                    "exec_timeout": self.workflow_args.exec_timeout,
                    "queue_timeout": self.workflow_args.queue_timeout,
                },
            },
        }

    def render_yaml(self) -> str:
        """Render the workflow dict to YAML text."""
        return yaml.dump(
            self.generate_workflow(),
            default_flow_style=False,
            sort_keys=False,
            default_style="",
        )

    def submit_workflow(
        self,
        dry_run: bool = False,
        pool: str | None = None,
        priority: str | None = None,
    ) -> int:
        """Render the workflow and either print it or submit it to OSMO."""
        rendered = self.render_yaml()
        if dry_run:
            print("[dry-run] Rendered workflow YAML:\n")
            print(rendered)
            return 0

        return self._submit_rendered_workflow(rendered=rendered, pool=pool, priority=priority)

    def _get_tasks(self) -> list[BaseTask]:
        """Instantiate task objects for this workflow."""
        tasks = []
        for task_cls, task_args in zip(self.task_cls_list, self.task_args_list):
            assert issubclass(task_cls, BaseTask)
            tasks.append(task_cls(self.workflow_type, self.workflow_args, task_args))
        return tasks

    def _submit_rendered_workflow(
        self,
        rendered: str,
        pool: str | None = None,
        priority: str | None = None,
    ) -> int:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", prefix="arena_", delete=False) as f:
            f.write(rendered)
            rendered_path = f.name

        cmd = ["osmo", "workflow", "submit", rendered_path]
        if pool:
            cmd.extend(["--pool", pool])
        if priority:
            cmd.extend(["--priority", priority])

        print(f"Submitting workflow '{self.workflow_args.workflow_name}':")
        print(f"  {' '.join(cmd)}\n")

        try:
            result = subprocess.run(cmd)
            return result.returncode
        finally:
            Path(rendered_path).unlink(missing_ok=True)

    def _create_resource_dict(self) -> dict[str, Any]:
        return {
            "cpu": self.workflow_args.cpus,
            "gpu": self.workflow_args.gpus,
            "memory": self.workflow_args.memory,
            "platform": self.workflow_args.platform,
            "storage": self.workflow_args.storage,
        }

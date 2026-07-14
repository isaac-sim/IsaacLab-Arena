# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Workflow class for Isaac Lab Arena OSMO workflows.

Workflows and their tasks declare parameters as config dataclasses (WorkflowCfg plus a per-workflow
task config). Only the top-level submit script turns those configs into CLI flags; in-program
callers construct the config objects directly.
"""

from __future__ import annotations

import subprocess
import tempfile
import yaml
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from osmo.tasks.base_task import BaseTask, TaskCfg
from osmo.workflows.utils.yaml_utils import block_literal_str  # noqa: F401  (registers representer)


class WorkflowPriority(str, Enum):
    """OSMO scheduling priority for a workflow."""

    HIGH = "HIGH"
    NORMAL = "NORMAL"
    LOW = "LOW"


@dataclass
class WorkflowCfg:
    """Workflow-level configuration shared by every task in the workflow."""

    workflow_name: str = "arena-workflow"
    """OSMO workflow name."""

    pool: str = "isaac-dev-l40s-04"
    """Target OSMO compute pool."""

    priority: WorkflowPriority = WorkflowPriority.NORMAL
    """OSMO scheduling priority."""

    cpus: int = 15
    """Requested CPU cores."""

    gpus: int = 1
    """Requested GPUs."""

    memory: str = "128Gi"
    """Requested memory."""

    storage: str = "200Gi"
    """Requested storage."""

    platform: str = "ovx-l40s"
    """Target hardware platform."""

    exec_timeout: str = "1d"
    """Maximum execution time before the workflow is killed."""

    queue_timeout: str = "2d"
    """Maximum time the workflow may wait in the queue."""

    dry_run: bool = False
    """Render the workflow YAML and print it instead of submitting to OSMO."""


class Workflow:
    """Builds, renders, and submits an Arena OSMO workflow."""

    task_cls_list: list[type[BaseTask]] = []
    """Task classes that make up this workflow, in group order. Subclasses must set this."""

    task_cfg_type: type[TaskCfg] = TaskCfg
    """Config dataclass type for this workflow's lead task. Subclasses set this."""

    lead_list: list[bool] | None = None
    """Per-task lead flags; ``None`` lets a single-task workflow default its task to lead."""

    def __init__(
        self,
        workflow_cfg: WorkflowCfg,
        task_cfg: TaskCfg,
        group_name: str = "arena",
    ) -> None:
        assert len(self.task_cls_list) > 0, "Workflow subclasses must set task_cls_list"
        self.workflow_cfg = workflow_cfg
        self.task_cfg = task_cfg
        # Single-task workflows may leave lead_list unset; that task defaults to lead.
        self.lead_flags = self.lead_list if self.lead_list is not None else [True]
        self._assert_single_lead_task(self.lead_flags)
        self.group_name = group_name

    def _assert_single_lead_task(self, lead_flags: list[bool]) -> None:
        """Assert the lead flags cover every task with exactly one designated lead."""
        assert len(lead_flags) == len(self.task_cls_list), "Each task requires one lead flag"
        assert sum(lead_flags) == 1, "Exactly one task must be designated as lead (lead=True)"

    def generate_workflow(self) -> dict[str, Any]:
        """Create and return the workflow dictionary."""
        return self.create_workflow_dict()

    def create_workflow_dict(self) -> dict[str, Any]:
        """Build the full OSMO workflow dict."""
        return {
            "version": 2,
            "workflow": {
                "name": self.workflow_cfg.workflow_name,
                "groups": [{
                    "name": self.group_name,
                    "tasks": [task.create_task_dict() for task in self._get_tasks()],
                }],
                "resources": {"default": self._create_resource_dict()},
                "timeout": {
                    "exec_timeout": self.workflow_cfg.exec_timeout,
                    "queue_timeout": self.workflow_cfg.queue_timeout,
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

    def submit_workflow(self) -> int:
        """Render the workflow and either print it (dry-run) or submit it to OSMO."""
        rendered = self.render_yaml()
        if self.workflow_cfg.dry_run:
            print("[dry-run] Rendered workflow YAML:\n")
            print(rendered)
            return 0

        return self._submit_rendered_workflow(rendered=rendered)

    def _get_tasks(self) -> list[BaseTask]:
        """Instantiate task objects for this workflow."""
        tasks = []
        for task_cls, lead in zip(self.task_cls_list, self.lead_flags):
            assert issubclass(task_cls, BaseTask)
            tasks.append(task_cls(self.task_cfg, lead=lead))
        return tasks

    def _submit_rendered_workflow(self, rendered: str) -> int:
        with tempfile.TemporaryDirectory(prefix="arena_") as staging_dir_str:
            staging_dir = Path(staging_dir_str)
            rendered_path = staging_dir / "workflow.yaml"
            rendered_path.write_text(rendered, encoding="utf-8")
            self._stage_submission_files(staging_dir)

            cmd = ["osmo", "workflow", "submit", rendered_path.name]
            if self.workflow_cfg.pool:
                cmd.extend(["--pool", self.workflow_cfg.pool])
            if self.workflow_cfg.priority:
                cmd.extend(["--priority", self.workflow_cfg.priority.value])

            print(f"Submitting workflow '{self.workflow_cfg.workflow_name}':")
            print(f"  {' '.join(cmd)}\n")

            result = subprocess.run(cmd, cwd=staging_dir)
            return result.returncode

    def _stage_submission_files(self, staging_dir: Path) -> None:
        """Stage local files referenced by the workflow alongside its rendered YAML."""

    def _create_resource_dict(self) -> dict[str, Any]:
        return {
            "cpu": self.workflow_cfg.cpus,
            "gpu": self.workflow_cfg.gpus,
            "memory": self.workflow_cfg.memory,
            "platform": self.workflow_cfg.platform,
            "storage": self.workflow_cfg.storage,
        }

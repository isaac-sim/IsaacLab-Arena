# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Workflow classes for Isaac Lab Arena OSMO workflows.

A ``Workflow`` renders and submits one OSMO workflow; a ``CompositeWorkflow`` submits an
ordered sequence of them; both share the ``SubmittableWorkflow`` contract the submit script
drives. Workflows and their tasks declare parameters as config dataclasses (WorkflowCfg plus
a per-workflow task config). Only the top-level submit script turns those configs into CLI
flags; in-program callers construct the config objects directly.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import yaml
from abc import ABC, abstractmethod
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


@dataclass(frozen=True)
class WorkflowSubmissionResult:
    """Outcome of submitting a workflow to OSMO."""

    returncode: int
    """Exit code from ``osmo workflow submit`` (0 on success, and 0 for a dry-run)."""

    workflow_id: str | None = None
    """OSMO workflow ID parsed from the submit output; None on a dry-run or parse failure."""


class SubmittableWorkflow(ABC):
    """A configurable unit of work that can be submitted to OSMO.

    Implementations are either a single OSMO workflow (``Workflow``) or a composite that
    submits several in sequence (``CompositeWorkflow``). The top-level submit script needs
    only this contract: the two config dataclass types to build a CLI from, a constructor
    that takes the parsed configs, and ``submit_workflow``.
    """

    task_cfg_type: type[TaskCfg] = TaskCfg
    """Config dataclass type for this workflow's lead task. Subclasses set this."""

    workflow_cfg_type: type[WorkflowCfg] = WorkflowCfg
    """Workflow config dataclass type; subclasses override it to change resource defaults."""

    def __init__(self, workflow_cfg: WorkflowCfg, task_cfg: TaskCfg) -> None:
        self.workflow_cfg = workflow_cfg
        self.task_cfg = task_cfg

    @abstractmethod
    def submit_workflow(self) -> WorkflowSubmissionResult:
        """Submit to OSMO (or print the rendered spec on a dry-run) and return the outcome."""


class Workflow(SubmittableWorkflow):
    """Builds, renders, and submits a single Arena OSMO workflow."""

    task_cls_list: list[type[BaseTask]] = []
    """Task classes that make up this workflow, in group order. Subclasses must set this."""

    lead_list: list[bool] | None = None
    """Per-task lead flags; ``None`` lets a single-task workflow default its task to lead."""

    def __init__(
        self,
        workflow_cfg: WorkflowCfg,
        task_cfg: TaskCfg,
        group_name: str = "arena",
    ) -> None:
        assert len(self.task_cls_list) > 0, "Workflow subclasses must set task_cls_list"
        super().__init__(workflow_cfg=workflow_cfg, task_cfg=task_cfg)
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

    def submit_workflow(self) -> WorkflowSubmissionResult:
        """Render the workflow and either print it (dry-run) or submit it to OSMO."""
        rendered = self.render_yaml()
        if self.workflow_cfg.dry_run:
            print("[dry-run] Rendered workflow YAML:\n")
            print(rendered)
            return WorkflowSubmissionResult(returncode=0)

        return self._submit_rendered_workflow(rendered=rendered)

    def _get_tasks(self) -> list[BaseTask]:
        """Instantiate task objects for this workflow."""
        tasks = []
        for task_cls, lead in zip(self.task_cls_list, self.lead_flags):
            assert issubclass(task_cls, BaseTask)
            tasks.append(task_cls(self.task_cfg, lead=lead))
        return tasks

    def _submit_rendered_workflow(self, rendered: str) -> WorkflowSubmissionResult:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", prefix="arena_", delete=False) as f:
            f.write(rendered)
            rendered_path = f.name

        cmd = ["osmo", "workflow", "submit", rendered_path]
        if self.workflow_cfg.pool:
            cmd.extend(["--pool", self.workflow_cfg.pool])
        if self.workflow_cfg.priority:
            cmd.extend(["--priority", self.workflow_cfg.priority.value])

        print(f"Submitting workflow '{self.workflow_cfg.workflow_name}':")
        print(f"  {' '.join(cmd)}\n")

        try:
            # Capture stdout only, to parse the workflow ID; stderr stays inherited so
            # interactive output (login prompts, progress) remains visible live.
            result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
            print(result.stdout, end="")
            return WorkflowSubmissionResult(
                returncode=result.returncode,
                workflow_id=self._parse_submitted_workflow_id(result.stdout),
            )
        finally:
            Path(rendered_path).unlink(missing_ok=True)

    @staticmethod
    def _parse_submitted_workflow_id(submit_stdout: str) -> str | None:
        """Extract the workflow ID from ``osmo workflow submit`` output, or None."""
        match = re.search(r"Workflow ID\s*[-:]\s*(\S+)", submit_stdout)
        return match.group(1) if match else None

    def _create_resource_dict(self) -> dict[str, Any]:
        return {
            "cpu": self.workflow_cfg.cpus,
            "gpu": self.workflow_cfg.gpus,
            "memory": self.workflow_cfg.memory,
            "platform": self.workflow_cfg.platform,
            "storage": self.workflow_cfg.storage,
        }


class CompositeWorkflow(SubmittableWorkflow):
    """A workflow that submits an ordered sequence of sub-workflows instead of one group.

    Use this when the work cannot live in a single OSMO workflow — for example tasks that
    must run in different pools, which OSMO cannot gang-schedule into one group. Subclasses
    implement ``_submit_steps`` to submit each sub-workflow and thread results (such as a
    workflow ID a later step needs) between them.

    ``_submit_steps`` is responsible for honoring ``workflow_cfg.dry_run`` — each sub-workflow
    already does (it prints its spec instead of submitting), and a dry-run must still exercise
    the step sequencing, so the base intentionally does not short-circuit it.
    """

    def submit_workflow(self) -> WorkflowSubmissionResult:
        return self._submit_steps()

    @abstractmethod
    def _submit_steps(self) -> WorkflowSubmissionResult:
        """Submit the sub-workflows in order; return the result representing the composite."""

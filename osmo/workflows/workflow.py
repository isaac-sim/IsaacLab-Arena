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
from workflows.utils.yaml_utils import block_literal_str  # noqa: F401  (registers representer)


class Workflow:
    """Builds, renders, and submits an Arena OSMO workflow."""

    task_cls_list: list[type[BaseTask]] = []
    """Task classes that make up this workflow, in group order. Subclasses must set this."""

    lead_list: list[bool] | None = None
    """Per-task lead flags; ``None`` lets a single-task workflow default its task to lead."""

    def __init__(
        self,
        workflow_args: argparse.Namespace,
        task_args: argparse.Namespace,
        group_name: str = "arena",
    ) -> None:
        assert len(self.task_cls_list) > 0, "Workflow subclasses must set task_cls_list"
        self.workflow_args = workflow_args
        self.task_args = task_args
        # Single-task workflows may leave lead_list unset; that task defaults to lead.
        self.lead_flags = self.lead_list if self.lead_list is not None else [True]
        self._assert_single_lead_task(self.lead_flags)
        self.group_name = group_name

    @classmethod
    def build_parser(cls, description: str, epilog: str | None = None) -> argparse.ArgumentParser:
        """Build an argument parser populated with this workflow's args.

        Aggregates the workflow's tasks args, as well as the common args.

        Args:
            description: Parser description shown in ``--help``.
            epilog: Optional epilog text, e.g. the submit script's usage examples.

        Returns:
            The configured argument parser.
        """
        parser = argparse.ArgumentParser(
            description=description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=epilog,
        )
        cls._deduplicate_args(parser)
        cls.add_common_arguments(parser)
        return parser

    @classmethod
    def _deduplicate_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add each task class's arguments to the parser once, deduplicated across the task list."""
        added: set[type[BaseTask]] = set()
        for task_cls in cls.task_cls_list:
            if task_cls not in added:
                task_cls.add_task_arguments(parser)
                added.add(task_cls)

    @staticmethod
    def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add common flags for all workflows."""
        resources = parser.add_argument_group("resources")
        resources.add_argument("--cpus", type=int, default=15)
        resources.add_argument("--gpus", type=int, default=1)
        resources.add_argument("--memory", default="128Gi")
        resources.add_argument("--storage", default="200Gi")
        resources.add_argument("--platform", default="ovx-l40s")

        timeouts = parser.add_argument_group("timeouts")
        timeouts.add_argument("--exec_timeout", default="1d")
        timeouts.add_argument("--queue_timeout", default="2d")

        workflow = parser.add_argument_group("workflow")
        workflow.add_argument("--workflow_name", default="arena-workflow", help="OSMO workflow name")
        workflow.add_argument("--pool", default="isaac-dev-l40s-04", help="Target a specific OSMO compute pool")
        workflow.add_argument("--priority", default="NORMAL", choices=["HIGH", "NORMAL", "LOW"])

        parser.add_argument("--dry-run", action="store_true", help="Render without submitting")
        return parser

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
        for task_cls, lead in zip(self.task_cls_list, self.lead_flags):
            assert issubclass(task_cls, BaseTask)
            tasks.append(task_cls(self.workflow_args, self.task_args, lead=lead))
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

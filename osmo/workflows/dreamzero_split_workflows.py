# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Split DreamZero workflows: the server and the policy runner in separate OSMO pools.

The DreamZero server needs an H100-class GPU (it OOMs on an L40's 48 GiB) while the
policy runner's Isaac Sim rendering needs an RTX-capable GPU (the RTX renderer crashes
on dgx-h100 nodes). No accessible OSMO pool offers both platforms, and a workflow's
gang-scheduled group lives in a single pool, so the pair runs as two workflows connected
by an in-task port-forward tunnel.

``DreamZeroEvaluationWorkflow`` is the entry point: one submission launches both
workflows, and the runner cancels the server workflow when it exits so the pair
finishes together.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from osmo.tasks.base_task import BaseTask
from osmo.tasks.dreamzero_policy_runner_task import DreamZeroPolicyRunnerTask, DreamZeroPolicyRunnerTaskCfg
from osmo.tasks.dreamzero_server_task import DreamZeroServerTask, DreamZeroServerTaskCfg
from osmo.workflows.workflow import Workflow, WorkflowCfg


@dataclass
class DreamZeroWorkflowCfg(WorkflowCfg):
    """Workflow config with the DreamZero policy-runner's resource defaults.

    RTX-capable pool for Isaac Sim rendering; 1-GPU tasks on these nodes may claim at
    most 1/8 of node memory (~123Gi). ``gpus``, ``exec_timeout``, and ``queue_timeout``
    apply to the server workflow as well; the other server resources come from the
    ``server_*`` fields of ``DreamZeroEvaluationTaskCfg``.
    """

    pool: str = "isaac-dev-l40-03"
    platform: str = "ovx-l40"
    memory: str = "120Gi"


@dataclass
class DreamZeroEvaluationTaskCfg(DreamZeroPolicyRunnerTaskCfg):
    """Config for a full DreamZero evaluation: the runner task plus the server's resources."""

    server_pool: str = "isaac-dev-h100-01"
    """OSMO pool for the server workflow."""

    server_platform: str = "dgx-h100"
    """Platform for the server workflow."""

    server_cpus: int = 11
    """CPUs for the server workflow."""

    server_memory: str = "128Gi"
    """Memory for the server workflow."""

    server_storage: str = "100Gi"
    """Storage for the server workflow."""

    server_image: str = DreamZeroServerTaskCfg.image
    """Container image for the server workflow's task."""


class DreamZeroServerWorkflow(Workflow):
    """Workflow containing only the DreamZero inference server, for cross-pool evaluation."""

    task_cls_list = [DreamZeroServerTask]
    task_cfg_type = DreamZeroServerTaskCfg


class DreamZeroPolicyRunnerWorkflow(Workflow):
    """Workflow containing one policy-runner task that tunnels to a DreamZero server workflow.

    Internal to ``DreamZeroEvaluationWorkflow``, which supplies the server workflow ID
    the runner task requires.
    """

    task_cls_list = [DreamZeroPolicyRunnerTask]
    task_cfg_type = DreamZeroPolicyRunnerTaskCfg

    def __init__(
        self,
        workflow_cfg: WorkflowCfg,
        task_cfg: DreamZeroPolicyRunnerTaskCfg,
        server_workflow_id: str,
        group_name: str = "arena",
    ) -> None:
        super().__init__(workflow_cfg=workflow_cfg, task_cfg=task_cfg, group_name=group_name)
        self.server_workflow_id = server_workflow_id

    def _get_tasks(self) -> list[BaseTask]:
        return [DreamZeroPolicyRunnerTask(self.task_cfg, server_workflow_id=self.server_workflow_id, lead=True)]


class DreamZeroEvaluationWorkflow(Workflow):
    """Submit a full DreamZero evaluation: the server workflow, then the runner wired to it.

    Both submissions happen in one command: the server workflow is submitted first, its
    OSMO workflow ID is captured from the submit output, and the policy-runner workflow is
    submitted pointing at it. The runner tolerates the server's startup (image pull plus
    checkpoint load) through its tunnel wait loop and the policy's retrying initial
    connect, so no manual sequencing is needed, and it cancels the server workflow when it
    exits so both workflows finish together.
    """

    # Declared for parity with the workflow contract; the tasks are always rendered into
    # their own single-task workflows by submit_workflow, never into one group.
    task_cls_list = [DreamZeroPolicyRunnerTask, DreamZeroServerTask]
    task_cfg_type = DreamZeroEvaluationTaskCfg
    workflow_cfg_type = DreamZeroWorkflowCfg
    lead_list = [True, False]

    def generate_workflow(self) -> dict[str, Any]:
        """Guard against rendering this launcher as a single workflow."""
        raise AssertionError(
            "DreamZeroEvaluationWorkflow submits two separate single-task workflows; its tasks"
            " cannot be rendered into one gang-scheduled group. Use submit_workflow()."
        )

    def submit_workflow(self) -> int:
        """Submit the server workflow, then the policy-runner workflow tunnelled to it."""
        server_workflow = DreamZeroServerWorkflow(
            self._build_server_workflow_cfg(),
            DreamZeroServerTaskCfg(image=self.task_cfg.server_image),
        )
        returncode = server_workflow.submit_workflow()
        if returncode != 0:
            return returncode

        if self.workflow_cfg.dry_run:
            server_workflow_id = "dry-run-server-workflow-id"
        else:
            assert server_workflow.submitted_workflow_id, (
                "Could not parse the server workflow ID from the submit output. The server workflow"
                " may have been submitted anyway — check `osmo workflow list` and cancel it."
            )
            server_workflow_id = server_workflow.submitted_workflow_id
            print(f"DreamZero server workflow: {server_workflow_id}")

        runner_workflow = DreamZeroPolicyRunnerWorkflow(
            self.workflow_cfg, self.task_cfg, server_workflow_id=server_workflow_id
        )
        returncode = runner_workflow.submit_workflow()
        if returncode != 0 and not self.workflow_cfg.dry_run:
            print(
                f"Policy-runner submission failed; the DreamZero server workflow {server_workflow_id}"
                f" is still running — cancel it with `osmo workflow cancel {server_workflow_id}`."
            )
        return returncode

    def _build_server_workflow_cfg(self) -> WorkflowCfg:
        """Return a workflow config carrying the server's resource values and workflow name."""
        return replace(
            self.workflow_cfg,
            workflow_name=f"{self.workflow_cfg.workflow_name}-server",
            pool=self.task_cfg.server_pool,
            platform=self.task_cfg.server_platform,
            cpus=self.task_cfg.server_cpus,
            memory=self.task_cfg.server_memory,
            storage=self.task_cfg.server_storage,
        )

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

from dataclasses import dataclass, fields

from osmo.tasks.base_task import BaseTask
from osmo.tasks.dreamzero_policy_runner_task import DreamZeroPolicyRunnerTask, DreamZeroPolicyRunnerTaskCfg
from osmo.tasks.dreamzero_server_task import DreamZeroServerTask, DreamZeroServerTaskCfg
from osmo.workflows.workflow import CompositeWorkflow, Workflow, WorkflowCfg, WorkflowSubmissionResult

DREAMZERO_SERVER_TASK_NAME = "dreamzero_server"
DREAMZERO_POLICY_RUNNER_TASK_NAME = "policy_runner"


@dataclass
class DreamZeroWorkflowCfg(WorkflowCfg):
    """Workflow config with the DreamZero policy-runner's resource defaults.

    RTX-capable pool for Isaac Sim rendering; 1-GPU tasks on these nodes may claim at
    most 1/8 of node memory (~123Gi).
    """

    pool: str = "isaac-dev-l40-03"
    platform: str = "ovx-l40"
    memory: str = "120Gi"


@dataclass
class DreamZeroServerWorkflowCfg(WorkflowCfg):
    """Workflow config with the DreamZero server's H100 resource defaults."""

    pool: str = "isaac-dev-h100-01"
    platform: str = "dgx-h100"
    cpus: int = 11
    memory: str = "128Gi"
    storage: str = "100Gi"


@dataclass
class DreamZeroEvaluationTaskCfg(DreamZeroPolicyRunnerTaskCfg):
    """Config for a full DreamZero evaluation.

    Adds the server workflow's resource overrides. They are prefixed because the launcher
    exposes a single flat CLI for both sub-workflows, so their shared field names (``pool``,
    ``memory``, ...) would otherwise collide; the unprefixed fields configure the runner.
    Each ``server_*`` default mirrors the matching ``DreamZeroServerWorkflowCfg`` field so
    the two stay in sync; this reads the field default off the class, which is only valid
    because every mirrored field uses a plain (non-factory) default.
    """

    server_pool: str = DreamZeroServerWorkflowCfg.pool
    """OSMO pool for the server workflow."""

    server_platform: str = DreamZeroServerWorkflowCfg.platform
    """Platform for the server workflow."""

    server_cpus: int = DreamZeroServerWorkflowCfg.cpus
    """CPUs for the server workflow."""

    server_memory: str = DreamZeroServerWorkflowCfg.memory
    """Memory for the server workflow."""

    server_storage: str = DreamZeroServerWorkflowCfg.storage
    """Storage for the server workflow."""

    server_image: str = DreamZeroServerTaskCfg.image
    """Container image for the server workflow's task."""


class DreamZeroServerWorkflow(Workflow):
    """Workflow containing only the DreamZero inference server, for cross-pool evaluation."""

    task_cls_list = [DreamZeroServerTask]
    task_names = [DREAMZERO_SERVER_TASK_NAME]
    task_cfg_type = DreamZeroServerTaskCfg
    workflow_cfg_type = DreamZeroServerWorkflowCfg


class DreamZeroPolicyRunnerWorkflow(Workflow):
    """Workflow containing one policy-runner task that tunnels to a DreamZero server workflow.

    Internal to ``DreamZeroEvaluationWorkflow``, which supplies the server workflow ID
    the runner task requires.
    """

    task_cls_list = [DreamZeroPolicyRunnerTask]
    task_names = [DREAMZERO_POLICY_RUNNER_TASK_NAME]
    task_cfg_type = DreamZeroPolicyRunnerTaskCfg

    def __init__(
        self,
        workflow_cfg: WorkflowCfg,
        task_cfg: DreamZeroPolicyRunnerTaskCfg,
        server_workflow_id: str,
        group_name: str = "arena",
        *,
        server_task_name: str,
    ) -> None:
        super().__init__(workflow_cfg=workflow_cfg, task_cfg=task_cfg, group_name=group_name)
        self.server_workflow_id = server_workflow_id
        self.server_task_name = server_task_name

    def _get_tasks(self) -> list[BaseTask]:
        return [
            DreamZeroPolicyRunnerTask(
                task_name=self.task_names[0],
                task_cfg=self.task_cfg,
                server_workflow_id=self.server_workflow_id,
                server_task_name=self.server_task_name,
                lead=True,
            )
        ]


class DreamZeroEvaluationWorkflow(CompositeWorkflow):
    """Submit a full DreamZero evaluation: the server workflow, then the runner wired to it.

    Both submissions happen in one command: the server workflow is submitted first, its
    OSMO workflow ID is captured from the submit output, and the policy-runner workflow is
    submitted pointing at it. The runner tolerates the server's startup (image pull plus
    checkpoint load) through its tunnel wait loop and the policy's retrying initial
    connect, so no manual sequencing is needed, and it cancels the server workflow when it
    exits so both workflows finish together.
    """

    task_cfg_type = DreamZeroEvaluationTaskCfg
    workflow_cfg_type = DreamZeroWorkflowCfg

    def _submit_steps(self) -> WorkflowSubmissionResult:
        """Submit the server workflow, then the policy-runner workflow tunnelled to it.

        Returns the policy-runner workflow's submission result, whose workflow ID names
        the run that produces the evaluation outputs.
        """
        server_workflow = DreamZeroServerWorkflow(
            self._build_server_workflow_cfg(),
            DreamZeroServerTaskCfg(image=self.task_cfg.server_image),
        )
        server_result = server_workflow.submit_workflow()
        if server_result.returncode != 0:
            return server_result

        if self.workflow_cfg.dry_run:
            server_workflow_id = "dry-run-server-workflow-id"
        else:
            assert server_result.workflow_id, (
                "Could not parse the server workflow ID from the submit output. The server workflow"
                " may have been submitted anyway — check `osmo workflow list` and cancel it."
            )
            server_workflow_id = server_result.workflow_id
            print(f"DreamZero server workflow: {server_workflow_id}")

        runner_workflow = DreamZeroPolicyRunnerWorkflow(
            self.workflow_cfg,
            self.task_cfg,
            server_workflow_id=server_workflow_id,
            server_task_name=server_workflow.task_names[0],
        )
        runner_result = runner_workflow.submit_workflow()
        if runner_result.returncode != 0 and not self.workflow_cfg.dry_run:
            print(
                f"Policy-runner submission failed; the DreamZero server workflow {server_workflow_id}"
                f" is still running — cancel it with `osmo workflow cancel {server_workflow_id}`."
            )
        return runner_result

    def _build_server_workflow_cfg(self) -> DreamZeroServerWorkflowCfg:
        """Build the server workflow's config from its resource fields and the shared run settings.

        The resource fields (pool, platform, cpus, memory, storage) come from the
        ``server_*`` overrides, and the workflow name gets a ``-server`` suffix; every
        remaining field (priority, GPU count, timeouts, dry-run) is carried over verbatim
        from the runner workflow's config. Carrying the full runner config forward — rather
        than naming individual fields — keeps the server in sync automatically if
        ``WorkflowCfg`` gains a run-scoped field.
        """
        shared = {field.name: getattr(self.workflow_cfg, field.name) for field in fields(WorkflowCfg)}
        return DreamZeroServerWorkflowCfg(**{
            **shared,
            "workflow_name": f"{self.workflow_cfg.workflow_name}-server",
            "pool": self.task_cfg.server_pool,
            "platform": self.task_cfg.server_platform,
            "cpus": self.task_cfg.server_cpus,
            "memory": self.task_cfg.server_memory,
            "storage": self.task_cfg.server_storage,
        })

# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""OSMO workflows for evaluating complete Arena Experiments."""

from __future__ import annotations

import math
import re
from copy import deepcopy
from dataclasses import replace
from typing import Any

from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentCfg
from isaaclab_arena.evaluation.arena_run import ArenaRunCfg
from isaaclab_arena_openpi.policy.pi0_remote_config import Pi0RemotePolicyCfg
from osmo.tasks.base_task import BaseTask
from osmo.tasks.experiment_output_coordinator_task import ExperimentOutputCoordinatorTask
from osmo.tasks.experiment_runner_task import ExperimentRunnerTask, ExperimentRunnerTaskCfg
from osmo.tasks.pi0_server_task import Pi0ServerTask, Pi0ServerTaskCfg
from osmo.workflows.workflow import CompositeWorkflow, Workflow, WorkflowCfg, WorkflowSubmissionResult
from osmo.workflows.workflow_constants import DATASETS_HTTPS_URL, DATASETS_SWIFT_URL, POLICY_SERVER_PORT

_COORDINATOR_WORKFLOW_NAME = "arena-experiment-output-coordinator"
_DRY_RUN_SOURCE_WORKFLOW_ID = "dry-run-experiment-workflow-id"
_COORDINATOR_TIMEOUT_GRACE_SECONDS = 300
_ISO_8601_DURATION_PATTERN = re.compile(
    r"^P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?" r"(?:(?P<seconds>\d+(?:\.\d+)?)S)?)?$"
)
_SECONDS_BY_DURATION_UNIT = {
    "d": 24 * 60 * 60,
    "h": 60 * 60,
    "m": 60,
    "s": 1,
    "ms": 1 / 1_000,
    "us": 1 / 1_000_000,
}


def _osmo_duration_seconds(duration: str) -> float:
    """Convert an OSMO duration string to seconds."""
    if duration.startswith("P"):
        match = _ISO_8601_DURATION_PATTERN.fullmatch(duration)
        assert match and any(match.groupdict().values()), f"Invalid OSMO duration '{duration}'"
        return (
            int(match.group("days") or 0) * _SECONDS_BY_DURATION_UNIT["d"]
            + int(match.group("hours") or 0) * _SECONDS_BY_DURATION_UNIT["h"]
            + int(match.group("minutes") or 0) * _SECONDS_BY_DURATION_UNIT["m"]
            + float(match.group("seconds") or 0)
        )

    duration_unit = next((unit for unit in ("ms", "us", "d", "h", "m", "s") if duration.endswith(unit)), None)
    assert duration_unit is not None, f"Invalid OSMO duration '{duration}'"
    duration_value = duration[: -len(duration_unit)]
    assert duration_value.isdigit(), f"Invalid OSMO duration '{duration}'"
    return int(duration_value) * _SECONDS_BY_DURATION_UNIT[duration_unit]


def _experiment_output_coordinator_exec_timeout(source_workflow_cfg: WorkflowCfg) -> str:
    """Cover source queue/execution, report collection, and short orchestration delays."""
    source_queue_seconds = _osmo_duration_seconds(source_workflow_cfg.queue_timeout)
    source_execution_seconds = _osmo_duration_seconds(source_workflow_cfg.exec_timeout)
    coordinator_execution_seconds = (
        source_queue_seconds + 2 * source_execution_seconds + _COORDINATOR_TIMEOUT_GRACE_SECONDS
    )
    return f"{math.ceil(coordinator_execution_seconds)}s"


class Pi0ArenaExperimentRunsWorkflow(Workflow):
    """Run every Arena Experiment Run in its own OSMO group."""

    constructs_groups_directly = True
    task_cfg_type = ExperimentRunnerTaskCfg
    server_task_cfg_type = Pi0ServerTaskCfg

    def __init__(
        self,
        workflow_cfg: WorkflowCfg,
        experiment_cfg: ArenaExperimentCfg,
        server_task_cfg: Pi0ServerTaskCfg,
        group_name: str = "arena",
        task_cfg: ExperimentRunnerTaskCfg | None = None,
    ) -> None:
        assert isinstance(experiment_cfg, ArenaExperimentCfg)
        self.experiment_cfg = deepcopy(experiment_cfg)
        self.pi0_server_task_cfg = server_task_cfg
        super().__init__(
            workflow_cfg=workflow_cfg,
            task_cfg=task_cfg or ExperimentRunnerTaskCfg(),
            group_name=group_name,
        )

        # Every Pi0 Run gets a dedicated server task. Verify that all of those
        # clients request the variant configured for the server deployment.
        pi0_policy_variants_by_run = self._get_pi0_policy_variants_by_run()
        self._assert_pi0_server_compatible(pi0_policy_variants_by_run)

    @property
    def experiment_runner_task_names_by_run_name(self) -> dict[str, str]:
        """Map every Run name to its deterministic OSMO Experiment Runner task name."""
        return {
            run_name: f"experiment-runner-{run_index}" for run_index, run_name in enumerate(self.experiment_cfg.runs)
        }

    def _get_group_dicts(self) -> list[dict[str, Any]]:
        """Create one independently scheduled group per Run."""
        return [
            self._create_run_group_dict(run_index, run_name, run_config)
            for run_index, (run_name, run_config) in enumerate(self.experiment_cfg.runs.items())
        ]

    def _create_run_group_dict(
        self,
        run_index: int,
        run_name: str,
        run_config: ArenaRunCfg,
    ) -> dict[str, Any]:
        """Create one OSMO group that executes a single-Run Arena Experiment."""
        experiment_runner_task_name = self.experiment_runner_task_names_by_run_name[run_name]
        single_run_experiment_config = ArenaExperimentCfg(runs={run_name: deepcopy(run_config)})

        pi0_policy_server_tasks: list[BaseTask] = []
        run_policy_config = single_run_experiment_config.runs[run_name].policy
        if isinstance(run_policy_config, Pi0RemotePolicyCfg):
            pi0_server_task_name = f"policy-server-{run_index}"
            self._configure_pi0_remote_policy_for_server(run_policy_config, pi0_server_task_name)
            pi0_policy_server_tasks.append(
                Pi0ServerTask(
                    self.pi0_server_task_cfg,
                    lead=False,
                    task_name=pi0_server_task_name,
                )
            )

        # Construct this after connecting the policy because the task snapshots the Experiment.
        experiment_runner_task = ExperimentRunnerTask(
            task_cfg=self.task_cfg,
            experiment_cfg=single_run_experiment_config,
            lead=True,
            task_name=experiment_runner_task_name,
            published_output_url=None,
        )
        run_group_tasks = [experiment_runner_task, *pi0_policy_server_tasks]
        return {
            "name": f"arena-run-{run_index}",
            "tasks": [run_group_task.create_task_dict() for run_group_task in run_group_tasks],
        }

    def _get_pi0_policy_variants_by_run(self) -> dict[str, str]:
        """Return effective pi0-remote Run variants needed for compatibility checks."""
        pi0_policy_variants_by_run = {}
        for run_name, run_config in self.experiment_cfg.runs.items():
            if not isinstance(run_config.policy, Pi0RemotePolicyCfg):
                continue
            pi0_policy_variants_by_run[run_name] = run_config.policy.policy_variant
        return pi0_policy_variants_by_run

    def _assert_pi0_server_compatible(self, pi0_policy_variants_by_run: dict[str, str]) -> None:
        """Require Pi0RemotePolicy Runs whose variants match the deployed server."""
        assert pi0_policy_variants_by_run, "pi0 server requires at least one Run using Pi0RemotePolicy"
        incompatible_policy_variants_by_run = {
            run_name: policy_variant
            for run_name, policy_variant in pi0_policy_variants_by_run.items()
            if policy_variant != self.pi0_server_task_cfg.policy_variant
        }
        assert not incompatible_policy_variants_by_run, (
            f"pi0_remote Runs require variants {incompatible_policy_variants_by_run}, but the pi0 server is configured"
            f" for '{self.pi0_server_task_cfg.policy_variant}'"
        )

    def _configure_pi0_remote_policy_for_server(
        self,
        pi0_remote_policy_config: Pi0RemotePolicyCfg,
        pi0_server_task_name: str,
    ) -> None:
        """Configure a Pi0 remote policy to use its dedicated OSMO server task."""
        pi0_remote_policy_config.remote_host = Pi0ServerTask.host_token(pi0_server_task_name)
        pi0_remote_policy_config.remote_port = POLICY_SERVER_PORT
        # The first OSMO inference may compile longer than the policy's normal
        # keepalive timeout. Use the timeout owned by this server deployment.
        pi0_remote_policy_config.ping_timeout = self.pi0_server_task_cfg.client_ping_timeout_s


class ExperimentOutputCoordinatorWorkflow(Workflow):
    """Wait for one Runs workflow and collect only its successful task outputs."""

    constructs_groups_directly = True
    task_cfg_type = ExperimentRunnerTaskCfg

    def __init__(
        self,
        workflow_cfg: WorkflowCfg,
        task_cfg: ExperimentRunnerTaskCfg,
        source_workflow_id: str,
        experiment_runner_task_names_by_run_name: dict[str, str],
        published_output_url: str,
    ) -> None:
        self.source_workflow_id = source_workflow_id
        self.experiment_runner_task_names_by_run_name = experiment_runner_task_names_by_run_name
        self.published_output_url = published_output_url
        super().__init__(workflow_cfg=workflow_cfg, task_cfg=task_cfg, group_name="arena-experiment-output")

    def _get_group_dicts(self) -> list[dict[str, Any]]:
        """Create the independent collector group with no source task dependencies."""
        coordinator_task = ExperimentOutputCoordinatorTask(
            task_name="collect-successful-experiment-outputs",
            image=self.task_cfg.image,
            source_workflow_id=self.source_workflow_id,
            experiment_runner_task_names_by_run_name=self.experiment_runner_task_names_by_run_name,
            published_output_url=self.published_output_url,
            lead=True,
        )
        return [{"name": self.group_name, "tasks": [coordinator_task.create_task_dict()]}]


class Pi0ArenaExperimentWorkflow(CompositeWorkflow):
    """Submit Pi0 Arena Runs and their successful-output collector with one command."""

    task_cfg_type = ExperimentRunnerTaskCfg
    server_task_cfg_type = Pi0ServerTaskCfg

    def __init__(
        self,
        workflow_cfg: WorkflowCfg,
        experiment_cfg: ArenaExperimentCfg,
        server_task_cfg: Pi0ServerTaskCfg,
        group_name: str = "arena",
        task_cfg: ExperimentRunnerTaskCfg | None = None,
    ) -> None:
        experiment_runner_task_cfg = task_cfg or ExperimentRunnerTaskCfg()
        super().__init__(workflow_cfg=workflow_cfg, task_cfg=experiment_runner_task_cfg)
        self.experiment_runs_workflow = Pi0ArenaExperimentRunsWorkflow(
            workflow_cfg=workflow_cfg,
            experiment_cfg=experiment_cfg,
            server_task_cfg=server_task_cfg,
            group_name=group_name,
            task_cfg=experiment_runner_task_cfg,
        )

    def _submit_steps(self) -> WorkflowSubmissionResult:
        """Submit the Runs first, then an independent workflow that collects their successful outputs."""
        experiment_runs_result = self.experiment_runs_workflow.submit_workflow()
        if experiment_runs_result.returncode != 0:
            return experiment_runs_result

        if self.workflow_cfg.dry_run:
            source_workflow_id = _DRY_RUN_SOURCE_WORKFLOW_ID
        else:
            assert experiment_runs_result.workflow_id, (
                "Could not parse the Arena Experiment workflow ID from the OSMO submission output. The Runs may have"
                " been submitted anyway; check `osmo workflow list` before retrying."
            )
            source_workflow_id = experiment_runs_result.workflow_id
            print(f"Arena Experiment Runs workflow: {source_workflow_id}")

        published_output_url = f"{DATASETS_SWIFT_URL}/{source_workflow_id}"
        experiment_report_url = f"{DATASETS_HTTPS_URL}/{source_workflow_id}/index.html"
        coordinator_workflow = ExperimentOutputCoordinatorWorkflow(
            workflow_cfg=self._create_coordinator_workflow_cfg(),
            task_cfg=self.task_cfg,
            source_workflow_id=source_workflow_id,
            experiment_runner_task_names_by_run_name=(
                self.experiment_runs_workflow.experiment_runner_task_names_by_run_name
            ),
            published_output_url=published_output_url,
        )
        coordinator_result = coordinator_workflow.submit_workflow()
        if coordinator_result.returncode != 0 and not self.workflow_cfg.dry_run:
            print(
                f"Output collector submission failed; the Arena Experiment Runs workflow {source_workflow_id}"
                " is still running and its successful task outputs remain stored by OSMO."
            )
        elif not self.workflow_cfg.dry_run:
            print(f"Arena Experiment report (available after collection): {experiment_report_url}")
        return WorkflowSubmissionResult(
            returncode=coordinator_result.returncode,
            workflow_id=experiment_runs_result.workflow_id,
        )

    def _create_coordinator_workflow_cfg(self) -> WorkflowCfg:
        """Create a CPU-only collector config with enough time to wait for and collect the Runs."""
        return replace(
            self.workflow_cfg,
            workflow_name=_COORDINATOR_WORKFLOW_NAME,
            cpus=2,
            gpus=0,
            memory="8Gi",
            exec_timeout=_experiment_output_coordinator_exec_timeout(self.workflow_cfg),
        )

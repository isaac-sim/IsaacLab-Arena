# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""OSMO workflow for evaluating complete Arena Experiments."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentCfg
from isaaclab_arena.evaluation.arena_run import ArenaRunCfg
from osmo.tasks.base_task import BaseTask
from osmo.tasks.collect_experiment_outputs_task import CollectExperimentOutputsTask
from osmo.tasks.experiment_runner_task import ExperimentRunnerTask, ExperimentRunnerTaskCfg
from osmo.workflows.server_bindings import REMOTE_POLICY_SERVERS, ServerBinding, ServersCfg
from osmo.workflows.workflow import Workflow, WorkflowCfg


class ArenaExperimentWorkflow(Workflow):
    """Run every Arena Experiment Run in its own OSMO group, co-scheduling each Run's server.

    The server for a Run is derived from its client policy config type via
    ``REMOTE_POLICY_SERVERS``: a matching Run gets a dedicated server task wired to it, and a
    Run whose policy has no registered server (e.g. a local zero-action policy) runs standalone.
    Its per-type server deployment config comes from ``servers.<name>``.
    """

    constructs_groups_directly = True
    task_cfg_type = ExperimentRunnerTaskCfg
    experiment_output_resource_name = "experiment-output"

    def __init__(
        self,
        workflow_cfg: WorkflowCfg,
        experiment_cfg: ArenaExperimentCfg,
        servers: ServersCfg,
        group_name: str = "arena",
        task_cfg: ExperimentRunnerTaskCfg | None = None,
    ) -> None:
        assert isinstance(experiment_cfg, ArenaExperimentCfg)
        self.experiment_cfg = deepcopy(experiment_cfg)
        self.servers = servers
        super().__init__(
            workflow_cfg=workflow_cfg,
            task_cfg=task_cfg or ExperimentRunnerTaskCfg(),
            group_name=group_name,
        )
        self._assert_servers_share_one_pool()
        self._run_server_checks()

    def _server_cfg_for(self, binding: ServerBinding) -> Any:
        """Return the deployment config for a server type from the ``servers`` map."""
        return getattr(self.servers, binding.name)

    def _runs_by_binding(self) -> dict[ServerBinding, list[ArenaRunCfg]]:
        """Group the Runs that need a server by the server binding that serves them."""
        runs_by_binding: dict[ServerBinding, list[ArenaRunCfg]] = {}
        for run_cfg in self.experiment_cfg.runs.values():
            binding = REMOTE_POLICY_SERVERS.get(type(run_cfg.policy))
            if binding is not None:
                runs_by_binding.setdefault(binding, []).append(run_cfg)
        return runs_by_binding

    def _assert_servers_share_one_pool(self) -> None:
        """Require every server the Experiment needs to run on one resource (one submission = one pool)."""
        required_resources = {(binding.pool, binding.platform) for binding in self._runs_by_binding()}
        assert len(required_resources) <= 1, (
            f"Experiment needs servers on different resources {sorted(required_resources)}; a submission runs on"
            " a single pool. Run the incompatible policies as separate submissions."
        )

    def _run_server_checks(self) -> None:
        """Run each server type's compatibility check against the Runs it serves."""
        for binding, runs_using_binding in self._runs_by_binding().items():
            binding.check(runs_using_binding, self._server_cfg_for(binding))

    def _get_group_dicts(self) -> list[dict[str, Any]]:
        """Create one independently scheduled group per Run, then collect their outputs into one Experiment output."""
        run_group_dicts: list[dict[str, Any]] = []
        experiment_runner_task_names_by_run_name: dict[str, str] = {}
        for run_index, (run_name, run_config) in enumerate(self.experiment_cfg.runs.items()):
            run_group_dict, experiment_runner_task_name = self._create_run_group_dict(
                run_index,
                run_name,
                run_config,
            )
            run_group_dicts.append(run_group_dict)
            experiment_runner_task_names_by_run_name[run_name] = experiment_runner_task_name

        experiment_output_group_dict = self._create_experiment_output_group_dict(
            experiment_runner_task_names_by_run_name
        )
        return [*run_group_dicts, experiment_output_group_dict]

    def _create_run_group_dict(
        self,
        run_index: int,
        run_name: str,
        run_config: ArenaRunCfg,
    ) -> tuple[dict[str, Any], str]:
        """Create one OSMO group that executes a single-Run Arena Experiment, plus its server if any."""
        experiment_runner_task_name = f"experiment-runner-{run_index}"
        single_run_experiment_config = ArenaExperimentCfg(runs={run_name: deepcopy(run_config)})

        policy_server_tasks: list[BaseTask] = []
        run_policy_config = single_run_experiment_config.runs[run_name].policy
        binding = REMOTE_POLICY_SERVERS.get(type(run_policy_config))
        if binding is not None:
            server_task_name = f"policy-server-{run_index}"
            server_cfg = self._server_cfg_for(binding)
            binding.configure_client(run_policy_config, server_task_name, server_cfg)
            policy_server_tasks.append(binding.server_task_cls(server_cfg, lead=False, task_name=server_task_name))

        # Construct this after connecting the policy because the task snapshots the Experiment.
        experiment_runner_task = ExperimentRunnerTask(
            task_cfg=self.task_cfg,
            experiment_cfg=single_run_experiment_config,
            lead=True,
            task_name=experiment_runner_task_name,
            published_output_url=None,
        )
        run_group_tasks = [experiment_runner_task, *policy_server_tasks]

        run_group_dict = {
            "name": f"arena-run-{run_index}",
            "tasks": [run_group_task.create_task_dict() for run_group_task in run_group_tasks],
        }
        return run_group_dict, experiment_runner_task_name

    def _create_experiment_output_group_dict(
        self,
        experiment_runner_task_names_by_run_name: dict[str, str],
    ) -> dict[str, Any]:
        """Collect every Experiment Runner task output into one published Experiment output."""
        collect_experiment_outputs_task = CollectExperimentOutputsTask(
            task_name="collect-experiment-outputs",
            image=self.task_cfg.image,
            experiment_runner_task_names_by_run_name=experiment_runner_task_names_by_run_name,
            lead=True,
            resource=self.experiment_output_resource_name,
        )
        return {
            "name": "arena-experiment-output",
            "tasks": [collect_experiment_outputs_task.create_task_dict()],
        }

    def _create_resources_dict(self) -> dict[str, dict[str, Any]]:
        """Use configured resources for Runs and a CPU-only resource for collecting the Experiment output."""
        run_task_resource = self._create_resource_dict()
        experiment_output_task_resource = {**run_task_resource, "gpu": 0}
        return {
            "default": run_task_resource,
            self.experiment_output_resource_name: experiment_output_task_resource,
        }

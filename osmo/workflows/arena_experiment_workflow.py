# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""OSMO workflows for evaluating complete Arena Experiments."""

from __future__ import annotations

import yaml
from collections.abc import Sequence
from pathlib import Path

from isaaclab_arena.hydra.typed_experiment_loader import load_experiment_run_definitions_from_yaml
from osmo.tasks.base_task import BaseTask
from osmo.tasks.eval_runner_task import EvalRunnerTask, EvalRunnerTaskCfg
from osmo.tasks.pi0_server_task import DEFAULT_PI0_POLICY_VARIANT, Pi0ServerTask, Pi0ServerTaskCfg
from osmo.workflows.workflow import Workflow, WorkflowCfg
from osmo.workflows.workflow_constants import POLICY_SERVER_PORT


class ArenaExperimentWorkflow(Workflow):
    """Run a complete Arena Experiment without deploying a policy server."""

    task_cls_list = [EvalRunnerTask]
    task_cfg_type = EvalRunnerTaskCfg
    lead_list = [True]

    def __init__(
        self,
        workflow_cfg: WorkflowCfg,
        experiment_config_path: str | Path,
        experiment_overrides: Sequence[str] = (),
        group_name: str = "arena",
        task_cfg: EvalRunnerTaskCfg | None = None,
    ) -> None:
        experiment_path = Path(experiment_config_path).expanduser().resolve()
        assert experiment_path.is_file(), f"Experiment config does not exist: {experiment_path}"
        assert experiment_path.suffix.lower() in {".yaml", ".yml"}, "OSMO supports typed YAML Experiments only"

        self.experiment_config_path = experiment_path
        self.experiment_overrides = list(experiment_overrides)

        super().__init__(
            workflow_cfg=workflow_cfg,
            task_cfg=task_cfg or EvalRunnerTaskCfg(),
            group_name=group_name,
        )

    def _get_tasks(self) -> list[BaseTask]:
        """Create the lead evaluator."""
        return [self._create_eval_runner_task()]

    def _create_eval_runner_task(self) -> EvalRunnerTask:
        """Create the Experiment's lead eval-runner task."""
        return EvalRunnerTask(
            task_cfg=self.task_cfg,
            experiment_config_path=self.experiment_config_path,
            experiment_overrides=self.experiment_overrides,
            lead=self.lead_flags[0],
        )


class Pi0ArenaExperimentWorkflow(ArenaExperimentWorkflow):
    """Run an Arena Experiment with one shared pi0 policy server."""

    task_cls_list = [EvalRunnerTask, Pi0ServerTask]
    server_task_cfg_type = Pi0ServerTaskCfg
    """Configuration type loaded from the selected server definition."""

    lead_list = [True, False]

    def __init__(
        self,
        workflow_cfg: WorkflowCfg,
        experiment_config_path: str | Path,
        server_task_cfg: Pi0ServerTaskCfg,
        experiment_overrides: Sequence[str] = (),
        group_name: str = "arena",
        task_cfg: EvalRunnerTaskCfg | None = None,
    ) -> None:
        self.pi0_server_task_cfg = server_task_cfg
        super().__init__(
            workflow_cfg=workflow_cfg,
            experiment_config_path=experiment_config_path,
            task_cfg=task_cfg,
            experiment_overrides=experiment_overrides,
            group_name=group_name,
        )

        pi0_run_variants = self._read_pi0_run_variants()
        assert pi0_run_variants, "pi0 server requires at least one Run with policy.type 'pi0_remote'"
        self._apply_variant_overrides(pi0_run_variants)
        incompatible_runs = {
            run_name: variant
            for run_name, variant in pi0_run_variants.items()
            if variant != self.pi0_server_task_cfg.policy_variant
        }
        assert not incompatible_runs, (
            f"pi0_remote Runs require variants {incompatible_runs}, but the pi0 server is configured for "
            f"'{self.pi0_server_task_cfg.policy_variant}'"
        )
        self.experiment_overrides.extend(self._get_endpoint_overrides(list(pi0_run_variants)))

    def _get_tasks(self) -> list[BaseTask]:
        """Create the lead evaluator and non-lead pi0 server."""
        return [
            self._create_eval_runner_task(),
            Pi0ServerTask(self.pi0_server_task_cfg, lead=self.lead_flags[1]),
        ]

    def _read_pi0_run_variants(self) -> dict[str, str]:
        """Return literal pi0-remote Run variants needed for compatibility checks."""
        run_values_by_name = load_experiment_run_definitions_from_yaml(self.experiment_config_path)
        pi0_run_variants = {}
        for run_name, run_value in run_values_by_name.items():
            policy_values = run_value.get("policy")
            if not isinstance(policy_values, dict) or policy_values.get("type") != "pi0_remote":
                continue
            policy_variant = policy_values.get("policy_variant", DEFAULT_PI0_POLICY_VARIANT)
            assert isinstance(policy_variant, str), f"pi0 Run '{run_name}' policy_variant must be a string"
            pi0_run_variants[run_name] = policy_variant
        return pi0_run_variants

    def _apply_variant_overrides(self, run_variants: dict[str, str]) -> None:
        """Apply exact per-Run policy-variant overrides in declaration order."""
        variant_paths = {f"runs.{run_name}.policy.policy_variant": run_name for run_name in run_variants}
        for override in self.experiment_overrides:
            override_path, separator, raw_value = override.partition("=")
            run_name = variant_paths.get(override_path.lstrip("+"))
            if run_name is None:
                continue
            assert separator, f"pi0 policy_variant override must assign a value: {override}"
            try:
                policy_variant = yaml.safe_load(raw_value)
            except yaml.YAMLError as exc:
                raise AssertionError(f"Invalid pi0 policy_variant override: {override}") from exc
            assert isinstance(policy_variant, str), f"pi0 policy_variant override must be a string: {override}"
            run_variants[run_name] = policy_variant

    @staticmethod
    def _get_endpoint_overrides(run_names: Sequence[str]) -> list[str]:
        """Connect matching pi0 Runs to the shared server task."""
        host_token = Pi0ServerTask.host_token()
        overrides = []
        for run_name in run_names:
            overrides.extend([
                f'runs.{run_name}.policy.remote_host="{host_token}"',
                f"runs.{run_name}.policy.remote_port={POLICY_SERVER_PORT}",
            ])
        return overrides

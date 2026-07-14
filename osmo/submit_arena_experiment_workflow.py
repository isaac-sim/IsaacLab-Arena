# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compose and submit one Arena Experiment as an OSMO workflow.

Example:

    python -m osmo.submit_arena_experiment_workflow \
        experiment_config=openpi_experiment \
        server_config=pi0 \
        osmo_config.pool=isaac-dev-l40-03 \
        osmo_config.platform=ovx-l40 \
        osmo_config.memory=120Gi \
        osmo_config.workflow_name=my-evaluation \
        eval_runner_config.image=nvcr.io/example/isaaclab_arena:experiment_runner \
        experiment_config.runs.openpi_maple_table.rollout_limit.num_episodes=4

The named config groups select an Experiment definition, optional policy-server
definition, OSMO scheduling settings, and eval-runner task settings. Hydra
applies trailing field overrides after the selected files.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra import main as hydra_main
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

from isaaclab_arena.evaluation.arena_experiment_config_loader import load_arena_experiment_from_config_file
from isaaclab_arena.evaluation.arena_run import ArenaRunCfg
from osmo.tasks.base_task import TaskCfg
from osmo.tasks.eval_runner_task import EvalRunnerTaskCfg
from osmo.workflows.arena_experiment_workflow import ArenaExperimentWorkflow, Pi0ArenaExperimentWorkflow
from osmo.workflows.workflow import WorkflowCfg

CONFIG_DIR = Path(__file__).parent / "config"
SUBMISSION_CONFIG_NAME = "arena_experiment_submission"
SUBMISSION_SCHEMA_NAME = "arena_experiment_submission_schema"

POLICY_SERVER_WORKFLOWS = {
    "pi0": Pi0ArenaExperimentWorkflow,
}


@dataclass
class ArenaExperimentSubmissionCfg:
    """Configuration domains composed for one OSMO Experiment submission."""

    experiment_config: dict[str, Any] = MISSING
    """Selected Arena Experiment definition."""

    osmo_config: WorkflowCfg = field(default_factory=WorkflowCfg)
    """OSMO scheduling, resource, and timeout configuration."""

    eval_runner_config: EvalRunnerTaskCfg = field(default_factory=EvalRunnerTaskCfg)
    """Configuration for the task that executes ``eval_runner.py``."""

    server_config: TaskCfg | None = None
    """Optional policy-server definition."""


_config_store = ConfigStore.instance()
_config_store.store(name=SUBMISSION_SCHEMA_NAME, node=ArenaExperimentSubmissionCfg)
for server_name, workflow_cls in POLICY_SERVER_WORKFLOWS.items():
    _config_store.store(group="server_config", name=server_name, node=workflow_cls.server_task_cfg_type)


def compose_arena_experiment_submission(overrides: list[str] | None = None) -> ArenaExperimentSubmissionCfg:
    """Compose the submission config from its named groups and Hydra overrides.

    Args:
        overrides: Hydra config-group selections and field overrides.

    Returns:
        The fully composed typed submission configuration.
    """
    _register_typed_experiment_configs()
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        composed = compose(config_name=SUBMISSION_CONFIG_NAME, overrides=overrides or [])
    return _submission_cfg_from_hydra(composed)


def _submission_cfg_from_hydra(composed: Any) -> ArenaExperimentSubmissionCfg:
    """Convert Hydra's composed object into the typed submission root."""
    submission_cfg = OmegaConf.to_object(composed)
    assert isinstance(submission_cfg, ArenaExperimentSubmissionCfg)
    return submission_cfg


def submit_arena_experiment_workflow(submission_cfg: ArenaExperimentSubmissionCfg) -> int:
    """Build and submit the OSMO workflow described by ``submission_cfg``.

    Args:
        submission_cfg: Composed Experiment, task, server, and OSMO configuration.

    Returns:
        The OSMO submission process status.
    """
    experiment_config = _validated_experiment_config(submission_cfg.experiment_config)
    server_task_cfg = deepcopy(submission_cfg.server_config)
    if server_task_cfg is None:
        workflow = ArenaExperimentWorkflow(
            workflow_cfg=submission_cfg.osmo_config,
            experiment_config=experiment_config,
            task_cfg=submission_cfg.eval_runner_config,
        )
    else:
        workflows_by_server_cfg_type = {
            workflow_type.server_task_cfg_type: workflow_type for workflow_type in POLICY_SERVER_WORKFLOWS.values()
        }
        workflow_cls = workflows_by_server_cfg_type.get(type(server_task_cfg))
        assert (
            workflow_cls is not None
        ), f"No policy-server workflow is registered for configuration type {type(server_task_cfg).__name__}"
        workflow = workflow_cls(
            workflow_cfg=submission_cfg.osmo_config,
            experiment_config=experiment_config,
            server_task_cfg=server_task_cfg,
            task_cfg=submission_cfg.eval_runner_config,
        )
    return workflow.submit_workflow()


def _validated_experiment_config(experiment_config: dict[str, Any]) -> dict[str, Any]:
    """Return an independent Arena Experiment mapping with a valid envelope."""
    assert isinstance(experiment_config, dict), "Experiment config must be a mapping"
    unknown_fields = sorted(set(experiment_config) - {"runs"})
    assert not unknown_fields, f"Unknown Experiment fields: {', '.join(unknown_fields)}"
    runs = experiment_config.get("runs")
    assert isinstance(runs, dict) and runs, "Experiment config must define a non-empty 'runs' mapping"
    for run_name, run_cfg in runs.items():
        assert isinstance(run_cfg, ArenaRunCfg), f"Run '{run_name}' must be a typed ArenaRunCfg"
        assert run_cfg.name == run_name, f"Run key '{run_name}' does not match its typed name '{run_cfg.name}'"
    return deepcopy(experiment_config)


@cache
def _register_typed_experiment_configs() -> None:
    """Expose repository Experiment YAML files as typed Hydra group options."""
    import isaaclab_arena_environments

    experiment_config_dir = Path(isaaclab_arena_environments.__file__).parent / "experiment_configs"
    config_paths = sorted([*experiment_config_dir.glob("*.yaml"), *experiment_config_dir.glob("*.yml")])
    assert config_paths, f"No Arena Experiment configs found in '{experiment_config_dir}'"
    for config_path in config_paths:
        experiment = load_arena_experiment_from_config_file(config_path, device="cuda:0")
        _config_store.store(
            group="experiment_config",
            name=config_path.stem,
            node={"runs": {run_cfg.name: run_cfg for run_cfg in experiment}},
        )


@hydra_main(version_base=None, config_path="config", config_name=SUBMISSION_CONFIG_NAME)
def _hydra_cli(composed: Any) -> None:
    """Submit the workflow composed by Hydra's command-line frontend."""
    status = submit_arena_experiment_workflow(_submission_cfg_from_hydra(composed))
    if status:
        raise SystemExit(status)


def main(cli_args: list[str] | None = None) -> int:
    """Run the Hydra CLI, or compose explicit overrides for an in-process caller."""
    if cli_args is None:
        _register_typed_experiment_configs()
        _hydra_cli()
        return 0
    return submit_arena_experiment_workflow(compose_arena_experiment_submission(cli_args))


if __name__ == "__main__":
    raise SystemExit(main())

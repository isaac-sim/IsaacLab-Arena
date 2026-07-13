# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Submit one typed Arena Experiment to OSMO with an optional policy server.

Example:

    python -m osmo.submit_arena_experiment_workflow \\
        --experiment_config path/to/experiment.yaml \\
        --server_config path/to/policy_server.yaml \\
        --osmo_config osmo/config/workflow.yaml \\
        --dry_run
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from omegaconf import OmegaConf

from osmo.workflows.arena_experiment_workflow import ArenaExperimentWorkflow, Pi0ArenaExperimentWorkflow
from osmo.workflows.workflow import WorkflowCfg

POLICY_SERVER_WORKFLOWS = {
    "pi0": Pi0ArenaExperimentWorkflow,
}


def submit_arena_experiment_workflow(
    experiment_config: str | Path,
    server_config: str | Path | None = None,
    osmo_config: str | Path | None = None,
    experiment_overrides: list[str] | None = None,
    dry_run: bool = False,
) -> int:
    """Build and submit one OSMO workflow for a complete Arena Experiment.

    Args:
        experiment_config: Typed Arena Experiment YAML to stage and evaluate.
        server_config: Optional policy server definition YAML. When omitted, no server task is created.
        osmo_config: Optional generic OSMO workflow configuration YAML.
        experiment_overrides: Hydra overrides applied to the staged Experiment.
        dry_run: Whether to render the OSMO workflow without submitting it.

    Returns:
        The OSMO submission process status.
    """
    overrides = list(experiment_overrides or [])
    workflow_cfg = WorkflowCfg()
    if osmo_config is not None:
        workflow_cfg = OmegaConf.to_object(
            OmegaConf.merge(OmegaConf.structured(WorkflowCfg), OmegaConf.load(Path(osmo_config).expanduser()))
        )
        assert isinstance(workflow_cfg, WorkflowCfg)
    if dry_run:
        workflow_cfg = replace(workflow_cfg, dry_run=True)

    if server_config is None:
        workflow = ArenaExperimentWorkflow(
            workflow_cfg=workflow_cfg,
            experiment_config_path=experiment_config,
            experiment_overrides=overrides,
        )
    else:
        server_config_values = OmegaConf.load(Path(server_config).expanduser())
        assert OmegaConf.is_dict(server_config_values), "Policy server config must be a mapping"
        server_type = server_config_values.pop("type", None)
        assert (
            isinstance(server_type, str) and server_type
        ), "Policy server config must define a non-empty string 'type'"
        available_server_types = ", ".join(sorted(POLICY_SERVER_WORKFLOWS)) or "(none)"
        assert (
            server_type in POLICY_SERVER_WORKFLOWS
        ), f"Unknown policy server type '{server_type}'. Available types: {available_server_types}"
        workflow_cls = POLICY_SERVER_WORKFLOWS[server_type]
        server_task_cfg_type = workflow_cls.server_task_cfg_type
        server_task_cfg = OmegaConf.to_object(
            OmegaConf.merge(OmegaConf.structured(server_task_cfg_type), server_config_values)
        )
        assert isinstance(server_task_cfg, server_task_cfg_type)
        workflow = workflow_cls(
            workflow_cfg=workflow_cfg,
            experiment_config_path=experiment_config,
            server_task_cfg=server_task_cfg,
            experiment_overrides=overrides,
        )
    return workflow.submit_workflow()


def main(cli_args: list[str] | None = None) -> int:
    """Parse submission arguments and submit the Arena Experiment workflow."""
    parser = argparse.ArgumentParser(
        description="Submit one Arena Experiment to OSMO with an optional co-scheduled policy server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
        allow_abbrev=False,
    )
    parser.add_argument("--experiment_config", required=True, help="Typed Arena Experiment YAML")
    parser.add_argument(
        "--server_config",
        default=None,
        help="Policy server definition YAML (omit for an eval-only workflow)",
    )
    parser.add_argument(
        "--osmo_config",
        default=None,
        help="Generic OSMO workflow configuration YAML",
    )
    parser.add_argument("--dry_run", action="store_true", help="Render the OSMO workflow without submitting it")
    args, experiment_overrides = parser.parse_known_args(cli_args)
    unknown_flags = [argument for argument in experiment_overrides if argument.startswith("-")]
    if unknown_flags:
        parser.error(f"Unrecognized arguments: {' '.join(unknown_flags)}")
    return submit_arena_experiment_workflow(
        experiment_config=args.experiment_config,
        server_config=args.server_config,
        osmo_config=args.osmo_config,
        experiment_overrides=experiment_overrides,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

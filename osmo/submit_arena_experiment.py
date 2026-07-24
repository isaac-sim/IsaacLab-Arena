# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Submit typed Arena Experiments as OSMO workflows."""

from __future__ import annotations

import argparse
import sys
import yaml
from dataclasses import dataclass, field
from pathlib import Path

from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from isaaclab_arena.evaluation.arena_experiment import ArenaExperimentCfg
from isaaclab_arena.evaluation.arena_experiment_config_loader import load_arena_experiment_from_config_file
from isaaclab_arena.hydra.typed_experiment_serializer import serialize_arena_experiment_to_yaml
from isaaclab_arena.utils.hydra_overrides import assert_hydra_overrides
from osmo.tasks.experiment_runner_task import ExperimentRunnerTaskCfg
from osmo.workflows.arena_experiment_workflow import ArenaExperimentWorkflow
from osmo.workflows.server_bindings import REMOTE_POLICY_SERVERS, ServerBinding, ServersCfg
from osmo.workflows.workflow import WorkflowCfg

SUBMISSION_CONFIG_NAME = "osmo_arena_experiment_submission"


@dataclass
class ArenaExperimentSubmissionCfg:
    """Combine an Experiment Definition with its OSMO execution settings."""

    experiment_cfg: ArenaExperimentCfg
    """Evaluation semantics executed by ``experiment_runner.py``."""

    servers: ServersCfg = field(default_factory=ServersCfg)
    """Per-server-type deployment config; the server(s) launched are derived from the Runs."""

    osmo: WorkflowCfg = field(default_factory=WorkflowCfg)
    """OSMO scheduling, resource, and timeout configuration."""

    experiment_runner: ExperimentRunnerTaskCfg = field(default_factory=ExperimentRunnerTaskCfg)
    """Configuration for the task that executes ``experiment_runner.py``."""


def submit_arena_experiment(submission_cfg: ArenaExperimentSubmissionCfg) -> int:
    """Build and submit the OSMO workflow described by ``submission_cfg``.

    Args:
        submission_cfg: Composed Experiment, servers, and OSMO configuration.

    Returns:
        The OSMO submission process status.
    """
    workflow = ArenaExperimentWorkflow(
        workflow_cfg=submission_cfg.osmo,
        experiment_cfg=submission_cfg.experiment_cfg,
        servers=submission_cfg.servers,
        task_cfg=submission_cfg.experiment_runner,
    )
    return workflow.submit_workflow().returncode


def _used_server_bindings(experiment_cfg: ArenaExperimentCfg) -> set[ServerBinding]:
    """Return the server bindings the Experiment needs, derived from each Run's client policy."""
    return {
        REMOTE_POLICY_SERVERS[type(run_cfg.policy)]
        for run_cfg in experiment_cfg.runs.values()
        if type(run_cfg.policy) in REMOTE_POLICY_SERVERS
    }


def _osmo_cfg_for_experiment(experiment_cfg: ArenaExperimentCfg) -> WorkflowCfg:
    """Derive the workflow resource from the Experiment's servers (they must share one pool)."""
    required_resources = {(binding.pool, binding.platform) for binding in _used_server_bindings(experiment_cfg)}
    assert len(required_resources) <= 1, (
        f"Experiment needs servers on different resources {sorted(required_resources)}; a submission runs on a"
        " single pool. Run the incompatible policies as separate submissions."
    )
    if not required_resources:
        return WorkflowCfg()
    ((pool, platform),) = required_resources
    return WorkflowCfg(pool=pool, platform=platform)


def build_arena_experiment_submission_cfg(
    experiment_cfg_path: str | Path,
    overrides: list[str] | None = None,
) -> ArenaExperimentSubmissionCfg:
    """Load an Experiment, derive its servers' resource, and apply typed overrides.

    Args:
        experiment_cfg_path: Arena Experiment configuration file.
        overrides: Hydra field overrides rooted at the composed submission.

    Returns:
        The fully composed typed submission configuration.
    """
    experiment_cfg_path = Path(experiment_cfg_path).expanduser()
    assert experiment_cfg_path.suffix.lower() in {
        ".yaml",
        ".yml",
    }, f"OSMO Experiment submission requires a typed YAML Experiment Definition; got '{experiment_cfg_path}'"
    experiment_cfg = load_arena_experiment_from_config_file(experiment_cfg_path, device="cuda:0")
    base_submission = ArenaExperimentSubmissionCfg(
        experiment_cfg=experiment_cfg,
        osmo=_osmo_cfg_for_experiment(experiment_cfg),
    )

    # The Experiment file determines the concrete config types. Register that concrete root so
    # Hydra validates every trailing override against it (each servers.<name> keeps its own type).
    ConfigStore.instance().store(name=SUBMISSION_CONFIG_NAME, node=base_submission)
    with initialize(version_base=None, config_path=None):
        composed = compose(config_name=SUBMISSION_CONFIG_NAME, overrides=overrides or [])
    submission_cfg = OmegaConf.to_object(composed)
    assert isinstance(submission_cfg, ArenaExperimentSubmissionCfg)
    return submission_cfg


def format_submission_config(submission_cfg: ArenaExperimentSubmissionCfg) -> str:
    """Render the composed submission as YAML; every leaf is a valid Hydra KEY=VALUE override.

    Only the server types the Experiment actually uses are shown. The experiment section reuses
    the Experiment serializer so graph-YAML environments render in their reload-friendly
    ``type:`` form rather than the internal argparse tokens.
    """
    used_server_names = {binding.name for binding in _used_server_bindings(submission_cfg.experiment_cfg)}
    servers_values = OmegaConf.to_container(
        OmegaConf.structured(submission_cfg.servers), resolve=True, enum_to_str=True
    )
    submission_values = {
        "osmo": OmegaConf.to_container(OmegaConf.structured(submission_cfg.osmo), resolve=True, enum_to_str=True),
        "experiment_runner": OmegaConf.to_container(
            OmegaConf.structured(submission_cfg.experiment_runner), resolve=True, enum_to_str=True
        ),
        "servers": {name: servers_values[name] for name in sorted(used_server_names)},
        "experiment_cfg": yaml.safe_load(serialize_arena_experiment_to_yaml(submission_cfg.experiment_cfg)),
    }
    return yaml.safe_dump(submission_values, sort_keys=False)


def _create_argument_parser() -> argparse.ArgumentParser:
    """Create the path-first submission command-line parser."""
    parser = argparse.ArgumentParser(
        usage="%(prog)s [-h] --experiment_cfg PATH [--dry_run] [--list_overrides] [OVERRIDE ...]",
        description="Submit a typed Arena Experiment as an OSMO workflow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
The policy server for each Run is derived from the Run's client policy in the Experiment;
override a server's deployment with servers.<name>.* (e.g. servers.pi0.policy_config=...).

Example:

  python -m osmo.submit_arena_experiment \
    --experiment_cfg isaaclab_arena_environments/experiment_configs/droid_pnp_srl_openpi_experiment.yaml \
    osmo.workflow_name=my-evaluation \
    experiment_cfg.runs.droid_pnp_srl_openpi_billiard_hall.rollout_limit.num_episodes=4

Hydra override precedence:

  typed defaults < Experiment YAML < CLI overrides
""",
    )
    parser.add_argument(
        "--experiment_cfg",
        dest="experiment_cfg_path",
        required=True,
        type=Path,
        metavar="PATH",
        help="path to a typed Arena Experiment YAML configuration",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help=(
            "render the workflow YAML and print it instead of submitting to OSMO "
            "(shorthand for the 'osmo.dry_run=true' override)"
        ),
    )
    parser.add_argument(
        "--list_overrides",
        action="store_true",
        help="print the composed submission configuration and exit; every leaf is a valid Hydra KEY=VALUE override",
    )
    parser.allow_abbrev = False
    return parser


def main(cli_args: list[str] | None = None) -> int:
    """Load the Experiment, apply overrides, and submit its OSMO workflow."""
    parser = _create_argument_parser()
    args, overrides = parser.parse_known_args(cli_args)
    assert_hydra_overrides(overrides, parser)
    # Translate the flag into the typed override, unless an explicit one is already present
    # (Hydra rejects two overrides for the same key).
    if args.dry_run and not any(override.startswith("osmo.dry_run=") for override in overrides):
        overrides = [*overrides, "osmo.dry_run=true"]
    submission_cfg = build_arena_experiment_submission_cfg(
        experiment_cfg_path=args.experiment_cfg_path,
        overrides=overrides,
    )
    if args.list_overrides:
        print(
            "# Composed OSMO submission configuration. Every leaf is a valid Hydra KEY=VALUE override, e.g.\n"
            "#   osmo.pool=isaac-dev-l40-03  servers.pi0.policy_variant=pi0  "
            "experiment_cfg.runs.<run_name>.rollout_limit.num_episodes=4\n"
        )
        print(format_submission_config(submission_cfg))
        return 0
    return submit_arena_experiment(submission_cfg)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

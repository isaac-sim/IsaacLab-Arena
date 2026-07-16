# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Submit typed Arena Experiments as OSMO workflows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from isaaclab_arena.evaluation.arena_experiment_config_loader import load_arena_experiment_from_config_file
from isaaclab_arena.hydra.config_override_help import print_config_override_help
from isaaclab_arena.utils.hydra_overrides import assert_hydra_overrides
from osmo.arena_experiment_submission import ArenaExperimentSubmissionCfg, submit_arena_experiment
from osmo.tasks.pi0_server_task import Pi0ServerTaskCfg

SUBMISSION_CONFIG_NAME = "osmo_arena_experiment_submission"
POLICY_SERVER_TASK_CFG_BY_NAME = {
    "pi0": Pi0ServerTaskCfg,
}


def _workflow_managed_override_paths(submission_cfg: ArenaExperimentSubmissionCfg) -> set[str]:
    """Return Experiment fields replaced when the OSMO workflow is assembled."""
    managed_paths = set()
    for run_name in submission_cfg.experiment_cfg.runs:
        run_prefix = f"experiment_cfg.runs.{run_name}"
        managed_paths.add(f"{run_prefix}.environment_builder.device")
        managed_paths.update({
            f"{run_prefix}.policy.remote_host",
            f"{run_prefix}.policy.remote_port",
            f"{run_prefix}.policy.ping_timeout",
        })
    return managed_paths


def build_arena_experiment_submission_cfg(
    experiment_cfg_path: str | Path,
    policy_server_name: str,
    overrides: list[str] | None = None,
) -> ArenaExperimentSubmissionCfg:
    """Load an Experiment, select its policy server, and apply typed overrides.

    Args:
        experiment_cfg_path: Arena Experiment configuration file.
        policy_server_name: Built-in policy-server implementation name.
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
    available_names = ", ".join(sorted(POLICY_SERVER_TASK_CFG_BY_NAME))
    assert (
        policy_server_name in POLICY_SERVER_TASK_CFG_BY_NAME
    ), f"Unknown policy server '{policy_server_name}'. Available policy servers: {available_names}"
    policy_server = POLICY_SERVER_TASK_CFG_BY_NAME[policy_server_name]()
    base_submission = ArenaExperimentSubmissionCfg(
        experiment_cfg=experiment_cfg,
        policy_server=policy_server,
    )

    # The Experiment file and policy-server selector determine the concrete config types.
    # Register that concrete root so Hydra validates every trailing override against it.
    ConfigStore.instance().store(name=SUBMISSION_CONFIG_NAME, node=base_submission)
    with initialize(version_base=None, config_path=None):
        composed = compose(config_name=SUBMISSION_CONFIG_NAME, overrides=overrides or [])
    submission_cfg = OmegaConf.to_object(composed)
    assert isinstance(submission_cfg, ArenaExperimentSubmissionCfg)
    return submission_cfg


def _create_argument_parser() -> argparse.ArgumentParser:
    """Create the path-first submission command-line parser."""
    policy_server_choices = ",".join(POLICY_SERVER_TASK_CFG_BY_NAME)
    parser = argparse.ArgumentParser(
        usage=f"%(prog)s [-h] --experiment-cfg PATH --policy-server {{{policy_server_choices}}} [OVERRIDE ...]",
        description="Submit a typed Arena Experiment as an OSMO workflow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
Example:

  python -m osmo.submit_arena_experiment \
    --experiment-cfg isaaclab_arena_environments/experiment_configs/droid_pnp_srl_openpi_experiment.yaml \
    --policy-server pi0 \
    osmo.workflow_name=my-evaluation \
    experiment_cfg.runs.droid_pnp_srl_openpi_billiard_hall.rollout_limit.num_episodes=4

Hydra override precedence:

  typed defaults < Experiment YAML < CLI overrides
""",
    )
    parser.add_argument(
        "--experiment-cfg",
        dest="experiment_cfg_path",
        required=True,
        type=Path,
        metavar="PATH",
        help="path to a typed Arena Experiment YAML configuration",
    )
    parser.add_argument(
        "--policy-server",
        required=True,
        choices=POLICY_SERVER_TASK_CFG_BY_NAME,
        help="co-scheduled policy-server implementation",
    )
    parser.add_argument(
        "--show-overrides",
        action="store_true",
        help="print Hydra override paths and effective values, then exit",
    )
    parser.allow_abbrev = False
    return parser


def main(cli_args: list[str] | None = None) -> int:
    """Load the Experiment, apply overrides, and submit its OSMO workflow."""
    # Argparse resolves the Experiment path and server selector first; they determine
    # the concrete configs Hydra receives for its remaining overrides.
    parser = _create_argument_parser()
    args, overrides = parser.parse_known_args(cli_args)
    assert_hydra_overrides(overrides, parser)
    submission_cfg = build_arena_experiment_submission_cfg(
        experiment_cfg_path=args.experiment_cfg_path,
        policy_server_name=args.policy_server,
        overrides=overrides,
    )
    if args.show_overrides:
        print("Available Hydra overrides for this OSMO submission:\n")
        print_config_override_help(
            submission_cfg,
            excluded_paths=_workflow_managed_override_paths(submission_cfg),
        )
        return 0
    return submit_arena_experiment(submission_cfg)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

r"""Submit a typed Arena Experiment Definition from an explicit YAML path.

The Experiment Definition describes the named Runs evaluated by the Experiment
Runner. An optional policy-server selector adds a built-in server task that OSMO
starts alongside that runner. The Experiment Definition may live anywhere
accessible to the submission process.

Example:

    python -m osmo.submit_arena_experiment \
        --experiment-definition isaaclab_arena_environments/experiment_configs/droid_pnp_srl_openpi_experiment.yaml \
        --policy-server pi0

The policy server is optional. Omit it when the Experiment uses only local
policies or connects to an externally hosted policy server. Hydra applies the
optional trailing field overrides after typed defaults and Experiment file
values. Files referenced by the Experiment must already exist in the runner
image or be remotely accessible; the submitter does not copy referenced files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from isaaclab_arena.evaluation.arena_experiment_config_loader import load_arena_experiment_from_config_file
from isaaclab_arena.utils.hydra_overrides import assert_hydra_overrides
from osmo.arena_experiment_submission import ArenaExperimentSubmissionCfg, submit_arena_experiment
from osmo.tasks.experiment_runner_task import ExperimentRunnerTaskCfg
from osmo.tasks.pi0_server_task import Pi0ServerTaskCfg
from osmo.workflows.workflow import WorkflowCfg
from osmo.workflows.workflow_constants import DATASET_SWIFT_URL

SUBMISSION_CONFIG_NAME = "osmo_arena_experiment_submission"
POLICY_SERVER_TASK_CFG_BY_NAME = {
    "pi0": Pi0ServerTaskCfg,
}


def build_arena_experiment_submission_cfg(
    experiment_definition_path: str | Path,
    policy_server_name: str | None = None,
    overrides: list[str] | None = None,
) -> ArenaExperimentSubmissionCfg:
    """Load an Experiment, select its optional server, and apply typed overrides.

    Args:
        experiment_definition_path: Arena Experiment configuration file.
        policy_server_name: Optional built-in policy-server implementation name.
        overrides: Hydra field overrides rooted at the composed submission.

    Returns:
        The fully composed typed submission configuration.
    """
    experiment_definition_path = Path(experiment_definition_path).expanduser()
    assert experiment_definition_path.suffix.lower() in {
        ".yaml",
        ".yml",
    }, f"OSMO Experiment submission requires a typed YAML Experiment Definition; got '{experiment_definition_path}'"
    experiment_definition = load_arena_experiment_from_config_file(experiment_definition_path, device="cuda:0")
    policy_server = None
    if policy_server_name is not None:
        available_names = ", ".join(sorted(POLICY_SERVER_TASK_CFG_BY_NAME))
        assert (
            policy_server_name in POLICY_SERVER_TASK_CFG_BY_NAME
        ), f"Unknown policy server '{policy_server_name}'. Available policy servers: {available_names}"
        policy_server = POLICY_SERVER_TASK_CFG_BY_NAME[policy_server_name]()
    base_submission = ArenaExperimentSubmissionCfg(
        experiment_definition=experiment_definition,
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
    workflow_defaults = WorkflowCfg()
    experiment_runner_defaults = ExperimentRunnerTaskCfg()
    pi0_server_defaults = Pi0ServerTaskCfg()
    result_url = DATASET_SWIFT_URL.replace("{{workflow_id}}", "WORKFLOW_ID")
    parser = argparse.ArgumentParser(
        usage=(
            f"%(prog)s [-h] --experiment-definition PATH [--policy-server {{{policy_server_choices}}}] [OVERRIDE ...]"
        ),
        description="Submit a typed Arena Experiment as an OSMO workflow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=rf"""
The Experiment Definition must be typed YAML at a filesystem location visible
inside the submitting container or process.

Minimal Pi0 example:

  python -m osmo.submit_arena_experiment \
    --experiment-definition isaaclab_arena_environments/experiment_configs/droid_pnp_srl_openpi_experiment.yaml \
    --policy-server pi0

The optional --policy-server deploys and connects a built-in server
implementation; it does not select the policy evaluated by the Experiment.
With --policy-server pi0, the Experiment YAML must already select
Pi0RemotePolicy with a compatible policy_variant. Override deployment values
with policy_server.<field>=<value>.

Values are resolved in this order:

  typed defaults < Experiment file values < trailing Hydra overrides

Trailing Hydra overrides are optional. Examples:

  osmo.workflow_name=my-evaluation
  osmo.pool=isaac-dev-l40-03
  osmo.dry_run=true
  experiment_definition.runs.my_run.rollout_limit.num_episodes=4

Current defaults, which can be overridden through the same field paths:

  osmo.pool={workflow_defaults.pool}
  osmo.platform={workflow_defaults.platform}
  osmo.memory={workflow_defaults.memory}
  experiment_runner.image={experiment_runner_defaults.image}
  policy_server.image={pi0_server_defaults.image}
  policy_server.client_ping_timeout={pi0_server_defaults.client_ping_timeout}

Referenced model, checkpoint, and config paths are not copied from the
submission container. They must exist in experiment_runner.image or refer to
storage the remote task can access.

Reports and episode results are uploaded to:

  {result_url}

After replacing WORKFLOW_ID with the ID printed by OSMO, download them with:

  mkdir -p results
  osmo data download {result_url} results
""",
    )
    parser.add_argument(
        "--experiment-definition",
        required=True,
        type=Path,
        metavar="PATH",
        help="typed Arena Experiment Definition YAML file",
    )
    parser.add_argument(
        "--policy-server",
        choices=POLICY_SERVER_TASK_CFG_BY_NAME,
        help="optional co-scheduled policy-server implementation",
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
        experiment_definition_path=args.experiment_definition,
        policy_server_name=args.policy_server,
        overrides=overrides,
    )
    return submit_arena_experiment(submission_cfg)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

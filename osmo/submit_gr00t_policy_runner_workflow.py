# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Configure and submit a GR00T policy-runner evaluation OSMO workflow.

Renders a single OSMO group with two tasks that talk over the shared group
network: a GR00T inference server (sidecar) and the Arena policy runner (lead)
that connects to it and uploads its evaluation outputs.

Usage examples:

    # Dry run (print rendered YAML without submitting)
    python osmo/submit_gr00t_policy_runner_workflow.py --dry-run

    # Submit to a pool
    python osmo/submit_gr00t_policy_runner_workflow.py --pool isaac-dev-l40s-04 --platform ovx-l40s

    # Evaluate a different environment / config
    python osmo/submit_gr00t_policy_runner_workflow.py \
        --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/droid_manip_gr00t_closedloop_config.yaml \
        --env_graph_spec_yaml isaaclab_arena_environments/robolab/mustard_raisin_box_linked.yaml \
        --pool isaac-dev-l40s-04
        --platform ovx-l40s
"""

from __future__ import annotations

import argparse
import sys

from tasks.gr00t_policy_runner_task import (
    DEFAULT_ENV_GRAPH_SPEC_YAML,
    DEFAULT_POLICY_CONFIG,
    DEFAULT_POLICY_RUNNER_ARGS,
    DEFAULT_POLICY_TYPE,
)
from tasks.gr00t_server_task import DEFAULT_EMBODIMENT_TAG, DEFAULT_MODEL_PATH, DEFAULT_SERVER_PORT
from workflows.gr00t_policy_runner_workflow import Gr00tPolicyRunnerWorkflow
from workflows.utils.workflow_types import WorkflowType


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Configure and submit a GR00T policy-runner evaluation OSMO workflow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    server = parser.add_argument_group("gr00t server")
    server.add_argument("--gr00t_server_image", default=None, help="Override the GR00T server image")
    server.add_argument("--gr00t_model_path", default=DEFAULT_MODEL_PATH, help="Model path served by the GR00T server")
    server.add_argument("--gr00t_embodiment_tag", default=DEFAULT_EMBODIMENT_TAG, help="GR00T server embodiment tag")
    server.add_argument("--server_port", type=int, default=DEFAULT_SERVER_PORT, help="GR00T server port")

    runner = parser.add_argument_group("policy runner")
    runner.add_argument("--policy_runner_image", default=None, help="Override the Arena policy-runner image")
    runner.add_argument("--policy_type", default=DEFAULT_POLICY_TYPE, help="Policy class import path")
    runner.add_argument(
        "--policy_config_yaml_path", default=DEFAULT_POLICY_CONFIG, help="GR00T closed-loop config YAML"
    )
    runner.add_argument(
        "--env_graph_spec_yaml", default=DEFAULT_ENV_GRAPH_SPEC_YAML, help="Arena environment spec YAML"
    )
    runner.add_argument("--env_overrides", default=None, help="Trailing Hydra-style environment overrides")
    runner.add_argument(
        "--remote_host",
        default=None,
        help="GR00T server host (defaults to the {{host:gr00t_server}} OSMO token for the sibling task)",
    )
    runner.add_argument(
        "--policy_runner_args",
        default=DEFAULT_POLICY_RUNNER_ARGS,
        help="Policy-runner arguments before the environment spec",
    )

    resources = parser.add_argument_group("resources")
    resources.add_argument("--cpus", type=int, default=15)
    resources.add_argument("--gpus", type=int, default=1)
    resources.add_argument("--memory", default="64Gi")
    resources.add_argument("--storage", default="200Gi")
    resources.add_argument("--platform", default="ovx-l40")

    timeouts = parser.add_argument_group("timeouts")
    timeouts.add_argument("--exec_timeout", default="1d")
    timeouts.add_argument("--queue_timeout", default="2d")

    workflow = parser.add_argument_group("workflow")
    workflow.add_argument("--workflow_name", default="arena-gr00t-policy-runner", help="OSMO workflow name")
    workflow.add_argument("--pool", default=None, help="Target a specific OSMO compute pool")
    workflow.add_argument("--priority", default="NORMAL", choices=["HIGH", "NORMAL", "LOW"])

    parser.add_argument("--dry-run", action="store_true", help="Render without submitting")

    return parser


def main(cli_args: list[str] | None = None) -> int:
    args = build_parser().parse_args(cli_args)

    workflow = Gr00tPolicyRunnerWorkflow(
        workflow_type=WorkflowType.GR00T_POLICY_RUNNER,
        workflow_args=args,
        task_args=args,
    )
    return workflow.submit_workflow(
        dry_run=args.dry_run,
        pool=args.pool,
        priority=args.priority,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

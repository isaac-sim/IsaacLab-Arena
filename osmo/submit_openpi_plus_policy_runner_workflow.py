# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Configure and submit a dummy openpi-server OSMO workflow.

Runs ``echo "hello world"`` inside the openpi server image to verify the image
is pullable and runnable on OSMO.

Usage examples:

    # Dry run (print rendered YAML without submitting)
    python osmo/submit_openpi_server_workflow.py --dry-run

    # Submit to a pool
    python osmo/submit_openpi_server_workflow.py --pool isaac-dev-l40-03
"""

from __future__ import annotations

import argparse
import sys

from tasks.policy_runner_task import DEFAULT_ARENA_ENV_ARGS

# from workflows.openpi_server_workflow import OpenpiServerWorkflow
from workflows.openpi_plus_policy_runner_workflow import OpenpiPlusPolicyRunnerWorkflow
from workflows.utils.policy_types import PolicyType
from workflows.utils.workflow_types import WorkflowType


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Configure and submit an Isaac Lab Arena evaluation OSMO workflow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    task = parser.add_argument_group("task")
    task.add_argument(
        "--workflow_type",
        default=WorkflowType.POLICY_RUNNER.value,
        choices=[workflow_type.value for workflow_type in WorkflowType],
        help="Workflow command set to run",
    )
    # TYPING THE POLICY TYPE HERE IS A HACK. SHOULD BE DONE IN THE WORKFLOW.
    task.add_argument(
        "--policy_type",
        # default=PolicyType.ZERO_ACTION.value,
        default=PolicyType.PI0_REMOTE.value,
        choices=[policy_type.value for policy_type in PolicyType],
        help="Registered policy type to run",
    )
    task.add_argument(
        "--policy_runner_args",
        default=None,
        help="Additional policy-runner arguments before the Arena environment args",
    )
    task.add_argument(
        "--arena_env_args",
        default=DEFAULT_ARENA_ENV_ARGS,
        help="Arena environment name and env-related arguments",
    )

    resources = parser.add_argument_group("resources")
    resources.add_argument("--cpus", type=int, default=15)
    resources.add_argument("--gpus", type=int, default=1)
    resources.add_argument("--memory", default="128Gi")
    resources.add_argument("--storage", default="200Gi")
    resources.add_argument("--platform", default="ovx-l40")

    timeouts = parser.add_argument_group("timeouts")
    timeouts.add_argument("--exec_timeout", default="1d")
    timeouts.add_argument("--queue_timeout", default="2d")

    workflow = parser.add_argument_group("workflow")
    workflow.add_argument("--workflow_name", default="arena-evaluation", help="OSMO workflow name")
    workflow.add_argument("--pool", default=None, help="Target a specific OSMO compute pool")
    workflow.add_argument("--priority", default="NORMAL", choices=["HIGH", "NORMAL", "LOW"])

    parser.add_argument("--dry-run", action="store_true", help="Render without submitting")

    return parser


def main(cli_args: list[str] | None = None) -> int:
    args = build_parser().parse_args(cli_args)

    workflow = OpenpiPlusPolicyRunnerWorkflow(
        workflow_type=WorkflowType.POLICY_RUNNER,
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

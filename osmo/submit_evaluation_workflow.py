# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Configure and submit an Isaac Lab Arena evaluation OSMO workflow.

Usage examples:

    # Default policy_runner evaluation (zero_action on kitchen_pick_and_place)
    python osmo/submit_evaluation_workflow.py --pool isaac-dev-l40-03

    # Custom policy_runner.py and Arena environment arguments
    python osmo/submit_evaluation_workflow.py \
        --gpus 1 \
        --cpus 15 \
        --memory 64Gi \
        --storage 200Gi \
        --platform ovx-l40 \
        --exec_timeout 1d \
        --queue_timeout 2d \
        --workflow_name arena-evaluation \
        --priority NORMAL \
        --pool isaac-dev-l40-03 \
        --workflow_type policy_runner \
        --policy_type zero_action \
        --policy_runner_args '--num_steps 500 --headless' \
        --arena_env_args 'kitchen_pick_and_place --object cracker_box --embodiment franka_ik'

    # Dry run (print rendered YAML without submitting)
    python osmo/submit_evaluation_workflow.py --pool isaac-dev-l40-03 --dry-run
"""

from __future__ import annotations

import argparse
import sys

from tasks.policy_runner_task import DEFAULT_ARENA_ENV_ARGS
from workflows.policy_runner_workflow import PolicyRunnerWorkflow
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
    task.add_argument(
        "--policy_type",
        default=PolicyType.ZERO_ACTION.value,
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
    resources.add_argument("--memory", default="64Gi")
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
    workflow_type = WorkflowType(args.workflow_type)

    workflow = PolicyRunnerWorkflow(
        workflow_type=workflow_type,
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

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
from workflows.workflow import Workflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Configure and submit an Isaac Lab Arena evaluation OSMO workflow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    task = parser.add_argument_group("task")
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

    Workflow.add_common_arguments(parser)
    return parser


def main(cli_args: list[str] | None = None) -> int:
    args = build_parser().parse_args(cli_args)

    workflow = PolicyRunnerWorkflow(
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

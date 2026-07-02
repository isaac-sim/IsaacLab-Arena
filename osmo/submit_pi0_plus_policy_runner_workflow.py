# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Configure and submit the pi0 policy-runner + server OSMO workflow.

Co-schedules an Isaac Lab Arena policy-runner task with the pi0 inference server it connects to.

Usage examples:

    # Default policy-runner + pi0 server
    python osmo/submit_pi0_plus_policy_runner_workflow.py --pool isaac-dev-l40s-04

    # Custom policy_runner.py and Arena environment arguments
    python osmo/submit_pi0_plus_policy_runner_workflow.py \
        --gpus 2 \
        --cpus 15 \
        --memory 256Gi \
        --storage 200Gi \
        --platform ovx-l40s \
        --exec_timeout 1d \
        --queue_timeout 2d \
        --workflow_name arena-pi0-plus \
        --priority NORMAL \
        --pool isaac-dev-l40s-04 \
        --policy_runner_args '--num_steps 500 --headless' \
        --arena_env_args 'kitchen_pick_and_place --object cracker_box --embodiment franka_ik'

    # Dry run (print rendered YAML without submitting)
    python osmo/submit_pi0_plus_policy_runner_workflow.py --pool isaac-dev-l40s-04 --dry-run
"""

from __future__ import annotations

import sys

from workflows.pi0_plus_policy_runner_workflow import Pi0PlusPolicyRunnerWorkflow


def main(cli_args: list[str] | None = None) -> int:
    parser = Pi0PlusPolicyRunnerWorkflow.build_parser(
        description="Configure and submit the pi0 policy-runner + server OSMO workflow.",
        epilog=__doc__,
    )
    args = parser.parse_args(cli_args)

    workflow = Pi0PlusPolicyRunnerWorkflow(
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

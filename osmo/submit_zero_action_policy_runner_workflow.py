# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys

from workflows.zero_action_policy_runner_workflow import ZeroActionPolicyRunnerWorkflow


def main(cli_args: list[str] | None = None) -> int:
    parser = ZeroActionPolicyRunnerWorkflow.build_parser(
        description="Configure and submit a zero-action policy-runner OSMO workflow.",
        epilog=__doc__,
    )
    args = parser.parse_args(cli_args)

    workflow = ZeroActionPolicyRunnerWorkflow(
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

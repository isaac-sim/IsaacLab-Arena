# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys

from workflows.openpi_plus_policy_runner_workflow import OpenpiPlusPolicyRunnerWorkflow


def main(cli_args: list[str] | None = None) -> int:
    parser = OpenpiPlusPolicyRunnerWorkflow.build_parser(
        description="Configure and submit the openpi policy-runner + server OSMO workflow.",
        epilog=__doc__,
    )
    args = parser.parse_args(cli_args)

    workflow = OpenpiPlusPolicyRunnerWorkflow(
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

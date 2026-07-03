# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Configure and submit an Isaac Lab Arena policy-runner OSMO workflow.

Select the policy with ``--policy``:

  * ``zero_action`` -- a single policy-runner task running the built-in zero-action policy.
  * ``pi0``         -- a policy-runner task co-scheduled with the pi0 inference server it queries.

Usage examples:

    # Zero-action policy runner
    python osmo/submit_evaluation_workflow.py \
        --policy zero_action \
        --arena_env_args 'kitchen_pick_and_place --object cracker_box --embodiment franka_ik'

    # pi0 policy runner + server
    python osmo/submit_evaluation_workflow.py \
        --policy pi0 \
        --arena_env_args 'kitchen_pick_and_place --object cracker_box --embodiment franka_ik'

    # Dry run (print rendered YAML without submitting)
    python osmo/submit_evaluation_workflow.py --policy zero_action --arena_env_args '...' --dry-run
"""

from __future__ import annotations

import argparse
import sys

from workflows.pi0_plus_policy_runner_workflow import Pi0PlusPolicyRunnerWorkflow
from workflows.workflow import Workflow
from workflows.zero_action_policy_runner_workflow import ZeroActionPolicyRunnerWorkflow

POLICIES: dict[str, type[Workflow]] = {
    "zero_action": ZeroActionPolicyRunnerWorkflow,
    "pi0": Pi0PlusPolicyRunnerWorkflow,
}


def main(cli_args: list[str] | None = None) -> int:
    # Resolve --policy first so we can build the parser for the selected workflow's tasks.
    policy_parser = argparse.ArgumentParser(add_help=False)
    policy_parser.add_argument("--policy", choices=POLICIES, required=True)
    policy_args, _ = policy_parser.parse_known_args(cli_args)
    workflow_cls = POLICIES[policy_args.policy]

    parser = workflow_cls.build_parser(
        description="Configure and submit an Isaac Lab Arena policy-runner OSMO workflow.",
        epilog=__doc__,
    )
    parser.add_argument("--policy", choices=POLICIES, required=True, help="Which policy-runner workflow to submit")
    args = parser.parse_args(cli_args)

    workflow = workflow_cls(workflow_args=args, task_args=args)
    return workflow.submit_workflow(dry_run=args.dry_run, pool=args.pool, priority=args.priority)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

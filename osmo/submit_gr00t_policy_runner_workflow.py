# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Configure and submit a GR00T policy-runner evaluation OSMO workflow.

Renders a single OSMO group with two tasks that talk over the shared group
network: a GR00T inference server and the Arena policy runner (lead)
that connects to it and uploads its evaluation outputs.

Usage examples:

    # Dry run (print rendered YAML without submitting)
    python osmo/submit_gr00t_policy_runner_workflow.py \
        --arena_env_args "kitchen_pick_and_place --object cracker_box" --dry-run

    # Submit to a pool
    python osmo/submit_gr00t_policy_runner_workflow.py \
        --arena_env_args "kitchen_pick_and_place --object cracker_box" \
        --pool isaac-dev-l40s-04 --platform ovx-l40s

    # Evaluate a different environment / config
    python osmo/submit_gr00t_policy_runner_workflow.py \
        --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/droid_manip_gr00t_closedloop_config.yaml \
        --arena_env_args "kitchen_pick_and_place --object cracker_box" \
        --pool isaac-dev-l40s-04 \
        --platform ovx-l40s
"""

from __future__ import annotations

import sys

from workflows.gr00t_policy_runner_workflow import Gr00tPolicyRunnerWorkflow


def main(cli_args: list[str] | None = None) -> int:
    parser = Gr00tPolicyRunnerWorkflow.build_parser(
        description="Configure and submit a GR00T policy-runner evaluation OSMO workflow.",
        epilog=__doc__,
    )
    args = parser.parse_args(cli_args)

    workflow = Gr00tPolicyRunnerWorkflow(
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

# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Configure and submit the GR00T policy-runner + server OSMO workflow.

Co-schedules an Isaac Lab Arena policy-runner task with the GR00T inference server it connects to.

Usage examples:

    # Default policy-runner + GR00T server
    python osmo/submit_gr00t_policy_runner_workflow.py --pool isaac-dev-l40s-04

    # Custom policy config and Arena environment arguments
    python osmo/submit_gr00t_policy_runner_workflow.py \
        --gpus 1 \
        --cpus 15 \
        --memory 64Gi \
        --storage 200Gi \
        --platform ovx-l40s \
        --exec_timeout 1d \
        --queue_timeout 2d \
        --workflow_name arena-gr00t \
        --priority NORMAL \
        --pool isaac-dev-l40s-04 \
        --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/droid_manip_gr00t_closedloop_config.yaml \
        --policy_runner_args '--num_episodes 2 --headless --enable_cameras --num_envs 4 --record_camera_video' \
        --arena_env_args 'kitchen_pick_and_place --object cracker_box'

    # Dry run (print rendered YAML without submitting)
    python osmo/submit_gr00t_policy_runner_workflow.py --pool isaac-dev-l40s-04 --dry-run
"""

from __future__ import annotations

import sys

from workflows.gr00t_policy_runner_workflow import Gr00tPolicyRunnerWorkflow


def main(cli_args: list[str] | None = None) -> int:
    parser = Gr00tPolicyRunnerWorkflow.build_parser(
        description="Configure and submit the GR00T policy-runner + server OSMO workflow.",
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

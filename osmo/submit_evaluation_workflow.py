# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Configure and submit an Isaac Lab Arena policy-runner OSMO workflow.

Select the policy with ``--policy``:

  * ``zero_action`` -- a single policy-runner task running the built-in zero-action policy.
  * ``pi0``         -- a policy-runner task co-scheduled with the pi0 inference server it queries.
  * ``gr00t``       -- a policy-runner task co-scheduled with the GR00T inference server it queries.
  * ``dreamzero``   -- a full DreamZero evaluation: submits the inference-server workflow
                       (H100 pool) plus a policy-runner workflow (RTX pool) tunnelled to it;
                       the runner cancels the server workflow when it exits.

Usage examples:

    # Zero-action policy runner
    python -m osmo.submit_evaluation_workflow \
        --policy zero_action \
        --arena_env kitchen_pick_and_place \
        --arena_env_args '--object cracker_box --embodiment franka_ik'

    # pi0 policy runner + server
    python -m osmo.submit_evaluation_workflow \
        --policy pi0 \
        --arena_env kitchen_pick_and_place \
        --arena_env_args '--object cracker_box --embodiment franka_ik'

    # GR00T policy runner + server
    python -m osmo.submit_evaluation_workflow \
        --policy gr00t \
        --arena_env kitchen_pick_and_place

    # DreamZero: server workflow (H100 pool) + tunnelled policy-runner workflow (RTX pool)
    python -m osmo.submit_evaluation_workflow \
        --policy dreamzero \
        --arena_env pick_and_place_maple_table \
        --arena_env_args '--embodiment droid_abs_joint_pos' \
        --policy_runner_args '--num_episodes 5 --enable_cameras --record_camera_video \
            --language_instruction "Pick up the cube and place it in the bowl"'

    # Dry run (print rendered YAML without submitting)
    python -m osmo.submit_evaluation_workflow --policy zero_action --arena_env ... --dry_run
"""

from __future__ import annotations

import argparse
import sys

from isaaclab_arena.cli.dataclass_cli import add_dataclass_cli_args, dataclass_from_cli
from osmo.workflows.dreamzero_split_workflows import DreamZeroEvaluationWorkflow
from osmo.workflows.server_plus_policy_runner_workflow import Gr00tPolicyRunnerWorkflow, Pi0PlusPolicyRunnerWorkflow
from osmo.workflows.workflow import SubmittableWorkflow
from osmo.workflows.zero_action_policy_runner_workflow import ZeroActionPolicyRunnerWorkflow

POLICIES: dict[str, type[SubmittableWorkflow]] = {
    "zero_action": ZeroActionPolicyRunnerWorkflow,
    "pi0": Pi0PlusPolicyRunnerWorkflow,
    "gr00t": Gr00tPolicyRunnerWorkflow,
    "dreamzero": DreamZeroEvaluationWorkflow,
}


def main(cli_args: list[str] | None = None) -> int:
    # Resolve --policy first so we can generate the flags for the selected workflow's configs.
    policy_parser = argparse.ArgumentParser(add_help=False)
    policy_parser.add_argument("--policy", choices=POLICIES, required=True)
    policy_args, _ = policy_parser.parse_known_args(cli_args)
    workflow_cls = POLICIES[policy_args.policy]

    parser = argparse.ArgumentParser(
        description="Configure and submit an Isaac Lab Arena policy-runner OSMO workflow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--policy", choices=POLICIES, required=True, help="Which policy-runner workflow to submit")
    add_dataclass_cli_args(parser, workflow_cls.workflow_cfg_type)
    add_dataclass_cli_args(parser, workflow_cls.task_cfg_type)
    args = parser.parse_args(cli_args)

    workflow_cfg = dataclass_from_cli(workflow_cls.workflow_cfg_type, args)
    task_cfg = dataclass_from_cli(workflow_cls.task_cfg_type, args)
    workflow = workflow_cls(workflow_cfg=workflow_cfg, task_cfg=task_cfg)
    return workflow.submit_workflow().returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

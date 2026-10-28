# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Submit the agentic environment generation time-to-resolved-layout benchmark as an OSMO workflow.

Each case and environment count runs in its own GPU group; a final CPU group publishes the per-configuration results
and a combined ``all_results.json`` to the workflow's output URL.

Usage examples:

    # Full matrix: every case at 1, 16, 64, and 256 environments, 100 samples each
    python -m osmo.submit_time_to_resolved_layout_benchmark

    # Smoke test of one case at one environment count
    python -m osmo.submit_time_to_resolved_layout_benchmark \
        --cases tabletop_banana_plate_distractors_14 --num_envs 1 --num_runs 1 --exec_timeout 2h

    # Benchmark a branch image instead of the image main's CI publishes
    python -m osmo.submit_time_to_resolved_layout_benchmark \
        --image nvcr.io/nvstaging/isaac-amr/isaaclab_arena:<tag>

    # Dry run (print rendered YAML without submitting)
    python -m osmo.submit_time_to_resolved_layout_benchmark --dry_run
"""

from __future__ import annotations

import argparse
import sys

from isaaclab_arena.cli.dataclass_cli import add_dataclass_cli_args, dataclass_from_cli
from osmo.workflows.time_to_resolved_layout_benchmark_workflow import TimeToResolvedLayoutBenchmarkWorkflow


def main(cli_args: list[str] | None = None) -> int:
    workflow_cls = TimeToResolvedLayoutBenchmarkWorkflow
    parser = argparse.ArgumentParser(
        description="Submit the time-to-resolved-layout benchmark as an OSMO workflow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_dataclass_cli_args(parser, workflow_cls.workflow_cfg_type)
    add_dataclass_cli_args(parser, workflow_cls.task_cfg_type)
    args = parser.parse_args(cli_args)

    workflow_cfg = dataclass_from_cli(workflow_cls.workflow_cfg_type, args)
    task_cfg = dataclass_from_cli(workflow_cls.task_cfg_type, args)
    workflow = workflow_cls(workflow_cfg=workflow_cfg, task_cfg=task_cfg)
    return workflow.submit_workflow().returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

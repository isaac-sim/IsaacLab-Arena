# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Submit the agentic environment generation time-to-first-spec benchmark as an OSMO workflow.

One CPU task runs every benchmark case and publishes the per-case results, the generated specs, and a combined
``all_results.json`` to the workflow's output URL. The task reads the inference API keys from the
``nvidia-inference-api`` OSMO credential, which needs ``NVIDIA_API_KEY`` and/or ``NV_API_KEY`` entries.

Usage examples:

    # Every case, 100 requests each, on the default endpoint
    python -m osmo.submit_time_to_first_spec_benchmark

    # Smoke test: one request per case
    python -m osmo.submit_time_to_first_spec_benchmark --num_runs 1 --exec_timeout 2h

    # Benchmark a branch image on the internal endpoint
    python -m osmo.submit_time_to_first_spec_benchmark \
        --image nvcr.io/nvstaging/isaac-amr/isaaclab_arena:<tag> --inference_endpoint internal

    # Dry run (print rendered YAML without submitting)
    python -m osmo.submit_time_to_first_spec_benchmark --dry_run
"""

from __future__ import annotations

import argparse
import sys

from isaaclab_arena.cli.dataclass_cli import add_dataclass_cli_args, dataclass_from_cli
from osmo.workflows.time_to_first_spec_benchmark_workflow import TimeToFirstSpecBenchmarkWorkflow


def main(cli_args: list[str] | None = None) -> int:
    workflow_cls = TimeToFirstSpecBenchmarkWorkflow
    parser = argparse.ArgumentParser(
        description="Submit the time-to-first-spec benchmark as an OSMO workflow.",
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

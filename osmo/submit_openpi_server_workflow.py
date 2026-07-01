# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Configure and submit a dummy openpi-server OSMO workflow.

Runs ``echo "hello world"`` inside the openpi server image to verify the image
is pullable and runnable on OSMO.

Usage examples:

    # Dry run (print rendered YAML without submitting)
    python osmo/submit_openpi_server_workflow.py --dry-run

    # Submit to a pool
    python osmo/submit_openpi_server_workflow.py --pool isaac-dev-l40-03
"""

from __future__ import annotations

import argparse
import sys

from workflows.openpi_server_workflow import OpenpiServerWorkflow
from workflows.workflow import Workflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Configure and submit a dummy openpi-server OSMO workflow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    Workflow.add_common_arguments(parser)
    return parser


def main(cli_args: list[str] | None = None) -> int:
    args = build_parser().parse_args(cli_args)

    workflow = OpenpiServerWorkflow(
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

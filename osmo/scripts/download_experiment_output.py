# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Download one Arena Experiment output from OSMO object storage."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

from osmo.workflows.utils.workflow_id import is_valid_workflow_id
from osmo.workflows.workflow_constants import DATASETS_SWIFT_URL

DEFAULT_OUTPUT_BASE_DIRECTORY = Path("arena_experiment_outputs")


def _workflow_id_argument(value: str) -> str:
    """Parse a workflow ID that is safe to use as a remote and local path component."""
    if not is_valid_workflow_id(value) or value in {".", ".."}:
        raise argparse.ArgumentTypeError(f"invalid OSMO workflow ID: {value!r}")
    return value


def download_experiment_output(workflow_id: str, output_directory: Path) -> int:
    """Download one complete Experiment output and return the OSMO process status.

    Args:
        workflow_id: OSMO workflow ID naming the published Experiment output.
        output_directory: Exact local destination for the Experiment output.

    Returns:
        The ``osmo data download`` process status.
    """
    assert is_valid_workflow_id(workflow_id) and workflow_id not in {
        ".",
        "..",
    }, f"Invalid OSMO workflow ID: {workflow_id!r}"
    output_directory = output_directory.expanduser()
    output_directory.mkdir(parents=True, exist_ok=True)
    assert not any(output_directory.iterdir()), f"Experiment output directory must be empty: '{output_directory}'"
    remote_uri = f"{DATASETS_SWIFT_URL}/{workflow_id}/"
    command = ["osmo", "data", "download", remote_uri, output_directory.as_posix()]
    print(f"$ {shlex.join(command)}", flush=True)
    result = subprocess.run(command)
    if result.returncode == 0:
        print(f"Experiment output downloaded to '{output_directory}'.")
        print(f"Open '{output_directory / 'index.html'}' to view the report.")
    return result.returncode


def _create_argument_parser() -> argparse.ArgumentParser:
    """Create the Experiment-output download command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Download one complete Arena Experiment output, including its report, per-Run results, JSONL outcomes, "
            "and videos."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  python3 -m osmo.scripts.download_experiment_output arena-experiment-123
  python3 -m osmo.scripts.download_experiment_output arena-experiment-123 --output-directory ./my-output
""",
    )
    parser.add_argument(
        "workflow_id",
        type=_workflow_id_argument,
        help="OSMO workflow ID printed by the Arena Experiment submission command",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="exact local destination (default: arena_experiment_outputs/<workflow-id>)",
    )
    parser.allow_abbrev = False
    return parser


def main(cli_args: list[str] | None = None) -> int:
    """Download the Experiment output described on the command line."""
    parsed_arguments = _create_argument_parser().parse_args(cli_args)
    output_directory = parsed_arguments.output_directory
    if output_directory is None:
        output_directory = DEFAULT_OUTPUT_BASE_DIRECTORY / parsed_arguments.workflow_id
    return download_experiment_output(parsed_arguments.workflow_id, output_directory)


if __name__ == "__main__":
    raise SystemExit(main())

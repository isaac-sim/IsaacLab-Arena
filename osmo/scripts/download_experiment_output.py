# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Download one Arena Experiment output from OSMO object storage."""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from osmo.workflows.utils.workflow_id import is_valid_workflow_id
from osmo.workflows.workflow_constants import DATASETS_SWIFT_URL


def _is_safe_workflow_id(value: str) -> bool:
    """Return whether a workflow ID is safe to use as a remote and local path component."""
    return is_valid_workflow_id(value) and value not in {".", ".."}


def _workflow_id_argument(value: str) -> str:
    """Parse a workflow ID that is safe to use as a remote and local path component."""
    if not _is_safe_workflow_id(value):
        raise argparse.ArgumentTypeError(f"invalid OSMO workflow ID: {value!r}")
    return value


def _parse_listed_output_paths(list_output: str, remote_uri: str) -> list[PurePosixPath]:
    """Return safe workflow-relative object paths from ``osmo data list`` output."""
    remote_path_parts = tuple(part for part in urlsplit(remote_uri).path.split("/") if part)
    if not remote_path_parts:
        raise ValueError(f"Remote URI has no object path: {remote_uri!r}")

    object_keys = []
    listed_object_count = None
    for line in list_output.splitlines():
        object_key = line.strip()
        if not object_key:
            continue
        count_match = re.fullmatch(r"Total (\d+) objects? found", object_key)
        if count_match:
            if listed_object_count is not None:
                raise ValueError("OSMO object listing contains multiple totals")
            listed_object_count = int(count_match.group(1))
        else:
            object_keys.append(object_key)
    if listed_object_count is None:
        raise ValueError("OSMO object listing has no total")
    if listed_object_count != len(object_keys):
        raise ValueError(f"OSMO listed {len(object_keys)} object keys but reported {listed_object_count}")

    relative_paths: set[PurePosixPath] = set()
    workflow_prefix: tuple[str, ...] | None = None
    for object_key in object_keys:
        object_parts = object_key.split("/")
        if any(part in {"", ".", ".."} for part in object_parts):
            raise ValueError(f"Unsafe remote object path: {object_key!r}")
        matching_prefix_lengths = [
            prefix_length
            for prefix_length in range(1, min(len(remote_path_parts), len(object_parts)) + 1)
            if tuple(object_parts[:prefix_length]) == remote_path_parts[-prefix_length:]
        ]
        if not matching_prefix_lengths:
            raise ValueError(f"Remote object is outside {remote_uri!r}: {object_key!r}")
        prefix_length = max(matching_prefix_lengths)
        listed_workflow_prefix = tuple(object_parts[:prefix_length])
        if workflow_prefix is None:
            workflow_prefix = listed_workflow_prefix
        if listed_workflow_prefix != workflow_prefix:
            raise ValueError(f"Inconsistent remote object prefix: {object_key!r}")
        relative_parts = object_parts[prefix_length:]
        if not relative_parts:
            raise ValueError(f"Remote listing contains the workflow prefix as an object: {object_key!r}")
        relative_path = PurePosixPath(*relative_parts)
        if relative_path in relative_paths:
            raise ValueError(f"Duplicate remote object path: {object_key!r}")
        relative_paths.add(relative_path)
    return sorted(relative_paths, key=lambda path: path.as_posix())


def _local_output_paths(output_directory: Path) -> set[PurePosixPath]:
    """Return file paths below an Experiment output directory."""
    return {
        PurePosixPath(path.relative_to(output_directory).as_posix())
        for path in output_directory.rglob("*")
        if path.is_file()
    }


def download_experiment_output(workflow_id: str, output_directory: Path, remote_base_uri: str) -> int:
    """Download one complete Experiment output and verify every listed object.

    Args:
        workflow_id: OSMO workflow ID naming the published Experiment output.
        output_directory: Exact local destination for the Experiment output.
        remote_base_uri: Object-storage base URI containing workflow outputs.

    Returns:
        Zero for a complete output, otherwise the failing OSMO or validation status.
    """
    assert _is_safe_workflow_id(workflow_id), f"Invalid OSMO workflow ID: {workflow_id!r}"
    output_directory = output_directory.expanduser()
    output_directory.mkdir(parents=True, exist_ok=True)
    assert not any(output_directory.iterdir()), f"Experiment output directory must be empty: '{output_directory}'"
    remote_uri = f"{remote_base_uri.rstrip('/')}/{workflow_id}"
    expected_path_set: set[PurePosixPath] = set()
    # OSMO 6.3.1 can truncate terminal listings, while its file-output mode materializes the complete inventory.
    with tempfile.TemporaryDirectory(prefix="arena-osmo-list-") as inventory_directory:
        for inventory_attempt in range(3):
            inventory_path = Path(inventory_directory) / f"objects-{inventory_attempt}.txt"
            list_command = ["osmo", "data", "list", "--recursive", remote_uri, inventory_path.as_posix()]
            print(f"$ {shlex.join(list_command)}", flush=True)
            list_result = subprocess.run(list_command, capture_output=True, text=True)
            if list_result.returncode != 0:
                if list_result.stdout:
                    print(list_result.stdout, end="")
                if list_result.stderr:
                    print(list_result.stderr, end="", file=sys.stderr)
                return list_result.returncode
            if not inventory_path.is_file():
                print("OSMO reported success without creating the object inventory.", file=sys.stderr)
                return 1
            list_output = f"{inventory_path.read_text(encoding='utf-8')}\n{list_result.stdout}"
            try:
                expected_path_set.update(_parse_listed_output_paths(list_output, remote_uri))
            except ValueError as error:
                print(f"Invalid OSMO object listing: {error}", file=sys.stderr)
                return 1
    expected_paths = sorted(expected_path_set, key=lambda path: path.as_posix())
    if PurePosixPath("index.html") not in expected_paths:
        print("OSMO object listing is incomplete: index.html is missing.", file=sys.stderr)
        return 1

    # OSMO 6.3.1 can omit objects from a directory download while returning success. Preserve the efficient bulk
    # operation, repair omitted objects through exact-object downloads, and verify against the materialized listing.
    download_command = ["osmo", "data", "download", remote_uri, output_directory.as_posix()]
    print(f"$ {shlex.join(download_command)}", flush=True)
    download_result = subprocess.run(download_command)
    if download_result.returncode != 0:
        return download_result.returncode

    downloaded_paths = _local_output_paths(output_directory)
    missing_paths = sorted(set(expected_paths) - downloaded_paths, key=lambda path: path.as_posix())
    if missing_paths:
        print(f"OSMO directory download omitted {len(missing_paths)} object(s); downloading them individually.")
    for relative_path in missing_paths:
        local_path = output_directory.joinpath(*relative_path.parts)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        remote_object_uri = f"{remote_uri}/{relative_path.as_posix()}"
        object_download_command = ["osmo", "data", "download", remote_object_uri, local_path.parent.as_posix()]
        print(f"$ {shlex.join(object_download_command)}", flush=True)
        object_download_result = subprocess.run(object_download_command)
        if object_download_result.returncode != 0:
            return object_download_result.returncode
        if not local_path.is_file():
            print(f"OSMO reported success without downloading '{relative_path.as_posix()}'.", file=sys.stderr)
            return 1

    downloaded_paths = _local_output_paths(output_directory)
    if downloaded_paths != set(expected_paths):
        missing_paths = sorted(set(expected_paths) - downloaded_paths, key=lambda path: path.as_posix())
        unexpected_paths = sorted(downloaded_paths - set(expected_paths), key=lambda path: path.as_posix())
        if missing_paths:
            print(
                f"Experiment output is missing: {', '.join(path.as_posix() for path in missing_paths)}", file=sys.stderr
            )
        if unexpected_paths:
            print(
                "Experiment output contains unexpected files:"
                f" {', '.join(path.as_posix() for path in unexpected_paths)}",
                file=sys.stderr,
            )
        return 1

    print(f"Experiment output downloaded to '{output_directory}'.")
    print(f"Open '{output_directory / 'index.html'}' to view the report.")
    return 0


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
  python3 -m osmo.scripts.download_experiment_output arena-experiment-123 --output-base-directory ./my-output
  python3 -m osmo.scripts.download_experiment_output arena-experiment-123 --remote-base-uri s3://my-bucket/outputs
""",
    )
    parser.add_argument(
        "workflow_id",
        type=_workflow_id_argument,
        help="OSMO workflow ID printed by the Arena Experiment submission command",
    )
    parser.add_argument(
        "--remote-base-uri",
        default=DATASETS_SWIFT_URL,
        help="object-storage base URI containing workflow outputs (default: %(default)s)",
    )
    parser.add_argument(
        "--output-base-directory",
        type=Path,
        default=Path("/eval"),
        help="local base directory (default destination: %(default)s/<workflow-id>)",
    )
    parser.allow_abbrev = False
    return parser


def main(cli_args: list[str] | None = None) -> int:
    """Download the Experiment output described on the command line."""
    parsed_arguments = _create_argument_parser().parse_args(cli_args)
    output_directory = parsed_arguments.output_base_directory / parsed_arguments.workflow_id
    return download_experiment_output(parsed_arguments.workflow_id, output_directory, parsed_arguments.remote_base_uri)


if __name__ == "__main__":
    raise SystemExit(main())

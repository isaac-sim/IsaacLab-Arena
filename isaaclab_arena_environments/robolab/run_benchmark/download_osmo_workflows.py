# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Download the results of a submit_osmo_workflows.py run, grouped by datetime tag.

Given the ``YYYYMMDD-HHMMSS`` datetime tag shared by one submission run, this queries the most
recent OSMO workflows, matches the tag against their run-id, and downloads each match into a
folder laid out as ``<output_dir>/<tag>/<policy>/<task>/``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from submit_osmo_workflows import POLICIES

# Datetime-tag format shared with submit_osmo_workflows._run_id.
RUN_ID_FORMAT = "%Y%m%d-%H%M%S"

# Swift URL prefix for per-workflow outputs; see osmo/workflows/workflow_constants.py.
WORKFLOWS_SWIFT_URL = "swift://pdx.s8k.io/AUTH_team-isaac/isaaclab_arena/workflows"

# A submitted name looks like ``<prefix>-<policy>-<task>-<YYYYMMDD-HHMMSS>-<N>`` where the trailing
# ``-<N>`` is OSMO's per-name dedup counter. Capture the policy, task, and datetime run-id.
_POLICY_ALTERNATION = "|".join(re.escape(policy) for policy in POLICIES)
_NAME_RE = re.compile(rf"-(?P<policy>{_POLICY_ALTERNATION})-(?P<task>.+?)-(?P<run_id>\d{{8}}-\d{{6}})(?:-\d+)?$")


def _datetime_tag(value: str) -> str:
    """Argparse type that accepts only a full ``YYYYMMDD-HHMMSS`` datetime tag."""
    try:
        datetime.strptime(value, RUN_ID_FORMAT)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a full datetime tag in {RUN_ID_FORMAT} format, e.g. 20260706-085745")
    return value


def _parse_workflow_name(name: str) -> tuple[str, str, str] | None:
    """Return ``(policy, task, run_id)`` parsed from a workflow name, or None if it doesn't match."""
    match = _NAME_RE.search(name)
    if match is None:
        return None
    return match.group("policy"), match.group("task"), match.group("run_id")


def _list_workflows(count: int) -> list[dict]:
    """Return the most recently submitted workflows as parsed JSON records."""
    command = ["osmo", "workflow", "list", "--count", str(count), "--order", "desc", "--format-type", "json"]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(f"'{' '.join(command)}' failed with exit code {result.returncode}")
    return json.loads(result.stdout)["workflows"]


def _matches(workflows: list[dict], tag: str) -> list[tuple[str, str, str]]:
    """Return ``(name, policy, task)`` for workflows whose run-id equals the tag."""
    matches = []
    for workflow in workflows:
        name = workflow["name"]
        parsed = _parse_workflow_name(name)
        if parsed is None:
            continue
        policy, task, run_id = parsed
        if run_id == tag:
            matches.append((name, policy, task))
    return matches


def _download(name: str, dest: Path) -> int:
    """Download one workflow's outputs from Swift into ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    command = ["osmo", "data", "download", f"{WORKFLOWS_SWIFT_URL}/{name}", dest.as_posix()]
    print(f"\n$ {' '.join(command)}", flush=True)
    return subprocess.run(command).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", type=_datetime_tag, help="Full datetime tag, e.g. '20260706-085745'")
    parser.add_argument("--count", type=int, default=100, help="Number of recent workflows to search")
    parser.add_argument("--output_dir", type=Path, default=Path("osmo_results"), help="Base directory for downloads")
    parser.add_argument("--dry-run", action="store_true", help="Print matches without downloading")
    args = parser.parse_args()

    workflows = _list_workflows(args.count)
    matches = _matches(workflows, args.tag)
    if not matches:
        print(f"No workflows in the last {args.count} match datetime tag '{args.tag}'.", file=sys.stderr)
        return 1

    print(f"Matched {len(matches)} workflow(s) for tag '{args.tag}':")
    for name, policy, task in matches:
        print(f"  {name}  ->  {args.tag}/{policy}/{task}")

    if args.dry_run:
        return 0

    tag_dir = args.output_dir / args.tag
    for name, policy, task in matches:
        dest = tag_dir / policy / task
        returncode = _download(name, dest)
        if returncode != 0:
            print(f"Download failed for '{name}' (exit code {returncode}).", file=sys.stderr)
            return returncode

    print(f"\nDownloaded {len(matches)} workflow(s) into {tag_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

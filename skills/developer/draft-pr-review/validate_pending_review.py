# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check a pending GitHub review payload against the pull request before sending it.

Usage (on the host, with ``gh`` authenticated):
    python3 -I skills/developer/draft-pr-review/validate_pending_review.py OWNER/REPO PR_NUMBER PAYLOAD.json

Checks that the payload has no ``event`` (so the review stays pending), that ``commit_id`` is the
current PR head, that the review body and every inline comment start with the AI attribution line,
and that every inline comment is anchored on a line inside GitHub's diff. Only read-only ``gh``
commands are run.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field

ATTRIBUTION_PREFIX = "_AI-drafted"
HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")


@dataclass
class DiffHunk:
    """Line numbers a review comment may target within one diff hunk."""

    right_lines: set[int] = field(default_factory=set)
    """New-file line numbers of added and context lines."""

    left_lines: set[int] = field(default_factory=set)
    """Old-file line numbers of deleted and context lines."""


def run_gh(arguments: list[str]) -> str:
    """Run a read-only ``gh`` command.

    Args:
        arguments: Arguments passed to ``gh``.

    Returns:
        The command's standard output.
    """
    return subprocess.run(["gh", *arguments], check=True, capture_output=True, text=True).stdout


def parse_hunks_by_path(diff_text: str) -> dict[str, list[DiffHunk]]:
    """Parse a unified diff into the hunks of each file.

    Args:
        diff_text: Unified diff as printed by ``gh pr diff``.

    Returns:
        Hunks keyed by the path GitHub uses for review comments (the old path for deleted files).
    """
    hunks_by_path: dict[str, list[DiffHunk]] = {}
    old_path = new_path = None
    current_hunk: DiffHunk | None = None
    old_line = new_line = 0
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            current_hunk = None
            continue
        if line.startswith("--- "):
            old_path = None if line == "--- /dev/null" else line[len("--- a/") :]
            continue
        if line.startswith("+++ "):
            new_path = None if line == "+++ /dev/null" else line[len("+++ b/") :]
            continue
        header_match = HUNK_HEADER.match(line)
        if header_match:
            old_line, new_line = int(header_match.group(1)), int(header_match.group(2))
            current_hunk = DiffHunk()
            comment_path = new_path if new_path is not None else old_path
            hunks_by_path.setdefault(comment_path, []).append(current_hunk)
            continue
        if current_hunk is None or line.startswith("\\"):
            continue
        if line.startswith("+"):
            current_hunk.right_lines.add(new_line)
            new_line += 1
        elif line.startswith("-"):
            current_hunk.left_lines.add(old_line)
            old_line += 1
        else:
            current_hunk.right_lines.add(new_line)
            current_hunk.left_lines.add(old_line)
            new_line += 1
            old_line += 1
    return hunks_by_path


def find_anchor_problem(comment: dict, hunks_by_path: dict[str, list[DiffHunk]]) -> str | None:
    """Explain why an inline comment cannot be anchored on the diff.

    Args:
        comment: One entry of the payload's ``comments`` list.
        hunks_by_path: Diff hunks from :func:`parse_hunks_by_path`.

    Returns:
        A description of the problem, or None when the comment can be anchored.
    """
    path = comment.get("path")
    side = comment.get("side", "RIGHT")
    end_line = comment.get("line")
    start_line = comment.get("start_line", end_line)
    start_side = comment.get("start_side", side)
    if path not in hunks_by_path:
        return f"{path} is not part of the PR diff"
    if end_line is None:
        return f"{path}: comment has no 'line'"
    if start_line > end_line:
        return f"{path}: start_line {start_line} is after line {end_line}"
    if start_side != side:
        return f"{path}:{start_line}-{end_line}: start_side and side differ; keep both on one side"
    for hunk in hunks_by_path[path]:
        allowed_lines = hunk.right_lines if side == "RIGHT" else hunk.left_lines
        if start_line in allowed_lines and end_line in allowed_lines:
            return None
    return f"{path}:{start_line}-{end_line} ({side}) is not inside a single diff hunk"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repository", help="OWNER/REPO, e.g. isaac-sim/IsaacLab-Arena")
    parser.add_argument("pull_request_number", type=int)
    parser.add_argument("payload_path", help="JSON payload for POST /pulls/{number}/reviews")
    arguments = parser.parse_args()

    with open(arguments.payload_path) as payload_file:
        payload = json.load(payload_file)
    head_sha = run_gh(
        ["api", f"repos/{arguments.repository}/pulls/{arguments.pull_request_number}", "--jq", ".head.sha"]
    ).strip()
    diff_text = run_gh(["pr", "diff", str(arguments.pull_request_number), "--repo", arguments.repository])
    hunks_by_path = parse_hunks_by_path(diff_text)

    problems: list[str] = []
    if "event" in payload:
        problems.append("payload has 'event'; remove it so the review is created as PENDING")
    if payload.get("commit_id") != head_sha:
        problems.append(f"commit_id {payload.get('commit_id')} is not the PR head {head_sha}")
    if not payload.get("body", "").startswith(ATTRIBUTION_PREFIX):
        problems.append(f"review body does not start with the attribution line ({ATTRIBUTION_PREFIX}...)")
    for index, comment in enumerate(payload.get("comments", [])):
        if not comment.get("body", "").startswith(ATTRIBUTION_PREFIX):
            problems.append(f"comment {index} ({comment.get('path')}) does not start with the attribution line")
        anchor_problem = find_anchor_problem(comment, hunks_by_path)
        if anchor_problem is not None:
            problems.append(f"comment {index}: {anchor_problem}")

    if problems:
        print("NOT READY:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"OK: {len(payload.get('comments', []))} inline comments, pinned to {head_sha[:10]}, no event")
    return 0


if __name__ == "__main__":
    sys.exit(main())

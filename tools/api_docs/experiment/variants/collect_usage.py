# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Extract tokens, tool calls and wall time per run from subagent transcripts.

Usage: ``python collect_usage.py <tasks_dir> <out_csv>``. A run is identified by the prompt file name in the first user
message (for example ``C2-M05.md``); when a run was repeated the latest transcript wins. Tokens are the final context size plus all output tokens (matches the figure in the harness completion notification).
"""

import csv
import datetime
import json
import re
import sys
from pathlib import Path


def parse_time(text: str) -> float:
    return datetime.datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


DEFAULT_PATTERN = r"prompts_v/(C\d)-(\w+)\.md"


def collect(path: Path, pattern: str = DEFAULT_PATTERN, skip_tools: tuple[str, ...] = ()):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    first = rows[0]["message"]["content"]
    first_text = first if isinstance(first, str) else json.dumps(first)
    match = re.search(pattern, first_text)
    if not match:
        return None
    usage_by_message = {}
    tools = 0
    for row in rows:
        message = row.get("message")
        if not isinstance(message, dict):
            continue
        if message.get("usage"):
            usage_by_message[message.get("id") or row["uuid"]] = message["usage"]
        content = message.get("content")
        if isinstance(content, list):
            tools += sum(
                1
                for item in content
                if isinstance(item, dict) and item.get("type") == "tool_use" and item.get("name") not in skip_tools
            )
    usages = list(usage_by_message.values())
    last = usages[-1]
    tokens = (
        last.get("input_tokens", 0)
        + last.get("cache_creation_input_tokens", 0)
        + last.get("cache_read_input_tokens", 0)
        + sum(u.get("output_tokens", 0) for u in usages)
    )
    seconds = parse_time(rows[-1]["timestamp"]) - parse_time(rows[0]["timestamp"])
    return match.group(1).lower(), match.group(2), tokens, tools, round(seconds, 3)


def main(tasks_dir: Path, out_csv: Path, pattern: str = DEFAULT_PATTERN, skip_tools: tuple[str, ...] = ()) -> None:
    latest: dict[tuple[str, str], tuple] = {}
    for path in sorted(tasks_dir.glob("*.output"), key=lambda p: p.stat().st_mtime):
        try:
            result = collect(path, pattern, skip_tools)
        except Exception:  # noqa: BLE001 - unreadable transcript, skip
            continue
        if result:
            latest[(result[0], result[1])] = result  # a repeated run replaces the earlier one
    out = list(latest.values())
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "id", "tokens", "tool_uses", "duration_s"])
        writer.writerows(sorted(out))
    print(len(out), "runs")


if __name__ == "__main__":
    # Optional: a prompt-name regex with two groups (variant, id) and tool names that are not counted.
    main(
        Path(sys.argv[1]),
        Path(sys.argv[2]),
        sys.argv[3] if len(sys.argv) > 3 else DEFAULT_PATTERN,
        tuple(sys.argv[4].split(",")) if len(sys.argv) > 4 else (),
    )

# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Summarize `docker buildx history logs --progress=rawjson` on stdin."""

import json
import re
import sys
from datetime import datetime


def parse_time(value):
    """Parse a BuildKit timestamp, including its UTC suffix."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def summarize(stream):
    """Return a Markdown report of cache imports and Dockerfile steps."""
    vertices = {}
    skipped_records = 0
    for line in stream:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            skipped_records += 1
            continue
        if not isinstance(record, dict):
            skipped_records += 1
            continue
        # Use BuildKit's JSON field spelling.
        for vertex in record.get("vertexes", []):  # codespell:ignore vertexes
            previous = vertices.setdefault(vertex["digest"], {})
            # Cache lookup, download, and extraction can reuse the same digest.
            for field, boundary in (("started", min), ("completed", max)):
                times = [v[field] for v in (previous, vertex) if v.get(field)]
                if times:
                    vertex[field] = boundary(times, key=parse_time)
            previous.update(vertex)

    def status(vertex):
        if vertex.get("error"):
            return "FAILED"
        if vertex.get("cached"):
            return "CACHED"
        return "RAN" if vertex.get("completed") else "INCOMPLETE"

    def seconds(vertex):
        if not vertex.get("started") or not vertex.get("completed"):
            return "—"
        elapsed = parse_time(vertex["completed"]) - parse_time(vertex["started"])
        return f"{elapsed.total_seconds():.1f}"

    imports = [v for v in vertices.values() if v["name"].startswith("importing cache manifest from ")]
    steps = [v for v in vertices.values() if re.match(r"\[[^]]+ \d+/\d+\] (RUN|COPY|ADD|WORKDIR)\b", v["name"])]
    lines = ["### Docker cache usage", ""]
    if skipped_records:
        lines += [
            f"Warning: skipped {skipped_records} non-JSON or non-object records; this report may be incomplete.",
            "See build-cache.jsonl for the original output.",
            "",
        ]
    for vertex in imports:
        result = vertex.get("error") or ("imported" if vertex.get("completed") else "incomplete")
        lines.append(f"- `{vertex['name']}`: {result}")
    if not imports:
        lines.append("- No registry cache import was recorded.")
    hits = sum(status(v) == "CACHED" for v in steps)
    lines += [
        "",
        f"**{hits}/{len(steps)} Dockerfile steps CACHED.** Base-image and frontend downloads are excluded.",
        "Cache import success alone does not mean any build step was reused.",
        "",
        "| Step | Result | Elapsed seconds |",
        "| --- | --- | ---: |",
    ]
    for vertex in steps:
        name = vertex["name"].replace("|", "\\|").replace("`", "'")
        lines.append(f"| `{name}` | {status(vertex)} | {seconds(vertex)} |")
    lines += [
        "",
        "Elapsed time spans the first start through the last completion, including fetching cached layers.",
        "Steps can overlap; their durations are not total build time.",
    ]
    if not steps:
        lines += ["", "No Dockerfile steps found; inspect the raw build record before drawing conclusions."]
    return "\n".join(lines)


if __name__ == "__main__":
    print(summarize(sys.stdin))

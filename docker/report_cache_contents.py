# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Report the contents of a CI cache directory and detect whether tests changed it."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

MEGABYTE = 1024 * 1024
# The SDK keeps its credential in the `login/` directory (account.json), outside the cached
# roots. These names are only a safety net: do not rely on this filter if a root ever widens.
IGNORED_NAMES = {"login"}
IGNORED_SUFFIXES = (".lock",)
# Content of these files is hashed because it can change without changing size.
CONTENT_HASHED_NAMES = {"usd_version.json"}


def snapshot_paths(paths):
    """Return file count, total bytes and a content digest for the files under `paths`.

    Missing directories count as empty. The digest covers relative paths, sizes and the
    content of `usd_version.json` files; modification times do not affect it.
    """
    entries = []
    for root in paths:
        root_path = Path(root)
        for directory, _, names in os.walk(root_path):
            for name in names:
                if name in IGNORED_NAMES or name.endswith(IGNORED_SUFFIXES):
                    continue
                path = Path(directory) / name
                if path.is_symlink() or not path.is_file():
                    continue
                entries.append((f"{root_path.name}/{path.relative_to(root_path)}", path))
    entries.sort(key=lambda entry: entry[0])

    digest = hashlib.sha256()
    total_bytes = 0
    for relative, path in entries:
        size = path.stat().st_size
        total_bytes += size
        digest.update(f"{relative}\t{size}\n".encode())
    for relative, path in entries:
        if path.name in CONTENT_HASHED_NAMES:
            digest.update(path.read_bytes())
    return {"files": len(entries), "bytes": total_bytes, "digest": digest.hexdigest()}


def describe_restore(matched_key, primary_key):
    """Return "miss", "exact hit" or "restore-key hit" for a cache restore."""
    if not matched_key:
        return "miss"
    return "exact hit" if matched_key == primary_key else "restore-key hit"


def render_report(name, restore_status, before, after=None):
    """Return a Markdown report of the cache; with `after`, include what the tests changed."""
    lines = [
        f"### Cache contents: {name}",
        "",
        f"- Restore: {restore_status}",
        f"- Files: {before['files']}",
        f"- Size: {before['bytes'] / MEGABYTE:.1f} MB",
    ]
    if after is not None:
        changed = after["digest"] != before["digest"]
        lines += [
            f"- Changed: {'yes' if changed else 'no'}",
            f"- Files added: {after['files'] - before['files']}",
            f"- Bytes added: {after['bytes'] - before['bytes']}",
            f"- Size after: {after['bytes'] / MEGABYTE:.1f} MB",
        ]
    return "\n".join(lines) + "\n"


def emit(report):
    """Print the report and append it to the GitHub step summary when available."""
    print(report)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as stream:
            stream.write(report + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    snapshot = commands.add_parser("snapshot", help="Report restored contents and save the snapshot.")
    snapshot.add_argument("--name", required=True)
    snapshot.add_argument("--matched-key", default="")
    snapshot.add_argument("--primary-key", required=True)
    snapshot.add_argument("--out", required=True)
    compare = commands.add_parser("compare", help="Report what changed since the saved snapshot.")
    compare.add_argument("--name", required=True)
    compare.add_argument("--before", required=True)
    for command in (snapshot, compare):
        command.add_argument("paths", nargs="+", metavar="PATH")
    args = parser.parse_args(argv)

    paths = [os.path.expanduser(path) for path in args.paths]
    current = snapshot_paths(paths)
    if args.command == "snapshot":
        Path(args.out).write_text(json.dumps(current))
        emit(render_report(args.name, describe_restore(args.matched_key, args.primary_key), current))
        return 0

    before = json.loads(Path(args.before).read_text())
    emit(render_report(args.name, "see restore step", before, current))
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a") as stream:
            stream.write(f"changed={'true' if current['digest'] != before['digest'] else 'false'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

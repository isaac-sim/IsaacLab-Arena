# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Auto-fix the start year of the Arena copyright header on newly added Python and YAML files.

A file added in the current commit must carry the current year alone (e.g. ``2026``);
a pasted range such as ``2025-2026`` is rewritten to ``2026``. Existing files and the
end year are left to the ``insert-license`` hook.
"""

import re
import subprocess
import sys
from datetime import date
from pathlib import Path

# Captures the year string of the Arena copyright line, splitting off the literal
# prefix/suffix so a rewrite only touches the year(s).
ARENA_RE = re.compile(r"(Copyright \(c\) )(\d{4}(?:-\d{4})?)(, The Isaac Lab Arena Project Developers)")


def header_years(text: str) -> str | None:
    """Return the Arena copyright header's year string (e.g. '2025' or '2025-2026'), or None if absent."""
    m = ARENA_RE.search(text)
    return m.group(2) if m else None


def fix_header_year(text: str, current_year: str, is_new: bool) -> str | None:
    """Return text with the Arena copyright start year set to current_year, or None if no change is needed.

    Only newly added files are rewritten; an existing file's start year is preserved (its end year is
    the insert-license hook's responsibility). A missing header or an already-correct year yields None.
    """
    years = header_years(text)
    if years is None or not is_new or years == current_year:
        return None
    return ARENA_RE.sub(rf"\g<1>{current_year}\g<3>", text, count=1)


def added_paths(status_output: str) -> set[str]:
    """Return the paths added from scratch, parsed from ``git diff --cached --name-status`` output.

    Each line is ``<status>\\t<path>`` (``R<score>\\t<old>\\t<new>`` for a rename). Only status ``A``
    counts as new; a rename is reported as ``R`` and excluded, so a moved file keeps its original
    start year instead of having it reset to the current year.
    """
    return {line.split("\t")[1] for line in status_output.splitlines() if line.startswith("A\t")}


def staged_status() -> str:
    """Return ``git diff --cached --name-status`` for the staged changes, with renames detected."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--find-renames", "--name-status"],
        capture_output=True,
        text=True,
    )
    return result.stdout


def main(argv: list[str]) -> int:
    current = str(date.today().year)
    new_files = added_paths(staged_status())
    exit_code = 0
    for path in argv:
        file = Path(path)
        try:
            text = file.read_text(encoding="utf-8")
        except OSError:
            continue
        new_text = fix_header_year(text, current, is_new=path in new_files)
        if new_text is None:
            continue
        file.write_text(new_text, encoding="utf-8")
        print(f"{path}: new file copyright year '{header_years(text)}' rewritten to '{current}'")
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the tools/fix_new_file_copyright_year.py pre-commit hook."""

from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

CURRENT_YEAR = str(date.today().year)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "tools" / "fix_new_file_copyright_year.py"
_LICENSE_TEMPLATE = _REPO_ROOT / ".github" / "LICENSE_HEADER.txt"


def _load_hook():
    """Import tools/fix_new_file_copyright_year.py (a standalone script, not an installed module)."""
    spec = importlib.util.spec_from_file_location("fix_new_file_copyright_year", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hook = _load_hook()


def _header(years: str) -> str:
    """Return a minimal Python source string with an Arena copyright header for the given year(s)."""
    return (
        f"# Copyright (c) {years}, The Isaac Lab Arena Project Developers "
        "(https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).\n"
        "# All rights reserved.\n#\n# SPDX-License-Identifier: Apache-2.0\n\nx = 1\n"
    )


def test_header_years_parses_single_and_range() -> None:
    assert hook.header_years(_header(CURRENT_YEAR)) == CURRENT_YEAR
    assert hook.header_years(_header("2020-2025")) == "2020-2025"
    assert hook.header_years("x = 1\n") is None


def test_regex_matches_the_real_license_template() -> None:
    # The other tests build the header by hand; this guards against ARENA_RE drifting away from the
    # canonical .github/LICENSE_HEADER.txt that insert-license actually writes (e.g. an org rename).
    assert hook.header_years(_LICENSE_TEMPLATE.read_text(encoding="utf-8")) is not None


def test_new_file_with_pasted_range_is_rewritten() -> None:
    fixed = hook.fix_header_year(_header(f"2020-{CURRENT_YEAR}"), CURRENT_YEAR, is_new=True)
    assert fixed == _header(CURRENT_YEAR)


def test_new_file_with_current_year_needs_no_change() -> None:
    assert hook.fix_header_year(_header(CURRENT_YEAR), CURRENT_YEAR, is_new=True) is None


def test_existing_file_is_left_alone() -> None:
    # A file already in HEAD keeps its start year even when it differs from the current year;
    # the end year is the insert-license hook's responsibility, not this one's.
    assert hook.fix_header_year(_header(f"2020-{CURRENT_YEAR}"), CURRENT_YEAR, is_new=False) is None


def test_file_without_arena_header_is_ignored() -> None:
    assert hook.fix_header_year("x = 1\n", CURRENT_YEAR, is_new=True) is None


def test_yaml_header_is_also_rewritten() -> None:
    # The header format and git detection are language-agnostic: YAML files carry the same
    # "# Copyright (c) ..." header, so widening the hook to YAML needs no logic change.
    yaml = (
        f"# Copyright (c) 2020-{CURRENT_YEAR}, The Isaac Lab Arena Project Developers "
        "(https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).\n"
        "# All rights reserved.\n#\n# SPDX-License-Identifier: Apache-2.0\n\nkey: value\n"
    )
    fixed = hook.fix_header_year(yaml, CURRENT_YEAR, is_new=True)
    assert fixed is not None
    assert fixed.startswith(f"# Copyright (c) {CURRENT_YEAR},")


def test_added_paths_keeps_adds_and_drops_renames_and_edits() -> None:
    # Lines from `git diff --cached --name-status`: A=added, R<score>=renamed, M=modified.
    # Only true additions are "new"; a rename (the bug this guards against) keeps its start year.
    status = "A\tfresh.py\nA\tconfig.yaml\nR100\told.py\tnew.py\nM\tseed.py\n"
    assert hook.added_paths(status) == {"fresh.py", "config.yaml"}

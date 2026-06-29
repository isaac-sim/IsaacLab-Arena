# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the tools/fix_new_file_copyright_year.py pre-commit hook."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

CURRENT_YEAR = str(date.today().year)
_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "fix_new_file_copyright_year.py"


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


# --- Integration tests for the git-based new-file detection (added_paths / main) ----------------


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _run_hook(repo: Path, *names: str) -> subprocess.CompletedProcess:
    # Run as a subprocess with cwd=repo so the hook's `git` queries resolve against that repo.
    return subprocess.run([sys.executable, str(_SCRIPT), *names], cwd=str(repo), capture_output=True, text=True)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A throwaway git repo with an initial commit so HEAD exists."""
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "seed.py").write_text(_header(CURRENT_YEAR))
    _git(tmp_path, "add", "seed.py")
    _git(tmp_path, "commit", "-qm", "init")
    return tmp_path


def test_staged_new_file_is_rewritten(git_repo: Path) -> None:
    f = git_repo / "fresh.py"
    f.write_text(_header(f"2020-{CURRENT_YEAR}"))
    _git(git_repo, "add", "fresh.py")
    result = _run_hook(git_repo, "fresh.py")
    assert result.returncode == 1
    assert f.read_text() == _header(CURRENT_YEAR)


def test_renamed_file_keeps_its_start_year(git_repo: Path) -> None:
    # Regression: a rename is staged as R (not A), so the moved file must be treated as existing
    # and keep its original start year rather than being reset to the current year.
    old = git_repo / "old.py"
    original = _header(f"2020-{CURRENT_YEAR}")
    old.write_text(original)
    _git(git_repo, "add", "old.py")
    _git(git_repo, "commit", "-qm", "add old")
    _git(git_repo, "mv", "old.py", "new.py")
    result = _run_hook(git_repo, "new.py")
    assert result.returncode == 0
    assert (git_repo / "new.py").read_text() == original

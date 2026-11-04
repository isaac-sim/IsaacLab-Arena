# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check that the tests of every Arena package are run by CI.

The main CI steps run the directories listed in ``testpaths`` in ``pytest.ini``. This test fails when a
package adds a ``tests`` directory without adding it there, so new tests are not silently left out of CI.
"""

import configparser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

EXCLUDED_TEST_DIRS = {
    "isaaclab_arena_gr00t/tests",
    "isaaclab_arena_curobo/tests",
}
"""Test directories deliberately left out of ``testpaths`` because they have their own marker-gated CI step."""


def _read_testpaths() -> list[str]:
    parser = configparser.ConfigParser()
    parser.read(REPO_ROOT / "pytest.ini")
    return parser["pytest"]["testpaths"].split()


def _find_package_test_dirs() -> set[str]:
    test_dirs = set()
    for tests_dir in REPO_ROOT.glob("isaaclab_arena*/tests"):
        if any(tests_dir.glob("test_*.py")):
            test_dirs.add(tests_dir.relative_to(REPO_ROOT).as_posix())
    return test_dirs


def test_every_package_test_dir_is_run_by_ci():
    unlisted = _find_package_test_dirs() - set(_read_testpaths()) - EXCLUDED_TEST_DIRS
    assert not unlisted, (
        f"Test directories not run by CI: {sorted(unlisted)}. Add them to testpaths in pytest.ini, or to "
        "EXCLUDED_TEST_DIRS in this file if they have their own CI step."
    )


def test_testpaths_are_existing_test_dirs():
    for test_dir in _read_testpaths():
        assert test_dir in _find_package_test_dirs(), f"testpaths entry {test_dir} has no test files"


def test_excluded_test_dirs_are_not_in_testpaths():
    duplicated = set(_read_testpaths()) & EXCLUDED_TEST_DIRS
    assert not duplicated, f"{sorted(duplicated)} would run twice: in testpaths and in their own CI step"


def test_isaaclab_arena_tests_run_first():
    # The persistent SimulationApp must be launched before any other test imports pxr.
    assert _read_testpaths()[0] == "isaaclab_arena/tests"

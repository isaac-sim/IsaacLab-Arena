# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from isaaclab_arena.utils.class_path import import_class_from_path


def test_import_class_from_dotted_path():
    assert import_class_from_path("pathlib.Path") is Path


def test_import_class_rejects_a_colon_with_the_dotted_form():
    with pytest.raises(AssertionError, match=r"Class path 'pathlib:Path' uses ':'\. .*'pathlib\.Path'"):
        import_class_from_path("pathlib:Path")


@pytest.mark.parametrize(
    ("class_path", "message"),
    [
        ("Path", "Expected a class path of the form 'module_path.ClassName', got 'Path'"),
        ("pathlib.MissingClass", "pathlib has no MissingClass"),
        ("os.path.join", "'os.path.join' is not a class"),
    ],
)
def test_import_class_rejects_invalid_paths(class_path, message):
    with pytest.raises(AssertionError, match=message):
        import_class_from_path(class_path)


def test_import_class_reports_a_missing_module():
    with pytest.raises(ModuleNotFoundError, match="missing_package"):
        import_class_from_path("missing_package.MissingClass")

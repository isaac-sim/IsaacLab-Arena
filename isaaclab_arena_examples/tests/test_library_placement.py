# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check that library scoring and the catalog agree with the book definitions."""

import pytest

from isaaclab_arena_examples.library.run_library import placement_ok
from isaaclab_arena_examples.library.scene import BOOKS, CATALOG


@pytest.mark.parametrize("book", BOOKS, ids=lambda book: book.name)
def test_book_at_target_passes_only_when_released(book):
    pose = [*book.target, 0.0, 0.0, 0.0, 1.0]
    assert placement_ok(book, pose, [0.0] * 6, released=True)
    assert not placement_ok(book, pose, [0.0] * 6, released=False)
    assert not placement_ok(book, pose, [0.1, 0, 0, 0, 0, 0], released=True)


@pytest.mark.parametrize("book", BOOKS, ids=lambda book: book.name)
def test_catalog_matches_books(book):
    entry = CATALOG[book.catalog_id]
    assert entry["name"] == book.name
    assert entry["shelf_slot"] == book.shelf_slot
    expected = book.source if book.name == "retrieve" else book.target
    assert entry["shelf_position"] == expected


def test_nonfinite_pose_cannot_succeed():
    book = BOOKS[0]
    pose = [*book.target, 0.0, 0.0, 0.0, 1.0]
    pose[0] = float("nan")
    assert not placement_ok(book, pose, [0.0] * 6, released=True)

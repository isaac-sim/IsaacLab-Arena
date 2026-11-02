# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests that the generated relations reference page matches the relation classes."""

from isaaclab_arena.relations.relations_reference import (
    REFERENCE_PATH,
    build_relation_reference_entries,
    format_relations_reference,
)


def test_relations_reference_page_is_up_to_date():
    """The checked-in page equals a fresh generation, so docstring changes reach the docs."""
    expected = format_relations_reference(build_relation_reference_entries())
    assert (
        REFERENCE_PATH.read_text() == expected
    ), f"{REFERENCE_PATH} is out of date. Regenerate it with: python -m isaaclab_arena.relations.relations_reference"


def test_every_registered_relation_documents_its_validation():
    """Each relation names the check that enforces it, or says it has none, and documents its parameters."""
    entries = build_relation_reference_entries()
    assert {entry.kind for entry in entries} >= {"on", "next_to", "align_with", "face_to", "is_anchor"}
    for entry in entries:
        assert entry.validation_check == "none" or entry.validation.startswith(f"the ``{entry.validation_check}``")
        for parameter in entry.parameters:
            assert parameter.description, f"{entry.class_name}.{parameter.name} has no Args description."

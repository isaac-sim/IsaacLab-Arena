# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sim-free tests for resolving clutter members into piles."""

from __future__ import annotations

import pytest

from isaaclab_arena.relations.clutter_groups import assert_group_parameters_agree, get_clutter_groups, is_clutter_member
from isaaclab_arena.relations.relations import ClutteredOn, IsAnchor


class _Asset:
    """Minimal stand-in exposing only what group resolution reads."""

    def __init__(self, name: str):
        self.name = name
        self.relations: list = []

    def add_relation(self, relation) -> None:
        self.relations.append(relation)

    def get_relations(self) -> list:
        return self.relations

    def has_relation(self, relation_type: type) -> bool:
        return any(isinstance(relation, relation_type) for relation in self.relations)


def _table_with(*member_names: str, group: str = "clutter", **kwargs) -> tuple[_Asset, list[_Asset]]:
    table = _Asset("table")
    table.add_relation(IsAnchor())
    members = []
    for name in member_names:
        member = _Asset(name)
        member.add_relation(ClutteredOn(table, group=group, **kwargs))
        members.append(member)
    return table, members


def test_members_are_identified_as_clutter():
    table, members = _table_with("a", "b")
    assert not is_clutter_member(table)
    assert all(is_clutter_member(member) for member in members)


def test_group_collects_members_in_declaration_order():
    table, members = _table_with("a", "b", "c")
    groups = get_clutter_groups([table, *members])
    assert len(groups) == 1
    assert groups[0].support is table
    assert [member.name for member in groups[0].members] == ["a", "b", "c"]


def test_distinct_group_names_on_one_support_are_separate_piles():
    table = _Asset("table")
    left = _Asset("left")
    right = _Asset("right")
    left.add_relation(ClutteredOn(table, group="left_pile"))
    right.add_relation(ClutteredOn(table, group="right_pile"))

    groups = get_clutter_groups([table, left, right])
    assert [group.name for group in groups] == ["left_pile", "right_pile"]
    assert [len(group.members) for group in groups] == [1, 1]


def test_same_group_name_on_different_supports_stays_separate():
    first, first_members = _table_with("a", group="tools")
    second, second_members = _table_with("b", group="tools")

    groups = get_clutter_groups([first, second, *first_members, *second_members])
    assert len(groups) == 2
    assert {group.support for group in groups} == {first, second}


def test_group_order_follows_first_member():
    table = _Asset("table")
    second = _Asset("second")
    first = _Asset("first")
    second.add_relation(ClutteredOn(table, group="b"))
    first.add_relation(ClutteredOn(table, group="a"))

    # 'b' is declared first, so it leads regardless of alphabetical order.
    groups = get_clutter_groups([table, second, first])
    assert [group.name for group in groups] == ["b", "a"]


def test_no_clutter_yields_no_groups():
    table = _Asset("table")
    table.add_relation(IsAnchor())
    assert get_clutter_groups([table]) == []


def test_agreeing_parameters_are_accepted():
    table, members = _table_with("a", "b", spread=0.5, gap_m=0.02)
    (group,) = get_clutter_groups([table, *members])
    assert_group_parameters_agree(group)


def test_conflicting_parameters_are_rejected():
    table = _Asset("table")
    agreeing = _Asset("agreeing")
    conflicting = _Asset("conflicting")
    agreeing.add_relation(ClutteredOn(table, group="tools", spread=0.5))
    conflicting.add_relation(ClutteredOn(table, group="tools", spread=0.9))

    (group,) = get_clutter_groups([table, agreeing, conflicting])
    with pytest.raises(AssertionError, match="conflicting spread"):
        assert_group_parameters_agree(group)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"spread": 0.0}, "spread must be in"),
        ({"spread": 1.5}, "spread must be in"),
        ({"gap_m": -0.01}, "gap_m must be non-negative"),
        ({"clearance_m": -0.01}, "clearance_m must be non-negative"),
        ({"group": ""}, "group must be a non-empty name"),
    ],
)
def test_invalid_parameters_are_rejected(kwargs, message):
    with pytest.raises(AssertionError, match=message):
        ClutteredOn(_Asset("table"), **kwargs)


def test_clutter_may_not_rest_on_clutter():
    table, (lower,) = _table_with("lower")
    upper = _Asset("upper")
    relation = ClutteredOn(lower, group="tools")
    upper.add_relation(relation)

    with pytest.raises(AssertionError, match="itself clutter"):
        relation.validate_placement_configuration(upper, {table, lower, upper})


def test_support_must_participate_in_placement():
    absent = _Asset("absent")
    member = _Asset("member")
    relation = ClutteredOn(absent, group="tools")
    member.add_relation(relation)

    with pytest.raises(AssertionError, match="not part of the placement"):
        relation.validate_placement_configuration(member, {member})

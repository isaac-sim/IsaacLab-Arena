# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sim-free tests for resolving clutter members into piles."""

from __future__ import annotations

import pytest

from isaaclab_arena.relations.clutter_groups import (
    assert_group_parameters_agree,
    assert_relations_do_not_target_clutter,
    get_clutter_groups,
    is_clutter_member,
)
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

    @property
    def is_anchor(self) -> bool:
        return self.has_relation(IsAnchor)

    def get_spatial_relations(self) -> list:
        from isaaclab_arena.relations.relations import Relation, UnaryRelation

        return [r for r in self.relations if isinstance(r, (Relation, UnaryRelation))]


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


def test_member_that_is_also_an_anchor_is_rejected():
    table, (member,) = _table_with("member")
    member.add_relation(IsAnchor())
    relation = member.get_relations()[0]

    with pytest.raises(AssertionError, match="also an anchor"):
        relation.validate_placement_configuration(member, {table, member})


def test_member_with_two_clutter_relations_is_rejected():
    table = _Asset("table")
    member = _Asset("member")
    member.add_relation(ClutteredOn(table, group="a"))
    member.add_relation(ClutteredOn(table, group="b"))

    with pytest.raises(AssertionError, match="belongs to one pile"):
        member.get_relations()[0].validate_placement_configuration(member, {table, member})


def test_member_combining_clutter_with_a_spatial_relation_is_rejected():
    from isaaclab_arena.relations.relations import On

    table = _Asset("table")
    member = _Asset("member")
    member.add_relation(ClutteredOn(table, group="tools"))
    member.add_relation(On(table))

    with pytest.raises(AssertionError, match="also carries"):
        member.get_relations()[0].validate_placement_configuration(member, {table, member})


def test_member_resting_on_itself_is_rejected():
    member = _Asset("member")
    relation = ClutteredOn(member, group="tools")
    member.add_relation(relation)

    with pytest.raises(AssertionError, match="cannot rest on itself"):
        relation.validate_placement_configuration(member, {member})


def test_relation_naming_a_clutter_member_as_parent_is_refused():
    from isaaclab_arena.relations.relations import On

    table, (member,) = _table_with("member")
    bystander = _Asset("bystander")
    bystander.add_relation(On(member))

    with pytest.raises(AssertionError, match="which is a clutter member"):
        assert_relations_do_not_target_clutter([table, member, bystander])


def test_face_to_carried_by_a_clutter_member_is_refused():
    from isaaclab_arena.relations.relations import FaceTo

    table, (member,) = _table_with("member")
    member.add_relation(FaceTo(table))

    with pytest.raises(AssertionError, match="pour cannot honour"):
        assert_relations_do_not_target_clutter([table, member])


def test_a_plain_pour_is_accepted():
    table, members = _table_with("a", "b")
    assert_relations_do_not_target_clutter([table, *members])


def test_relations_between_non_clutter_assets_are_untouched():
    from isaaclab_arena.relations.relations import On

    table, (member,) = _table_with("member")
    other = _Asset("other")
    other.add_relation(On(table))

    assert_relations_do_not_target_clutter([table, member, other])


def test_member_may_carry_a_rotate_marker():
    from isaaclab_arena.relations.relations import RotateAroundSolution

    table, (member,) = _table_with("member")
    member.add_relation(RotateAroundSolution(pitch_rad=1.0))

    assert_relations_do_not_target_clutter([table, member])


def test_member_may_not_carry_two_rotate_markers():
    from isaaclab_arena.relations.relations import RotateAroundSolution

    table, (member,) = _table_with("member")
    member.add_relation(RotateAroundSolution(pitch_rad=1.0))
    member.add_relation(RotateAroundSolution(roll_rad=1.0))

    with pytest.raises(AssertionError, match="2 RotateAroundSolution markers"):
        assert_relations_do_not_target_clutter([table, member])


def test_reachability_marker_is_preserved_on_clutter_target():
    from isaaclab_arena.relations.relations import RequiresReachability

    table, (member,) = _table_with("target")
    member.add_relation(RequiresReachability())
    assert_relations_do_not_target_clutter([table, member])
    assert any(isinstance(relation, RequiresReachability) for relation in member.get_relations())


def _test_spawned_support_physics_including_nested_references(simulation_app):
    from pxr import Usd, UsdGeom, UsdPhysics

    from isaaclab_arena.environments.arena_world_scene_access import prim_geometry_is_fixed

    stage = Usd.Stage.CreateInMemory()
    parent = UsdGeom.Xform.Define(stage, "/bin").GetPrim()
    floor = UsdGeom.Cube.Define(stage, "/bin/floor").GetPrim()
    UsdPhysics.CollisionAPI.Apply(floor)
    assert prim_geometry_is_fixed(floor)  # Static collision geometry needs no rigid-body override.
    body = UsdPhysics.RigidBodyAPI.Apply(parent)
    assert not prim_geometry_is_fixed(floor)  # The dynamic ancestor moves this reference.
    body.CreateKinematicEnabledAttr(True)
    assert prim_geometry_is_fixed(floor)
    link = UsdGeom.Cube.Define(stage, "/bin/moving_link").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(link)
    assert not prim_geometry_is_fixed(parent)  # A fixed root does not make its links fixed.
    assert prim_geometry_is_fixed(floor)

    return True


def test_spawned_support_physics_including_nested_references():
    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    assert run_function_with_persistent_simulation_app(_test_spawned_support_physics_including_nested_references)

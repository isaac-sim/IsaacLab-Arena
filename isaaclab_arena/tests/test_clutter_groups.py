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
    assert_support_can_hold_a_pile,
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


# ------------------------------------------------ support immovability


class _CfgAsset(_Asset):
    """Stand-in that also reports the spawn config the immovability check reads."""

    def __init__(self, name: str, cfg):
        super().__init__(name)
        self._cfg = cfg

    def get_object_cfg(self):
        return self.name, self._cfg


def _rigid_cfg(kinematic: bool):
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.sim.schemas import RigidBodyPropertiesCfg
    from isaaclab.sim.spawners.shapes import CuboidCfg

    return RigidObjectCfg(
        spawn=CuboidCfg(size=(1.0, 1.0, 1.0), rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=kinematic))
    )


def _group_on(support):
    member = _Asset("member")
    member.add_relation(ClutteredOn(support, group="pile"))
    return get_clutter_groups([support, member])[0]


def test_dynamic_rigid_support_is_refused():
    """IsAnchor fixes an asset for the solver; it says nothing about what physics does to it."""
    support = _CfgAsset("crate", _rigid_cfg(kinematic=False))
    support.add_relation(IsAnchor())

    with pytest.raises(AssertionError, match="which physics can move"):
        assert_support_can_hold_a_pile(_group_on(support))


def test_kinematic_rigid_support_is_accepted():
    support = _CfgAsset("crate", _rigid_cfg(kinematic=True))
    support.add_relation(IsAnchor())

    assert_support_can_hold_a_pile(_group_on(support))


def test_base_config_without_the_kinematic_flag_is_refused():
    """A base config is not evidence of no dynamics: Arena backgrounds are BASE yet spawn bodies."""
    from isaaclab.assets import AssetBaseCfg

    support = _CfgAsset("ground", AssetBaseCfg(prim_path="/World/ground"))
    support.add_relation(IsAnchor())

    with pytest.raises(AssertionError, match="which physics can move"):
        assert_support_can_hold_a_pile(_group_on(support))


def test_base_config_carrying_a_kinematic_body_is_accepted():
    """This is the real shape of office_table_background: ObjectType.BASE with rigid_props."""
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.sim.schemas import RigidBodyPropertiesCfg
    from isaaclab.sim.spawners.shapes import CuboidCfg

    cfg = AssetBaseCfg(
        prim_path="/World/table",
        spawn=CuboidCfg(size=(1.0, 1.0, 1.0), rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True)),
    )
    support = _CfgAsset("table", cfg)
    support.add_relation(IsAnchor())

    assert_support_can_hold_a_pile(_group_on(support))


def test_articulated_support_is_refused_even_when_its_root_is_kinematic():
    """Pinning the root says nothing about the links and joints carrying the pile."""
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sim.schemas import RigidBodyPropertiesCfg
    from isaaclab.sim.spawners.from_files import UsdFileCfg

    cfg = ArticulationCfg(
        spawn=UsdFileCfg(usd_path="/tmp/none.usd", rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True))
    )
    support = _CfgAsset("cabinet", cfg)
    support.add_relation(IsAnchor())

    with pytest.raises(AssertionError, match="which physics can move"):
        assert_support_can_hold_a_pile(_group_on(support))


def test_support_that_cannot_report_its_config_is_refused():
    """Fail closed: a support whose physics cannot be inspected is not provably immovable."""
    support = _Asset("mystery")
    support.add_relation(IsAnchor())

    with pytest.raises(AssertionError, match="which physics can move"):
        assert_support_can_hold_a_pile(_group_on(support))


# ------------------------------------------- relations against clutter


def test_relation_naming_a_clutter_member_as_parent_is_refused():
    """Otherwise this fails deep in the solver with a bare KeyError about a missing index."""
    from isaaclab_arena.relations.relations import On

    table, (member,) = _table_with("member")
    bystander = _Asset("bystander")
    bystander.add_relation(On(member))

    with pytest.raises(AssertionError, match="which is a clutter member"):
        assert_relations_do_not_target_clutter([table, member, bystander])


def test_face_to_carried_by_a_clutter_member_is_refused():
    """FaceTo derives from RelationBase, so the spatial-relation check never saw it."""
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
    """The pour reads RotateAroundSolution itself, so it is honoured rather than dropped."""
    from isaaclab_arena.relations.relations import RotateAroundSolution

    table, (member,) = _table_with("member")
    member.add_relation(RotateAroundSolution(pitch_rad=1.0))

    assert_relations_do_not_target_clutter([table, member])

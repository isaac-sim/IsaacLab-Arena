# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for finding rigid bodies and joints in a USD stage."""

from __future__ import annotations

from isaaclab_arena.tests.utils.usd_stages import add_body, add_joint, new_stage


def test_single_rigid_body_is_one_group():
    from isaaclab_arena.utils.usd.physics_structure import get_physics_structure

    stage = new_stage()
    add_body(stage, "body_01")

    structure = get_physics_structure(stage)

    assert len(structure.rigid_body_paths) == 1
    assert structure.is_single_rigid_body


def test_fixed_joint_puts_bodies_in_one_group():
    """Shaped like the disinfectant bottle: a cap fixed to a body cannot move."""
    from isaaclab_arena.utils.usd.physics_structure import get_physics_structure

    stage = new_stage()
    body = add_body(stage, "body_01")
    cup = add_body(stage, "cup_01")
    add_joint(stage, "joint_cup_01", "fixed", body, cup)

    structure = get_physics_structure(stage)

    assert len(structure.rigid_body_paths) == 2
    assert structure.body_groups == ((body, cup),)
    assert structure.moving_joints == ()
    assert structure.is_single_rigid_body


def test_revolute_joints_keep_bodies_apart():
    """Shaped like the cabinet: handles are fixed to doors, but the doors still swing."""
    from isaaclab_arena.utils.usd.physics_structure import get_physics_structure

    stage = new_stage()
    carcass = add_body(stage, "body_01")
    doors = [add_body(stage, f"part_0{index}") for index in (1, 2, 4)]
    handles = [add_body(stage, f"part_0{index}") for index in (3, 5)]
    for index, door in enumerate(doors):
        add_joint(stage, f"joint_door_{index}", "revolute", carcass, door)
    for handle, door in zip(handles, doors[1:]):
        add_joint(stage, f"joint_handle_{handle}", "fixed", door, handle)

    structure = get_physics_structure(stage)

    assert len(structure.rigid_body_paths) == 6
    # The frame, the door with no handle, and two door plus handle pairs.
    assert len(structure.body_groups) == 4
    assert len(structure.moving_joints) == 3
    assert not structure.is_single_rigid_body


def test_disabled_joint_groups_nothing():
    from isaaclab_arena.utils.usd.physics_structure import get_physics_structure

    stage = new_stage()
    body = add_body(stage, "body_01")
    cup = add_body(stage, "cup_01")
    add_joint(stage, "joint_cup_01", "fixed", body, cup, enabled=False)

    structure = get_physics_structure(stage)

    assert structure.joints == ()
    assert len(structure.body_groups) == 2
    assert not structure.is_single_rigid_body


def test_unjoined_bodies_stay_separate():
    from isaaclab_arena.utils.usd.physics_structure import get_physics_structure

    stage = new_stage()
    add_body(stage, "body_01")
    add_body(stage, "body_02")

    structure = get_physics_structure(stage)

    assert len(structure.body_groups) == 2
    assert not structure.is_single_rigid_body


def test_joint_attached_to_world_has_one_body():
    from isaaclab_arena.utils.usd.physics_structure import get_physics_structure

    stage = new_stage()
    body = add_body(stage, "body_01")
    add_joint(stage, "joint_world", "fixed", None, body)

    structure = get_physics_structure(stage)

    assert structure.joints[0].body_paths == (body,)
    assert structure.is_single_rigid_body


def test_stage_without_physics_has_no_bodies():
    from isaaclab_arena.utils.usd.physics_structure import get_physics_structure

    stage = new_stage()

    structure = get_physics_structure(stage)

    assert structure.rigid_body_paths == ()
    assert structure.body_groups == ()
    assert not structure.is_single_rigid_body

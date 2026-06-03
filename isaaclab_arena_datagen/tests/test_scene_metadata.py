# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Pure-logic tests for scene naming / path conventions (no Isaac Sim)."""

from pathlib import Path

from isaaclab_arena_datagen.scene_metadata import (
    ALL_SCENE_METADATA,
    MotionType,
    SceneCategory,
    get_dataset_subpath,
    get_registry_name,
    make_reference_metadata,
    make_single_object_metadata,
)


def test_registry_name_translation_is_unsuffixed():
    meta = make_single_object_metadata("lemon", MotionType.TRANSLATION)
    assert get_registry_name(meta) == "lemon"


def test_registry_name_motion_variants_suffixed():
    rot = make_single_object_metadata("lemon", MotionType.ROTATION)
    trot = make_single_object_metadata("lemon", MotionType.TRANSLATION_ROTATION)
    assert get_registry_name(rot) == "lemon_rotation"
    assert get_registry_name(trot) == "lemon_translation_rotation"


def test_reference_registry_name_and_subpath():
    meta = make_reference_metadata("ball_box_robot", num_objects=3)
    assert get_registry_name(meta) == "ball_box_robot"
    assert get_dataset_subpath(meta) == Path("miscellaneous") / "ball_box_robot"


def test_single_object_subpath_groups_motion_variants_by_base_name():
    base = make_single_object_metadata("lemon", MotionType.TRANSLATION)
    rot = make_single_object_metadata("lemon", MotionType.ROTATION)
    assert get_dataset_subpath(base) == Path("one_object") / "translation" / "lemon"
    assert get_dataset_subpath(rot) == Path("one_object") / "rotation" / "lemon"


def test_catalogue_has_84_scenes_with_unique_names():
    # 30 single-object (10x3) + 30 no-collision (10x3) + 20 collision (10x2) + 4 reference.
    assert len(ALL_SCENE_METADATA) == 84
    names = [get_registry_name(m) for m in ALL_SCENE_METADATA]
    assert len(names) == len(set(names)), "registry names must be unique"


def test_collision_has_no_rotation_only_variant():
    # Two static objects never collide, so the collision regime omits ROTATION.
    collision = [m for m in ALL_SCENE_METADATA if m.category is SceneCategory.PAIR_COLLISION]
    assert collision, "expected some collision scenes"
    assert all(m.motion_type is not MotionType.ROTATION for m in collision)

# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Identifying metadata for datagen scenes.

Each concrete scene class in ``datagen/isaaclab_arena_environments/`` carries
a :class:`SceneMetaData` instance describing its category, motion variant,
and number of objects.  The helpers :func:`get_registry_name` and
:func:`get_dataset_subpath` are the single source of truth for converting
that metadata into the CLI / registry key and the on-disk dataset
subpath, so name and path conventions cannot drift between sites.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from pathlib import Path


class MotionType(Enum):
    """Kind of motion applied to the dynamic object(s) in a scene.

    The value is used as both the on-disk subdirectory name (under the
    scene category) and the suffix appended to ``base_name`` when
    building the registry key.
    """

    TRANSLATION = "translation"
    ROTATION = "rotation"
    TRANSLATION_ROTATION = "translation_rotation"


class SceneCategory(Enum):
    """Top-level grouping for datagen scenes.

    The value doubles as the on-disk subpath under ``<output_root>/``.
    ``REFERENCE`` covers the legacy scenes kept for ``--single-scene``
    debugging that pre-date the structured one-object / two-object splits.
    """

    SINGLE_OBJECT = "one_object"
    PAIR_NO_COLLISION = "two_objects/no_collision"
    PAIR_COLLISION = "two_objects/collision"
    REFERENCE = "miscellaneous"


@dataclasses.dataclass(frozen=True)
class SceneMetaData:
    """Identifying metadata for a single datagen scene.

    ``base_name`` is the asset (or asset-pair) identifier without any
    motion-variant suffix (e.g. ``"lemon"``, ``"tuna_drill"``,
    ``"ball_box_robot"``).  The full registry key is built from
    ``base_name`` plus the motion-type suffix via
    :func:`get_registry_name`; the on-disk subpath is built from
    ``category`` plus ``motion_type`` plus ``base_name`` via
    :func:`get_dataset_subpath`.

    ``motion_type`` is ``None`` for :attr:`SceneCategory.REFERENCE`
    scenes, which do not have motion-variant siblings.
    """

    base_name: str
    category: SceneCategory
    motion_type: MotionType | None
    num_objects: int


def get_registry_name(meta: SceneMetaData) -> str:
    """Build the CLI / ``DATAGEN_ENVIRONMENTS`` key for a scene.

    The translation motion variant uses ``base_name`` unsuffixed so the
    common case stays terse; rotation / translation_rotation variants
    append the motion-type value as a suffix.  Reference scenes
    (motion_type=None) use ``base_name`` directly.

    Examples: ``"lemon"``, ``"lemon_rotation"``,
    ``"lemon_translation_rotation"``, ``"ball_box_robot"``.

    Args:
        meta: Scene metadata.
    """
    motion_type = meta.motion_type
    if motion_type is None or motion_type is MotionType.TRANSLATION:
        return meta.base_name
    return f"{meta.base_name}_{motion_type.value}"


def get_dataset_subpath(meta: SceneMetaData) -> Path:
    """Build the on-disk subpath (relative to ``<output_root>/``) for a scene.

    All motion variants of a given base scene share the same final
    ``base_name`` directory under their motion-type subdir, so e.g.
    ``one_object/translation/lemon/`` and ``one_object/rotation/lemon/``
    sit side by side.  Reference scenes have no motion-type level.

    Args:
        meta: Scene metadata.
    """
    if meta.category is SceneCategory.REFERENCE:
        return Path(meta.category.value) / meta.base_name
    motion_type = meta.motion_type
    assert motion_type is not None, f"non-REFERENCE scene {meta.base_name!r} must declare a motion_type"
    return Path(meta.category.value) / motion_type.value / meta.base_name


# ---------------------------------------------------------------------------
# Canonical scene catalogue
#
# This is the single source of truth for which scenes exist.  Both the
# environment classes (``datagen/isaaclab_arena_environments/``) and the
# batch driver (``datagen/generate_all_scenes.py``) consume this list, so
# adding a scene means appending its ``base_name`` to one of the tuples
# below and adding the matching ``scene_metadata = make_*_metadata(...)``
# field on the new environment class.
#
# The module deliberately avoids importing anything from
# ``isaaclab_arena`` / ``isaaclab`` so callers (e.g. the batch driver)
# can enumerate scenes without launching Isaac Sim.
# ---------------------------------------------------------------------------


def make_single_object_metadata(base_name: str, motion_type: MotionType = MotionType.TRANSLATION) -> SceneMetaData:
    """Build :class:`SceneMetaData` for a single-object scene."""
    return SceneMetaData(
        base_name=base_name,
        category=SceneCategory.SINGLE_OBJECT,
        motion_type=motion_type,
        num_objects=1,
    )


def make_pair_no_collision_metadata(base_name: str, motion_type: MotionType = MotionType.TRANSLATION) -> SceneMetaData:
    """Build :class:`SceneMetaData` for a non-colliding two-object pair."""
    return SceneMetaData(
        base_name=base_name,
        category=SceneCategory.PAIR_NO_COLLISION,
        motion_type=motion_type,
        num_objects=2,
    )


def make_pair_collision_metadata(base_name: str, motion_type: MotionType = MotionType.TRANSLATION) -> SceneMetaData:
    """Build :class:`SceneMetaData` for a colliding two-object pair."""
    return SceneMetaData(
        base_name=base_name,
        category=SceneCategory.PAIR_COLLISION,
        motion_type=motion_type,
        num_objects=2,
    )


def make_reference_metadata(base_name: str, num_objects: int) -> SceneMetaData:
    """Build :class:`SceneMetaData` for a legacy reference scene."""
    return SceneMetaData(
        base_name=base_name,
        category=SceneCategory.REFERENCE,
        motion_type=None,
        num_objects=num_objects,
    )


SINGLE_OBJECT_BASE_NAMES: tuple[str, ...] = (
    "lemon",
    "drill",
    "mug",
    "spatula",
    "sauce",
    "cheezit",
    "popcorn",
    "pepper",
    "smartphone",
    "keyboard",
)

PAIR_NO_COLLISION_BASE_NAMES: tuple[str, ...] = (
    "lemon_soup",
    "tuna_drill",
    "foam_bowl",
    "ketchup_spatula",
    "pomegranate_sauce",
    "cheezit_mug",
    "popcorn_spoons",
    "pepper_yogurt",
    "smartphone_lychee",
    "mouse_keyboard",
)

PAIR_COLLISION_BASE_NAMES: tuple[str, ...] = (
    "banana_mug",
    "sugar_mustard",
    "can_box",
    "jello_bowl",
    "cube_spam",
    "hammer_lime",
    "pitcher_broccoli",
    "rubiks_lemon",
    "carton_red",
    "coffee_avocado",
)

# Reference (miscellaneous) scenes: (base_name, num_objects). These four
# legacy scenes pre-date the structured one-object / two-object splits and
# have no motion-variant siblings.
REFERENCE_SCENES: tuple[tuple[str, int], ...] = (
    ("ball_and_box", 2),
    ("single_ball", 1),
    ("single_cracker_box", 1),
    ("ball_box_robot", 3),
)


def _build_all_scene_metadata() -> list[SceneMetaData]:
    """Enumerate every scene in canonical generation order.

    Order: single-object (T then R then TR), two-object no-collision (T,
    R, TR), two-object collision (T, TR), reference.  Matches the
    ordering of ``_ENVIRONMENT_CLASSES`` in
    ``isaaclab_arena_datagen.environments``.
    """
    scenes: list[SceneMetaData] = []
    for motion in (MotionType.TRANSLATION, MotionType.ROTATION, MotionType.TRANSLATION_ROTATION):
        for base_name in SINGLE_OBJECT_BASE_NAMES:
            scenes.append(make_single_object_metadata(base_name, motion))
    for motion in (MotionType.TRANSLATION, MotionType.ROTATION, MotionType.TRANSLATION_ROTATION):
        for base_name in PAIR_NO_COLLISION_BASE_NAMES:
            scenes.append(make_pair_no_collision_metadata(base_name, motion))
    # No rotation-only variant for colliding pairs (two static objects never collide).
    for motion in (MotionType.TRANSLATION, MotionType.TRANSLATION_ROTATION):
        for base_name in PAIR_COLLISION_BASE_NAMES:
            scenes.append(make_pair_collision_metadata(base_name, motion))
    for base_name, num_objects in REFERENCE_SCENES:
        scenes.append(make_reference_metadata(base_name, num_objects))
    return scenes


ALL_SCENE_METADATA: list[SceneMetaData] = _build_all_scene_metadata()

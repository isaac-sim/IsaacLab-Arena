# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Datagen environment registry.

To add a new environment, import its class and append it to
``_ENVIRONMENT_CLASSES``.  The ``DATAGEN_ENVIRONMENTS`` dict is built
automatically from each class's ``scene_metadata``; the registry key comes from
:func:`isaaclab_arena_datagen.scene_metadata.get_registry_name` (which is also
the value each class exposes as its ``name`` attribute).

Unlike the core ``isaaclab_arena_environments`` package -- which auto-registers
on import via the ``@register_environment`` decorator -- these datagen scenes
are registered explicitly through :func:`register_datagen_environments`.  Call
it once (the standalone ``run_datagen`` entry point and the policy-rollout
``DatagenCollector`` both do) before building the Arena CLI parser so the
scenes appear as ``example_environment`` subcommands.

Environments are organised into four groups, each identified by a
:class:`~isaaclab_arena_datagen.scene_metadata.SceneCategory`:

- ``single_object_environments`` (one moving object, three motion variants),
- ``no_collision_environments`` (two non-colliding objects, three variants),
- ``collision_environments`` (two colliding objects, two variants),
- ``miscellaneous`` (four legacy reference scenes).
"""

from isaaclab_arena_datagen.environments.collision_environments import (
    BananaMugEnvironment,
    BananaMugTranslationRotationEnvironment,
    CanBoxEnvironment,
    CanBoxTranslationRotationEnvironment,
    CartonRedEnvironment,
    CartonRedTranslationRotationEnvironment,
    CoffeeAvocadoEnvironment,
    CoffeeAvocadoTranslationRotationEnvironment,
    CubeSpamEnvironment,
    CubeSpamTranslationRotationEnvironment,
    HammerLimeEnvironment,
    HammerLimeTranslationRotationEnvironment,
    JelloBowlEnvironment,
    JelloBowlTranslationRotationEnvironment,
    PitcherBroccoliEnvironment,
    PitcherBroccoliTranslationRotationEnvironment,
    RubiksLemonEnvironment,
    RubiksLemonTranslationRotationEnvironment,
    SugarMustardEnvironment,
    SugarMustardTranslationRotationEnvironment,
)
from isaaclab_arena_datagen.environments.miscellaneous import (
    BallAndBoxEnvironment,
    BallBoxRobotEnvironment,
    SingleBallEnvironment,
    SingleCrackerBoxEnvironment,
)
from isaaclab_arena_datagen.environments.no_collision_environments import (
    CheezitMugEnvironment,
    CheezitMugRotationEnvironment,
    CheezitMugTranslationRotationEnvironment,
    FoamBowlEnvironment,
    FoamBowlRotationEnvironment,
    FoamBowlTranslationRotationEnvironment,
    KetchupSpatulaEnvironment,
    KetchupSpatulaRotationEnvironment,
    KetchupSpatulaTranslationRotationEnvironment,
    LemonSoupEnvironment,
    LemonSoupRotationEnvironment,
    LemonSoupTranslationRotationEnvironment,
    MouseKeyboardEnvironment,
    MouseKeyboardRotationEnvironment,
    MouseKeyboardTranslationRotationEnvironment,
    PepperYogurtEnvironment,
    PepperYogurtRotationEnvironment,
    PepperYogurtTranslationRotationEnvironment,
    PomegranateSauceEnvironment,
    PomegranateSauceRotationEnvironment,
    PomegranateSauceTranslationRotationEnvironment,
    PopcornSpoonsEnvironment,
    PopcornSpoonsRotationEnvironment,
    PopcornSpoonsTranslationRotationEnvironment,
    SmartphoneLycheeEnvironment,
    SmartphoneLycheeRotationEnvironment,
    SmartphoneLycheeTranslationRotationEnvironment,
    TunaDrillEnvironment,
    TunaDrillRotationEnvironment,
    TunaDrillTranslationRotationEnvironment,
)
from isaaclab_arena_datagen.environments.single_object_environments import (
    CheezitEnvironment,
    CheezitRotationEnvironment,
    CheezitTranslationRotationEnvironment,
    DrillEnvironment,
    DrillRotationEnvironment,
    DrillTranslationRotationEnvironment,
    KeyboardEnvironment,
    KeyboardRotationEnvironment,
    KeyboardTranslationRotationEnvironment,
    LemonEnvironment,
    LemonRotationEnvironment,
    LemonTranslationRotationEnvironment,
    MugEnvironment,
    MugRotationEnvironment,
    MugTranslationRotationEnvironment,
    PepperEnvironment,
    PepperRotationEnvironment,
    PepperTranslationRotationEnvironment,
    PopcornEnvironment,
    PopcornRotationEnvironment,
    PopcornTranslationRotationEnvironment,
    SauceEnvironment,
    SauceRotationEnvironment,
    SauceTranslationRotationEnvironment,
    SmartphoneEnvironment,
    SmartphoneRotationEnvironment,
    SmartphoneTranslationRotationEnvironment,
    SpatulaEnvironment,
    SpatulaRotationEnvironment,
    SpatulaTranslationRotationEnvironment,
)
from isaaclab_arena_datagen.scene_metadata import get_registry_name
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

_ENVIRONMENT_CLASSES: list[type[ExampleEnvironmentBase]] = [
    # One-object: translation
    LemonEnvironment,
    DrillEnvironment,
    MugEnvironment,
    SpatulaEnvironment,
    SauceEnvironment,
    CheezitEnvironment,
    PopcornEnvironment,
    PepperEnvironment,
    SmartphoneEnvironment,
    KeyboardEnvironment,
    # One-object: rotation
    LemonRotationEnvironment,
    DrillRotationEnvironment,
    MugRotationEnvironment,
    SpatulaRotationEnvironment,
    SauceRotationEnvironment,
    CheezitRotationEnvironment,
    PopcornRotationEnvironment,
    PepperRotationEnvironment,
    SmartphoneRotationEnvironment,
    KeyboardRotationEnvironment,
    # One-object: translation_rotation
    LemonTranslationRotationEnvironment,
    DrillTranslationRotationEnvironment,
    MugTranslationRotationEnvironment,
    SpatulaTranslationRotationEnvironment,
    SauceTranslationRotationEnvironment,
    CheezitTranslationRotationEnvironment,
    PopcornTranslationRotationEnvironment,
    PepperTranslationRotationEnvironment,
    SmartphoneTranslationRotationEnvironment,
    KeyboardTranslationRotationEnvironment,
    # Two-object no-collision: translation
    LemonSoupEnvironment,
    TunaDrillEnvironment,
    FoamBowlEnvironment,
    KetchupSpatulaEnvironment,
    PomegranateSauceEnvironment,
    CheezitMugEnvironment,
    PopcornSpoonsEnvironment,
    PepperYogurtEnvironment,
    SmartphoneLycheeEnvironment,
    MouseKeyboardEnvironment,
    # Two-object no-collision: rotation
    LemonSoupRotationEnvironment,
    TunaDrillRotationEnvironment,
    FoamBowlRotationEnvironment,
    KetchupSpatulaRotationEnvironment,
    PomegranateSauceRotationEnvironment,
    CheezitMugRotationEnvironment,
    PopcornSpoonsRotationEnvironment,
    PepperYogurtRotationEnvironment,
    SmartphoneLycheeRotationEnvironment,
    MouseKeyboardRotationEnvironment,
    # Two-object no-collision: translation_rotation
    LemonSoupTranslationRotationEnvironment,
    TunaDrillTranslationRotationEnvironment,
    FoamBowlTranslationRotationEnvironment,
    KetchupSpatulaTranslationRotationEnvironment,
    PomegranateSauceTranslationRotationEnvironment,
    CheezitMugTranslationRotationEnvironment,
    PopcornSpoonsTranslationRotationEnvironment,
    PepperYogurtTranslationRotationEnvironment,
    SmartphoneLycheeTranslationRotationEnvironment,
    MouseKeyboardTranslationRotationEnvironment,
    # Two-object collision: translation
    BananaMugEnvironment,
    SugarMustardEnvironment,
    CanBoxEnvironment,
    JelloBowlEnvironment,
    CubeSpamEnvironment,
    HammerLimeEnvironment,
    PitcherBroccoliEnvironment,
    RubiksLemonEnvironment,
    CartonRedEnvironment,
    CoffeeAvocadoEnvironment,
    # Two-object collision: translation_rotation
    BananaMugTranslationRotationEnvironment,
    SugarMustardTranslationRotationEnvironment,
    CanBoxTranslationRotationEnvironment,
    JelloBowlTranslationRotationEnvironment,
    CubeSpamTranslationRotationEnvironment,
    HammerLimeTranslationRotationEnvironment,
    PitcherBroccoliTranslationRotationEnvironment,
    RubiksLemonTranslationRotationEnvironment,
    CartonRedTranslationRotationEnvironment,
    CoffeeAvocadoTranslationRotationEnvironment,
    # Miscellaneous reference scenes (BASE_SCENES)
    BallAndBoxEnvironment,
    SingleBallEnvironment,
    SingleCrackerBoxEnvironment,
    BallBoxRobotEnvironment,
]

DATAGEN_ENVIRONMENTS: dict[str, type[ExampleEnvironmentBase]] = {
    get_registry_name(cls.scene_metadata): cls for cls in _ENVIRONMENT_CLASSES
}


def register_datagen_environments() -> None:
    """Register all datagen scenes into the shared Arena ``EnvironmentRegistry``.

    Idempotent: scenes already present (e.g. on a second call) are skipped. Call
    this before :func:`isaaclab_arena_environments.cli.get_isaaclab_arena_environments_cli_parser`
    so the datagen scenes are exposed as ``example_environment`` subcommands and
    resolvable by :func:`isaaclab_arena_environments.cli.get_arena_builder_from_cli`.
    """
    from isaaclab_arena.assets.registries import EnvironmentRegistry

    registry = EnvironmentRegistry()
    for name, cls in DATAGEN_ENVIRONMENTS.items():
        # ensure_loaded=False mirrors the @register_environment decorators: we only
        # need a duplicate-key check, and forcing a full asset load here could
        # re-enter an in-progress import.
        if not registry.is_registered(name, ensure_loaded=False):
            registry.register(cls, name)


__all__ = [
    "DATAGEN_ENVIRONMENTS",
    "register_datagen_environments",
]

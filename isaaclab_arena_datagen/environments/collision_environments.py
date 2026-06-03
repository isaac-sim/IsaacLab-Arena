# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Two-object collision scenes for data generation.

Each concrete class pairs two Arena objects with opposing initial velocities
so they collide near the camera target within ~50 frames.  All scenes
disable gravity and use no background -- only the two dynamic objects and
a light.

This module defines two motion variants per object pair (10 pairs total):

- ``translation`` (registry name = ``base_name``): linear velocity only
  on each object.  These are the original collision scenes.
- ``translation_rotation`` (suffix ``_translation_rotation``): the
  translation scene's linear velocities plus independent angular
  velocities on each object.  Linear paths are unchanged so the
  collision invariant is preserved.

A rotation-only variant is intentionally not provided: with linear
velocity zero, the two objects sit at their start positions and never
collide.

Category, motion variant, and other identifying data are carried by the
``scene_metadata`` :class:`SceneMetaData` class attribute; the registry
name and on-disk subpath are derived from it via
:func:`datagen.scene_metadata.get_registry_name` and
:func:`datagen.scene_metadata.get_dataset_subpath`.

To add a new collision scene, subclass :class:`CollisionSceneBase` at the
bottom of this file (set ``scene_metadata`` plus the asset / velocity /
camera attributes) and register it in ``__init__.py``.

See ``datagen/README.md`` for the generation loop that writes every scene
into the canonical ``/datasets/dynamic_scenes`` layout used by the
multi-scene training scripts.
"""

import argparse
import dataclasses
from typing import Any, ClassVar

from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.tasks.no_task import NoTask
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.velocity import Velocity
from isaaclab_arena_datagen.camera_trajectory import CameraViewTrajectory, Coord3D
from isaaclab_arena_datagen.scene_metadata import (
    MotionType,
    SceneMetaData,
    get_registry_name,
    make_pair_collision_metadata,
)
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

_NEUTRAL_ROTATION_XYZW = (0.0, 0.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class CollisionSceneBase(ExampleEnvironmentBase):
    """Two objects colliding with gravity disabled and no background.

    Subclasses set class-level attributes to configure the scene:
    ``asset_names``, ``initial_positions_xyz``, ``initial_velocities_xyz``,
    ``camera_position_xyz``, ``camera_target_xyz``, and optionally
    ``initial_angular_velocities_xyz`` (defaults to zero, used by the
    ``translation_rotation`` variant).

    Every concrete subclass must also declare a ``scene_metadata``
    :class:`SceneMetaData` instance.  The ``name`` attribute consumed by
    :class:`IsaacLabArenaEnvironment` and the ``DATAGEN_ENVIRONMENTS``
    registry is derived from ``scene_metadata`` automatically in
    ``__init_subclass__``; the translation_rotation variant shares its
    parent's ``base_name`` (set via ``dataclasses.replace``) so both
    variants land in the same on-disk dirname under
    ``two_objects/collision/<motion>/<base_name>/``.
    """

    scene_metadata: ClassVar[SceneMetaData]
    asset_names: tuple[str, str]
    initial_positions_xyz: tuple[Coord3D, Coord3D]
    initial_velocities_xyz: tuple[Coord3D, Coord3D]
    initial_angular_velocities_xyz: tuple[Coord3D, Coord3D] = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    camera_position_xyz: Coord3D
    camera_target_xyz: Coord3D

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "scene_metadata" in vars(cls):
            cls.name = get_registry_name(cls.scene_metadata)

    def get_env(self, _args_cli: argparse.Namespace) -> Any:
        """Build and return the Isaac Lab Arena environment.

        Args:
            _args_cli: Parsed CLI arguments (unused by collision scenes).
        """
        import isaaclab.sim as sim_utils

        objects = []
        for idx in range(2):
            obj = self.asset_registry.get_asset_by_name(self.asset_names[idx])()
            obj.set_initial_pose(
                Pose(
                    position_xyz=self.initial_positions_xyz[idx],
                    rotation_xyzw=_NEUTRAL_ROTATION_XYZW,
                )
            )
            obj.set_initial_velocity(
                Velocity(
                    linear_xyz=self.initial_velocities_xyz[idx],
                    angular_xyz=self.initial_angular_velocities_xyz[idx],
                )
            )
            # IsaacLab leaves `rigid_props=None` on the spawn cfg, so default-
            # construct it before flipping `disable_gravity`.
            if obj.object_cfg.spawn.rigid_props is None:
                obj.object_cfg.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg()
            obj.object_cfg.spawn.rigid_props.disable_gravity = True
            objects.append(obj)

        dome_cfg = sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0)
        light = self.asset_registry.get_asset_by_name("light")(spawner_cfg=dome_cfg)

        distant_cfg = sim_utils.DistantLightCfg(color=(1.0, 1.0, 1.0), intensity=3000.0, angle=0.53)
        distant_light = self.asset_registry.get_asset_by_name("light")(
            instance_name="distant_light",
            prim_path="/World/DistantLight",
            spawner_cfg=distant_cfg,
        )

        scene = Scene(assets=[*objects, light, distant_light])

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=None,
            scene=scene,
            task=NoTask(),
            teleop_device=None,
        )

    @classmethod
    def get_default_cameras(cls, _num_steps: int) -> list[CameraViewTrajectory]:
        """Return a static camera aimed at the collision region.

        Args:
            _num_steps: Number of simulation steps (unused for static cameras).
        """
        return [
            CameraViewTrajectory(
                position=cls.camera_position_xyz,
                target=cls.camera_target_xyz,
                focal_length_mm=24.0,
            ),
        ]

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        """No extra CLI arguments for collision scenes."""


# ---------------------------------------------------------------------------
# Concrete scenes
# ---------------------------------------------------------------------------


class BananaMugEnvironment(CollisionSceneBase):
    """Banana from upper-left and mug from lower-left, meeting at center."""

    scene_metadata = make_pair_collision_metadata("banana_mug")
    asset_names = ("banana_ycb_robolab", "mug_ycb_robolab")
    initial_positions_xyz = ((-0.35, -0.25, 0.55), (-0.15, 0.20, 0.15))
    initial_velocities_xyz = ((0.70, 0.50, -0.50), (0.30, -0.40, 0.30))
    camera_position_xyz = (0.0, -1.5, 0.45)
    camera_target_xyz = (0.0, 0.0, 0.30)


class SugarMustardEnvironment(CollisionSceneBase):
    """Sugar box from front-left and mustard from behind-right, meeting high."""

    scene_metadata = make_pair_collision_metadata("sugar_mustard")
    asset_names = ("sugar_box", "mustard_bottle")
    initial_positions_xyz = ((-0.35, -0.20, 0.55), (-0.10, 0.20, 0.15))
    initial_velocities_xyz = ((0.70, 0.40, -0.40), (0.20, -0.40, 0.40))
    camera_position_xyz = (0.0, -1.6, 0.50)
    camera_target_xyz = (0.0, 0.0, 0.35)


class CanBoxEnvironment(CollisionSceneBase):
    """Tomato can from upper-left and brown box from upper-right, both descending."""

    scene_metadata = make_pair_collision_metadata("can_box")
    asset_names = ("tomato_soup_can", "brown_box")
    initial_positions_xyz = ((-0.35, 0.20, 0.60), (0.15, 0.15, 0.45))
    initial_velocities_xyz = ((0.70, -0.40, -0.60), (-0.30, -0.30, -0.30))
    camera_position_xyz = (0.0, -1.5, 0.50)
    camera_target_xyz = (0.0, 0.0, 0.30)


class JelloBowlEnvironment(CollisionSceneBase):
    """Jello from right-front and bowl from right-back, converging left."""

    scene_metadata = make_pair_collision_metadata("jello_bowl")
    asset_names = ("jello_ycb_robolab", "bowl_ycb_robolab")
    initial_positions_xyz = ((0.25, -0.35, 0.60), (0.15, 0.15, 0.20))
    initial_velocities_xyz = ((-0.50, 0.70, -0.50), (-0.30, -0.30, 0.30))
    camera_position_xyz = (0.0, -1.5, 0.50)
    camera_target_xyz = (0.0, 0.0, 0.35)


class CubeSpamEnvironment(CollisionSceneBase):
    """Cube from front-left and spam from behind-right, meeting low."""

    scene_metadata = make_pair_collision_metadata("cube_spam")
    asset_names = ("dex_cube", "spam_can_ycb_robolab")
    initial_positions_xyz = ((-0.30, -0.35, 0.50), (0.20, -0.15, 0.10))
    initial_velocities_xyz = ((0.60, 0.70, -0.50), (-0.40, 0.30, 0.30))
    camera_position_xyz = (0.0, -1.4, 0.40)
    camera_target_xyz = (0.0, 0.0, 0.25)


class HammerLimeEnvironment(CollisionSceneBase):
    """Hammer from behind-right and lime from far back-left, meeting at frame ~42."""

    scene_metadata = make_pair_collision_metadata("hammer_lime")
    asset_names = ("hammer_handal_robolab", "lime01_fruits_veggies_robolab")
    initial_positions_xyz = ((0.35, 0.25, 0.60), (-0.30, 1.50, 0.95))
    initial_velocities_xyz = ((-0.50, -0.36, -0.36), (0.43, -2.14, -0.86))
    camera_position_xyz = (0.0, -1.6, 0.50)
    camera_target_xyz = (0.0, 0.0, 0.35)


class PitcherBroccoliEnvironment(CollisionSceneBase):
    """Pitcher from upper-left and broccoli from lower-left, meeting right."""

    scene_metadata = make_pair_collision_metadata("pitcher_broccoli")
    asset_names = ("pitcher_ycb_robolab", "broccoli")
    initial_positions_xyz = ((-0.40, -0.20, 0.60), (-0.15, 0.20, 0.15))
    initial_velocities_xyz = ((0.80, 0.40, -0.60), (0.30, -0.40, 0.30))
    camera_position_xyz = (0.0, -1.5, 0.45)
    camera_target_xyz = (0.0, 0.0, 0.30)


class RubiksLemonEnvironment(CollisionSceneBase):
    """Rubik's cube from left-back and lemon from right-back, meeting front."""

    scene_metadata = make_pair_collision_metadata("rubiks_lemon")
    asset_names = ("rubiks_cube_hot3d_robolab", "lemon_01_fruits_veggies_robolab")
    initial_positions_xyz = ((-0.30, 0.25, 0.60), (0.20, 0.20, 0.15))
    initial_velocities_xyz = ((0.60, -0.50, -0.50), (-0.40, -0.40, 0.40))
    camera_position_xyz = (0.0, -1.4, 0.50)
    camera_target_xyz = (0.0, 0.0, 0.35)


class CartonRedEnvironment(CollisionSceneBase):
    """OJ carton from left-back-high and red onion from right-back, meeting center."""

    scene_metadata = make_pair_collision_metadata("carton_red")
    asset_names = ("orange_juice_carton_hope_robolab", "red_onion_fruits_veggies_robolab")
    initial_positions_xyz = ((-0.25, 0.40, 0.60), (0.15, 0.20, 0.15))
    initial_velocities_xyz = ((0.50, -0.80, -0.60), (-0.30, -0.40, 0.30))
    camera_position_xyz = (0.0, -1.5, 0.50)
    camera_target_xyz = (0.0, 0.0, 0.30)


class CoffeeAvocadoEnvironment(CollisionSceneBase):
    """Coffee can from right-front and avocado from right-back, meeting left."""

    scene_metadata = make_pair_collision_metadata("coffee_avocado")
    asset_names = ("coffee_can_ycb_robolab", "avocado01_fruits_veggies_robolab")
    initial_positions_xyz = ((0.35, -0.35, 0.60), (0.15, 0.15, 0.15))
    initial_velocities_xyz = ((-0.70, 0.70, -0.60), (-0.30, -0.30, 0.30))
    camera_position_xyz = (0.0, -1.5, 0.45)
    camera_target_xyz = (0.0, 0.0, 0.30)


# ---------------------------------------------------------------------------
# Translation + rotation variants (linear velocities reused from the matching
# translation scene; angular velocities are independent random samples with
# magnitudes in [1.0, 3.5] rad/s).  Linear paths -- and hence the collision
# behaviour -- are unchanged.  A rotation-only variant is intentionally not
# defined: with linear velocity zero the objects sit at their start
# positions and never collide.
# ---------------------------------------------------------------------------


class BananaMugTranslationRotationEnvironment(BananaMugEnvironment):
    """``banana_mug``'s linear velocities plus independent angular velocities."""

    scene_metadata = dataclasses.replace(
        BananaMugEnvironment.scene_metadata, motion_type=MotionType.TRANSLATION_ROTATION
    )
    initial_angular_velocities_xyz = ((-0.43, -3.22, 1.20), (1.48, 1.05, -0.36))


class SugarMustardTranslationRotationEnvironment(SugarMustardEnvironment):
    """``sugar_mustard``'s linear velocities plus independent angular velocities."""

    scene_metadata = dataclasses.replace(
        SugarMustardEnvironment.scene_metadata, motion_type=MotionType.TRANSLATION_ROTATION
    )
    initial_angular_velocities_xyz = ((0.82, -0.97, -0.74), (-1.59, 0.53, -0.27))


class CanBoxTranslationRotationEnvironment(CanBoxEnvironment):
    """``can_box``'s linear velocities plus independent angular velocities."""

    scene_metadata = dataclasses.replace(CanBoxEnvironment.scene_metadata, motion_type=MotionType.TRANSLATION_ROTATION)
    initial_angular_velocities_xyz = ((0.0, 1.12, 1.78), (0.72, -0.86, 0.11))


class JelloBowlTranslationRotationEnvironment(JelloBowlEnvironment):
    """``jello_bowl``'s linear velocities plus independent angular velocities."""

    scene_metadata = dataclasses.replace(
        JelloBowlEnvironment.scene_metadata, motion_type=MotionType.TRANSLATION_ROTATION
    )
    initial_angular_velocities_xyz = ((2.53, -0.01, 2.30), (0.91, -0.45, 0.99))


class CubeSpamTranslationRotationEnvironment(CubeSpamEnvironment):
    """``cube_spam``'s linear velocities plus independent angular velocities."""

    scene_metadata = dataclasses.replace(
        CubeSpamEnvironment.scene_metadata, motion_type=MotionType.TRANSLATION_ROTATION
    )
    initial_angular_velocities_xyz = ((-1.64, 0.15, -1.15), (3.13, 1.21, -0.84))


class HammerLimeTranslationRotationEnvironment(HammerLimeEnvironment):
    """``hammer_lime``'s linear velocities plus independent angular velocities."""

    scene_metadata = dataclasses.replace(
        HammerLimeEnvironment.scene_metadata, motion_type=MotionType.TRANSLATION_ROTATION
    )
    initial_angular_velocities_xyz = ((-0.17, 1.75, 1.21), (-1.25, 0.66, 3.19))


class PitcherBroccoliTranslationRotationEnvironment(PitcherBroccoliEnvironment):
    """``pitcher_broccoli``'s linear velocities plus independent angular velocities."""

    scene_metadata = dataclasses.replace(
        PitcherBroccoliEnvironment.scene_metadata, motion_type=MotionType.TRANSLATION_ROTATION
    )
    initial_angular_velocities_xyz = ((-1.17, -0.43, 0.61), (-0.25, 0.82, 2.29))


class RubiksLemonTranslationRotationEnvironment(RubiksLemonEnvironment):
    """``rubiks_lemon``'s linear velocities plus independent angular velocities."""

    scene_metadata = dataclasses.replace(
        RubiksLemonEnvironment.scene_metadata, motion_type=MotionType.TRANSLATION_ROTATION
    )
    initial_angular_velocities_xyz = ((-0.96, -0.26, 0.57), (-2.70, -1.58, 0.02))


class CartonRedTranslationRotationEnvironment(CartonRedEnvironment):
    """``carton_red``'s linear velocities plus independent angular velocities."""

    scene_metadata = dataclasses.replace(
        CartonRedEnvironment.scene_metadata, motion_type=MotionType.TRANSLATION_ROTATION
    )
    initial_angular_velocities_xyz = ((0.26, 0.39, 1.11), (1.04, 2.43, 0.51))


class CoffeeAvocadoTranslationRotationEnvironment(CoffeeAvocadoEnvironment):
    """``coffee_avocado``'s linear velocities plus independent angular velocities."""

    scene_metadata = dataclasses.replace(
        CoffeeAvocadoEnvironment.scene_metadata, motion_type=MotionType.TRANSLATION_ROTATION
    )
    initial_angular_velocities_xyz = ((0.19, 2.09, -2.45), (0.06, 2.50, 0.48))

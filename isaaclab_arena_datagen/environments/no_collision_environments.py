# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Two-object non-colliding scenes for data generation.

Each concrete class pairs two Arena objects with initial positions and
velocities chosen so their 3D paths never intersect while their 2D
projections may cross visually in the camera image.  All scenes disable
gravity and use no background -- only the two dynamic objects and a light.

The non-collision invariant is enforced by placing the two objects on
different camera-depth (``y``) planes, separated by at least ~0.25 m at
every simulated step.  Each scene tunes its own camera pose, initial
positions, and velocities so the refinement model sees a diverse set of
viewpoints and motion patterns (horizontal passes, diverging diagonals,
perpendicular paths, X-crossings, and vertical crossings).

This module defines three motion variants per object pair (10 pairs total):

- ``translation`` (registry name = ``base_name``): linear velocity only
  on each object.  These are the original non-collision scenes.
- ``rotation`` (suffix ``_rotation``): linear velocity zero on both
  objects, independent angular velocities.  Both objects spin in place
  at their starting positions, so the non-collision invariant is
  trivially preserved.
- ``translation_rotation`` (suffix ``_translation_rotation``): the
  translation scene's linear velocities plus independent angular
  velocities.  Linear paths are unchanged so the non-collision
  invariant is preserved.

Category, motion variant, and other identifying data are carried by the
``scene_metadata`` :class:`SceneMetaData` class attribute; the registry
name and on-disk subpath are derived from it via
:func:`datagen.scene_metadata.get_registry_name` and
:func:`datagen.scene_metadata.get_dataset_subpath`.

To add a new non-collision scene, subclass :class:`NoCollisionSceneBase`
at the bottom of this file (set ``scene_metadata`` plus the asset /
velocity / camera attributes) and register it in ``__init__.py``.

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
    make_pair_no_collision_metadata,
)
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

_NEUTRAL_ROTATION_XYZW = (0.0, 0.0, 0.0, 1.0)
_DEFAULT_FOCAL_LENGTH_MM = 24.0


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class NoCollisionSceneBase(ExampleEnvironmentBase):
    """Two objects moving on disjoint 3D paths, with gravity disabled and no background.

    Subclasses set class-level attributes to configure the scene:
    ``asset_names``, ``initial_positions_xyz``, ``initial_velocities_xyz``,
    ``camera_position_xyz``, ``camera_target_xyz``, and optionally
    ``initial_angular_velocities_xyz`` (defaults to zero, used by
    ``rotation`` / ``translation_rotation`` variants) and
    ``focal_length_mm`` (defaults to 24 mm).

    Every concrete subclass must also declare a ``scene_metadata``
    :class:`SceneMetaData` instance.  The ``name`` attribute consumed by
    :class:`IsaacLabArenaEnvironment` and the ``DATAGEN_ENVIRONMENTS``
    registry is derived from ``scene_metadata`` automatically in
    ``__init_subclass__``; rotation / translation_rotation variants
    share their parent's ``base_name`` (set via
    ``dataclasses.replace``) so all three variants land in the same
    on-disk dirname under
    ``two_objects/no_collision/<motion>/<base_name>/``.
    """

    scene_metadata: ClassVar[SceneMetaData]
    asset_names: tuple[str, str]
    initial_positions_xyz: tuple[Coord3D, Coord3D]
    initial_velocities_xyz: tuple[Coord3D, Coord3D] = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    initial_angular_velocities_xyz: tuple[Coord3D, Coord3D] = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    camera_position_xyz: Coord3D
    camera_target_xyz: Coord3D
    focal_length_mm: float = _DEFAULT_FOCAL_LENGTH_MM

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "scene_metadata" in vars(cls):
            cls.name = get_registry_name(cls.scene_metadata)

    def get_env(self, _args_cli: argparse.Namespace) -> Any:
        """Build and return the Isaac Lab Arena environment.

        Args:
            _args_cli: Parsed CLI arguments (unused by non-collision scenes).
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
        """Return a static camera aimed at the working volume.

        Args:
            _num_steps: Number of simulation steps (unused for static cameras).
        """
        return [
            CameraViewTrajectory(
                position=cls.camera_position_xyz,
                target=cls.camera_target_xyz,
                focal_length_mm=cls.focal_length_mm,
            ),
        ]

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        """No extra CLI arguments for non-collision scenes."""


# ---------------------------------------------------------------------------
# Concrete scenes
# ---------------------------------------------------------------------------


class LemonSoupEnvironment(NoCollisionSceneBase):
    """Lemon and soup-can crossing horizontally at different depths."""

    scene_metadata = make_pair_no_collision_metadata("lemon_soup")
    asset_names = ("lemon_02_fruits_veggies_robolab", "soup_can_hot3d_robolab")
    initial_positions_xyz = ((-0.28, 0.15, 0.35), (0.28, 0.40, 0.35))
    initial_velocities_xyz = ((0.55, 0.00, 0.00), (-0.55, 0.00, 0.00))
    camera_position_xyz = (0.0, -1.00, 0.40)
    camera_target_xyz = (0.0, 0.0, 0.25)


class TunaDrillEnvironment(NoCollisionSceneBase):
    """Tuna can rising left while drill descends right, elevated camera."""

    scene_metadata = make_pair_no_collision_metadata("tuna_drill")
    asset_names = ("canned_tuna_hope_robolab", "cordless_drill_ycb_robolab")
    initial_positions_xyz = ((0.05, 0.20, 0.50), (-0.05, 0.50, 0.10))
    initial_velocities_xyz = ((-0.50, 0.00, 0.40), (0.50, 0.00, -0.20))
    camera_position_xyz = (0.0, -1.10, 0.55)
    camera_target_xyz = (0.0, 0.0, 0.20)


class FoamBowlEnvironment(NoCollisionSceneBase):
    """Foam roller sliding right while bowl rises, side-angle camera."""

    scene_metadata = make_pair_no_collision_metadata("foam_bowl")
    asset_names = ("foam_roller_hot3d_robolab", "wooden_bowl_hot3d_robolab")
    initial_positions_xyz = ((-0.40, 0.10, 0.35), (0.10, 0.45, 0.00))
    initial_velocities_xyz = ((0.70, 0.00, 0.00), (0.00, 0.00, 0.50))
    camera_position_xyz = (1.00, -1.20, 0.55)
    camera_target_xyz = (0.0, 0.0, 0.30)


class KetchupSpatulaEnvironment(NoCollisionSceneBase):
    """Ketchup descending while spatula ascends, low camera angle."""

    scene_metadata = make_pair_no_collision_metadata("ketchup_spatula")
    asset_names = ("ketchup_bottle_hope_robolab", "spatula_hot3d_robolab")
    initial_positions_xyz = ((-0.30, 0.15, 0.55), (-0.25, 0.50, 0.05))
    initial_velocities_xyz = ((0.45, 0.00, -0.35), (0.45, 0.00, 0.35))
    camera_position_xyz = (0.0, -1.00, 0.20)
    camera_target_xyz = (0.0, 0.0, 0.35)


class PomegranateSauceEnvironment(NoCollisionSceneBase):
    """Pomegranate and tomato-sauce can moving in parallel, off-axis camera."""

    scene_metadata = make_pair_no_collision_metadata("pomegranate_sauce")
    asset_names = ("pomegranate01_fruits_veggies_robolab", "tomato_sauce_can_hot3d_robolab")
    initial_positions_xyz = ((-0.30, 0.15, 0.40), (-0.30, 0.45, 0.20))
    initial_velocities_xyz = ((0.40, 0.00, -0.05), (0.40, 0.00, 0.10))
    camera_position_xyz = (-0.45, -1.10, 0.55)
    camera_target_xyz = (0.05, 0.0, 0.25)


class CheezitMugEnvironment(NoCollisionSceneBase):
    """Cheez-It box and mug forming an X at different depths, wide camera."""

    scene_metadata = make_pair_no_collision_metadata("cheezit_mug")
    asset_names = ("cheez_it_ycb_robolab", "mug_hot3d_robolab")
    initial_positions_xyz = ((-0.35, 0.10, 0.15), (0.35, 0.40, 0.55))
    initial_velocities_xyz = ((0.60, 0.00, 0.45), (-0.60, 0.00, -0.45))
    camera_position_xyz = (0.0, -1.30, 0.50)
    camera_target_xyz = (0.0, 0.0, 0.30)


class PopcornSpoonsEnvironment(NoCollisionSceneBase):
    """Popcorn box falling while spoons rise, tight telephoto view."""

    scene_metadata = make_pair_no_collision_metadata("popcorn_spoons")
    asset_names = ("popcorn_box_hope_robolab", "wooden_spoons_hot3d_robolab")
    initial_positions_xyz = ((-0.20, 0.15, 0.60), (0.20, 0.45, 0.05))
    initial_velocities_xyz = ((0.00, 0.00, -0.50), (0.00, 0.00, 0.55))
    camera_position_xyz = (0.0, -1.00, 0.40)
    camera_target_xyz = (0.0, 0.0, 0.30)
    focal_length_mm = 28.0


class PepperYogurtEnvironment(NoCollisionSceneBase):
    """Pepper and yogurt diverging from near-centre, slightly offset camera."""

    scene_metadata = make_pair_no_collision_metadata("pepper_yogurt")
    asset_names = ("red_bell_pepper_objaverse_robolab", "yogurt_cup_hope_robolab")
    initial_positions_xyz = ((0.00, 0.15, 0.35), (0.00, 0.45, 0.35))
    initial_velocities_xyz = ((-0.45, 0.00, 0.25), (0.45, 0.00, -0.25))
    camera_position_xyz = (0.20, -1.10, 0.45)
    camera_target_xyz = (-0.05, 0.0, 0.30)


class SmartphoneLycheeEnvironment(NoCollisionSceneBase):
    """Smartphone and lychee sweeping same direction at different heights, low camera."""

    scene_metadata = make_pair_no_collision_metadata("smartphone_lychee")
    asset_names = ("smartphone_hot3d_robolab", "lychee01_fruits_veggies_robolab")
    initial_positions_xyz = ((-0.30, 0.20, 0.40), (-0.30, 0.45, 0.10))
    initial_velocities_xyz = ((0.45, 0.00, -0.05), (0.45, 0.00, 0.05))
    camera_position_xyz = (0.0, -0.75, 0.25)
    camera_target_xyz = (0.0, 0.0, 0.30)
    focal_length_mm = 28.0


class MouseKeyboardEnvironment(NoCollisionSceneBase):
    """Mouse and keyboard leaving the frame in opposite directions, high camera."""

    scene_metadata = make_pair_no_collision_metadata("mouse_keyboard")
    asset_names = ("computer_mouse_hot3d_robolab", "keyboard_hot3d_robolab")
    initial_positions_xyz = ((-0.10, 0.10, 0.40), (0.10, 0.40, 0.40))
    initial_velocities_xyz = ((-0.50, 0.00, 0.30), (0.50, 0.00, -0.30))
    camera_position_xyz = (0.0, -1.10, 1.00)
    camera_target_xyz = (0.0, 0.0, 0.15)
    focal_length_mm = 20.0


# ---------------------------------------------------------------------------
# Rotation-only variants (linear velocities = 0 for both objects).
# Each class reuses the assets / positions / camera of the matching
# translation scene; both objects spin in place at the angular velocities
# below (deterministic random samples, magnitudes in [1.0, 3.5] rad/s).
# Both objects sit at their initial positions, so the non-collision
# invariant is trivially preserved.
# ---------------------------------------------------------------------------


class LemonSoupRotationEnvironment(LemonSoupEnvironment):
    """Lemon and soup-can spinning in place at their initial positions."""

    scene_metadata = dataclasses.replace(LemonSoupEnvironment.scene_metadata, motion_type=MotionType.ROTATION)
    initial_velocities_xyz = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    initial_angular_velocities_xyz = ((2.29, -0.15, 0.67), (-0.85, -1.96, 2.02))


class TunaDrillRotationEnvironment(TunaDrillEnvironment):
    """Tuna can and drill spinning in place at their initial positions."""

    scene_metadata = dataclasses.replace(TunaDrillEnvironment.scene_metadata, motion_type=MotionType.ROTATION)
    initial_velocities_xyz = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    initial_angular_velocities_xyz = ((0.08, 0.63, -1.67), (-0.30, 2.72, -1.94))


class FoamBowlRotationEnvironment(FoamBowlEnvironment):
    """Foam roller and wooden bowl spinning in place at their initial positions."""

    scene_metadata = dataclasses.replace(FoamBowlEnvironment.scene_metadata, motion_type=MotionType.ROTATION)
    initial_velocities_xyz = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    initial_angular_velocities_xyz = ((1.75, -1.72, -0.98), (-0.95, 0.73, 1.78))


class KetchupSpatulaRotationEnvironment(KetchupSpatulaEnvironment):
    """Ketchup bottle and spatula spinning in place at their initial positions."""

    scene_metadata = dataclasses.replace(KetchupSpatulaEnvironment.scene_metadata, motion_type=MotionType.ROTATION)
    initial_velocities_xyz = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    initial_angular_velocities_xyz = ((-0.19, 2.06, -1.22), (-0.26, 3.19, 0.55))


class PomegranateSauceRotationEnvironment(PomegranateSauceEnvironment):
    """Pomegranate and tomato-sauce can spinning in place at their initial positions."""

    scene_metadata = dataclasses.replace(PomegranateSauceEnvironment.scene_metadata, motion_type=MotionType.ROTATION)
    initial_velocities_xyz = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    initial_angular_velocities_xyz = ((-2.33, 1.71, -1.96), (-0.64, -0.04, -0.91))


class CheezitMugRotationEnvironment(CheezitMugEnvironment):
    """Cheez-It box and mug spinning in place at their initial positions."""

    scene_metadata = dataclasses.replace(CheezitMugEnvironment.scene_metadata, motion_type=MotionType.ROTATION)
    initial_velocities_xyz = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    initial_angular_velocities_xyz = ((2.22, 1.83, 0.76), (-0.84, 0.45, -1.71))


class PopcornSpoonsRotationEnvironment(PopcornSpoonsEnvironment):
    """Popcorn box and wooden spoons spinning in place at their initial positions."""

    scene_metadata = dataclasses.replace(PopcornSpoonsEnvironment.scene_metadata, motion_type=MotionType.ROTATION)
    initial_velocities_xyz = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    initial_angular_velocities_xyz = ((3.42, -0.08, 0.20), (0.38, -0.46, -2.74))


class PepperYogurtRotationEnvironment(PepperYogurtEnvironment):
    """Pepper and yogurt cup spinning in place at their initial positions."""

    scene_metadata = dataclasses.replace(PepperYogurtEnvironment.scene_metadata, motion_type=MotionType.ROTATION)
    initial_velocities_xyz = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    initial_angular_velocities_xyz = ((-0.69, -1.51, 0.12), (-0.83, -1.02, -1.62))


class SmartphoneLycheeRotationEnvironment(SmartphoneLycheeEnvironment):
    """Smartphone and lychee spinning in place at their initial positions."""

    scene_metadata = dataclasses.replace(SmartphoneLycheeEnvironment.scene_metadata, motion_type=MotionType.ROTATION)
    initial_velocities_xyz = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    initial_angular_velocities_xyz = ((-1.28, 0.38, 2.90), (-0.12, 1.44, 0.0))


class MouseKeyboardRotationEnvironment(MouseKeyboardEnvironment):
    """Mouse and keyboard spinning in place at their initial positions."""

    scene_metadata = dataclasses.replace(MouseKeyboardEnvironment.scene_metadata, motion_type=MotionType.ROTATION)
    initial_velocities_xyz = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    initial_angular_velocities_xyz = ((1.00, -0.61, 1.29), (-0.87, -1.03, 0.30))


# ---------------------------------------------------------------------------
# Translation + rotation variants (linear velocities reused from the matching
# translation scene; angular velocities are independent random samples).
# Linear paths are unchanged so the non-collision invariant is preserved.
# ---------------------------------------------------------------------------


class LemonSoupTranslationRotationEnvironment(LemonSoupEnvironment):
    """``lemon_soup``'s linear velocities plus independent angular velocities."""

    scene_metadata = dataclasses.replace(
        LemonSoupEnvironment.scene_metadata, motion_type=MotionType.TRANSLATION_ROTATION
    )
    initial_angular_velocities_xyz = ((0.23, -2.93, 0.23), (-0.08, -0.02, -1.81))


class TunaDrillTranslationRotationEnvironment(TunaDrillEnvironment):
    """``tuna_drill``'s linear velocities plus independent angular velocities."""

    scene_metadata = dataclasses.replace(
        TunaDrillEnvironment.scene_metadata, motion_type=MotionType.TRANSLATION_ROTATION
    )
    initial_angular_velocities_xyz = ((1.63, 0.20, 2.74), (0.52, -0.92, -0.44))


class FoamBowlTranslationRotationEnvironment(FoamBowlEnvironment):
    """``foam_bowl``'s linear velocities plus independent angular velocities."""

    scene_metadata = dataclasses.replace(
        FoamBowlEnvironment.scene_metadata, motion_type=MotionType.TRANSLATION_ROTATION
    )
    initial_angular_velocities_xyz = ((0.39, -0.38, 1.09), (-1.47, 0.13, -2.50))


class KetchupSpatulaTranslationRotationEnvironment(KetchupSpatulaEnvironment):
    """``ketchup_spatula``'s linear velocities plus independent angular velocities."""

    scene_metadata = dataclasses.replace(
        KetchupSpatulaEnvironment.scene_metadata, motion_type=MotionType.TRANSLATION_ROTATION
    )
    initial_angular_velocities_xyz = ((0.15, -1.46, -1.63), (-2.67, -0.86, -1.49))


class PomegranateSauceTranslationRotationEnvironment(PomegranateSauceEnvironment):
    """``pomegranate_sauce``'s linear velocities plus independent angular velocities."""

    scene_metadata = dataclasses.replace(
        PomegranateSauceEnvironment.scene_metadata, motion_type=MotionType.TRANSLATION_ROTATION
    )
    initial_angular_velocities_xyz = ((-1.70, 0.89, -1.35), (-0.18, -1.42, -1.06))


class CheezitMugTranslationRotationEnvironment(CheezitMugEnvironment):
    """``cheezit_mug``'s linear velocities plus independent angular velocities."""

    scene_metadata = dataclasses.replace(
        CheezitMugEnvironment.scene_metadata, motion_type=MotionType.TRANSLATION_ROTATION
    )
    initial_angular_velocities_xyz = ((2.00, -0.06, 0.63), (-1.01, -0.11, -1.18))


class PopcornSpoonsTranslationRotationEnvironment(PopcornSpoonsEnvironment):
    """``popcorn_spoons``'s linear velocities plus independent angular velocities."""

    scene_metadata = dataclasses.replace(
        PopcornSpoonsEnvironment.scene_metadata, motion_type=MotionType.TRANSLATION_ROTATION
    )
    initial_angular_velocities_xyz = ((-0.82, 1.32, 0.28), (0.25, 1.30, -2.21))


class PepperYogurtTranslationRotationEnvironment(PepperYogurtEnvironment):
    """``pepper_yogurt``'s linear velocities plus independent angular velocities."""

    scene_metadata = dataclasses.replace(
        PepperYogurtEnvironment.scene_metadata, motion_type=MotionType.TRANSLATION_ROTATION
    )
    initial_angular_velocities_xyz = ((0.24, 1.83, 2.55), (2.05, 0.98, -1.40))


class SmartphoneLycheeTranslationRotationEnvironment(SmartphoneLycheeEnvironment):
    """``smartphone_lychee``'s linear velocities plus independent angular velocities."""

    scene_metadata = dataclasses.replace(
        SmartphoneLycheeEnvironment.scene_metadata, motion_type=MotionType.TRANSLATION_ROTATION
    )
    initial_angular_velocities_xyz = ((0.50, 2.21, -2.46), (-2.67, -1.28, -0.16))


class MouseKeyboardTranslationRotationEnvironment(MouseKeyboardEnvironment):
    """``mouse_keyboard``'s linear velocities plus independent angular velocities."""

    scene_metadata = dataclasses.replace(
        MouseKeyboardEnvironment.scene_metadata, motion_type=MotionType.TRANSLATION_ROTATION
    )
    initial_angular_velocities_xyz = ((0.34, -0.91, -0.77), (-1.94, 0.90, -0.33))

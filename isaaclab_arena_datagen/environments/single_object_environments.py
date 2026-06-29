# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Single-object dynamic scenes for data generation.

Each concrete class places a single Arena object at a chosen initial
position and gives it a constant linear velocity, angular velocity, or
both.  All scenes disable gravity and use no background -- only the
dynamic object and a light.

Compared with the two-object no-collision and collision scenes, the
single-object scenes isolate per-object motion so the refinement model
can learn from clean trajectories without inter-object occlusion.

This module defines three motion variants per object (10 objects total):

- ``translation`` (registry name = ``base_name``): linear velocity only.
  These are the original single-object scenes; each picks a fully 3D
  motion direction (every velocity component is non-trivial -- no purely
  lateral, vertical, or depth motion), a unique speed, and a camera pose
  so the model sees a diverse set of viewpoints and motion patterns.
- ``rotation`` (suffix ``_rotation``): linear velocity zero, angular
  velocity non-zero.  Same asset / position / camera as the matching
  translation scene; the object spins in place.
- ``translation_rotation`` (suffix ``_translation_rotation``): the
  translation scene's linear velocity plus an independently chosen
  angular velocity.

Category, motion variant, and other identifying data are carried by the
``scene_metadata`` :class:`SceneMetaData` class attribute; the registry
name and on-disk subpath are derived from it via
:func:`datagen.scene_metadata.get_registry_name` and
:func:`datagen.scene_metadata.get_dataset_subpath`.

To add a new single-object scene, subclass :class:`SingleObjectSceneBase`
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
    make_single_object_metadata,
)
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

_NEUTRAL_ROTATION_XYZW = (0.0, 0.0, 0.0, 1.0)
_DEFAULT_FOCAL_LENGTH_MM = 24.0


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class SingleObjectSceneBase(ExampleEnvironmentBase):
    """One object moving on a constant-velocity path, gravity disabled, no background.

    Subclasses set class-level attributes to configure the scene:
    ``asset_name``, ``initial_position_xyz``, ``initial_velocity_xyz``,
    ``camera_position_xyz``, ``camera_target_xyz``, and optionally
    ``initial_angular_velocity_xyz`` (defaults to zero, used by
    ``rotation`` / ``translation_rotation`` variants) and
    ``focal_length_mm`` (defaults to 24 mm).

    Every concrete subclass must also declare a ``scene_metadata``
    :class:`SceneMetaData` instance.  The ``name`` attribute consumed by
    :class:`IsaacLabArenaEnvironment` and the ``DATAGEN_ENVIRONMENTS``
    registry is derived from ``scene_metadata`` automatically in
    ``__init_subclass__``; rotation / translation_rotation variants
    share their parent's ``base_name`` (set via
    ``dataclasses.replace``) so all three variants land in the same
    on-disk dirname under ``one_object/<motion>/<base_name>/``.
    """

    scene_metadata: ClassVar[SceneMetaData]
    asset_name: str
    initial_position_xyz: Coord3D
    initial_velocity_xyz: Coord3D = (0.0, 0.0, 0.0)
    initial_angular_velocity_xyz: Coord3D = (0.0, 0.0, 0.0)
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
            _args_cli: Parsed CLI arguments (unused by single-object scenes).
        """
        import isaaclab.sim as sim_utils

        obj = self.asset_registry.get_asset_by_name(self.asset_name)()
        obj.set_initial_pose(
            Pose(
                position_xyz=self.initial_position_xyz,
                rotation_xyzw=_NEUTRAL_ROTATION_XYZW,
            )
        )
        obj.set_initial_velocity(
            Velocity(
                linear_xyz=self.initial_velocity_xyz,
                angular_xyz=self.initial_angular_velocity_xyz,
            )
        )
        # IsaacLab leaves `rigid_props=None` on the spawn cfg, so default-
        # construct it before flipping `disable_gravity`.
        if obj.object_cfg.spawn.rigid_props is None:
            obj.object_cfg.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg()
        obj.object_cfg.spawn.rigid_props.disable_gravity = True

        dome_cfg = sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0)
        light = self.asset_registry.get_asset_by_name("light")(spawner_cfg=dome_cfg)

        distant_cfg = sim_utils.DistantLightCfg(color=(1.0, 1.0, 1.0), intensity=3000.0, angle=0.53)
        distant_light = self.asset_registry.get_asset_by_name("light")(
            instance_name="distant_light",
            prim_path="/World/DistantLight",
            spawner_cfg=distant_cfg,
        )

        scene = Scene(assets=[obj, light, distant_light])

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
        """No extra CLI arguments for single-object scenes."""


# ---------------------------------------------------------------------------
# Concrete scenes
# ---------------------------------------------------------------------------


class LemonEnvironment(SingleObjectSceneBase):
    """Lemon descending and sweeping leftward into the scene depth, |v|=0.78 m/s."""

    scene_metadata = make_single_object_metadata("lemon")
    asset_name = "lemon_01_fruits_veggies_robolab"
    initial_position_xyz = (0.29, 0.23, 0.14)
    initial_velocity_xyz = (-0.62, 0.35, -0.32)
    camera_position_xyz = (0.0, -1.00, 0.40)
    camera_target_xyz = (0.0, 0.0, 0.30)


class DrillEnvironment(SingleObjectSceneBase):
    """Drill rising while drifting back-left toward the camera, |v|=0.48 m/s."""

    scene_metadata = make_single_object_metadata("drill")
    asset_name = "cordless_drill_ycb_robolab"
    initial_position_xyz = (0.05, 0.48, 0.42)
    initial_velocity_xyz = (-0.33, -0.20, 0.28)
    camera_position_xyz = (0.0, -1.00, 0.50)
    camera_target_xyz = (0.0, 0.0, 0.30)


class MugEnvironment(SingleObjectSceneBase):
    """Mug drifting toward the camera, slightly down and right, |v|=0.36 m/s."""

    scene_metadata = make_single_object_metadata("mug")
    asset_name = "mug_hot3d_robolab"
    initial_position_xyz = (-0.26, 0.30, 0.37)
    initial_velocity_xyz = (0.25, -0.23, -0.11)
    camera_position_xyz = (0.0, -1.00, 0.40)
    camera_target_xyz = (0.0, 0.0, 0.30)


class SpatulaEnvironment(SingleObjectSceneBase):
    """Spatula falling diagonally to the right and away from the camera, |v|=0.54 m/s."""

    scene_metadata = make_single_object_metadata("spatula")
    asset_name = "spatula_hot3d_robolab"
    initial_position_xyz = (0.13, 0.19, 0.39)
    initial_velocity_xyz = (0.17, 0.41, -0.30)
    camera_position_xyz = (0.0, -1.00, 0.40)
    camera_target_xyz = (0.0, 0.0, 0.30)


class SauceEnvironment(SingleObjectSceneBase):
    """Tomato-sauce can rising while approaching the camera and crossing right, |v|=0.72 m/s."""

    scene_metadata = make_single_object_metadata("sauce")
    asset_name = "tomato_sauce_can_hot3d_robolab"
    initial_position_xyz = (0.08, 0.32, 0.37)
    initial_velocity_xyz = (0.44, -0.50, 0.27)
    camera_position_xyz = (0.40, -1.00, 0.40)
    camera_target_xyz = (0.0, 0.0, 0.30)


class CheezitEnvironment(SingleObjectSceneBase):
    """Cheez-It box drifting slowly toward the camera and down, |v|=0.30 m/s."""

    scene_metadata = make_single_object_metadata("cheezit")
    asset_name = "cheez_it_ycb_robolab"
    initial_position_xyz = (-0.30, 0.35, 0.49)
    initial_velocity_xyz = (0.10, -0.22, -0.18)
    camera_position_xyz = (0.0, -1.00, 0.40)
    camera_target_xyz = (0.0, 0.0, 0.30)


class PopcornEnvironment(SingleObjectSceneBase):
    """Popcorn box accelerating up-and-right toward the camera, |v|=0.84 m/s; tight telephoto."""

    scene_metadata = make_single_object_metadata("popcorn")
    asset_name = "popcorn_box_hope_robolab"
    initial_position_xyz = (0.21, 0.17, 0.15)
    initial_velocity_xyz = (0.69, -0.30, 0.38)
    camera_position_xyz = (0.0, -1.00, 0.40)
    camera_target_xyz = (0.0, 0.0, 0.30)
    focal_length_mm = 28.0


class PepperEnvironment(SingleObjectSceneBase):
    """Red bell pepper rising while drifting away to the left, |v|=0.66 m/s."""

    scene_metadata = make_single_object_metadata("pepper")
    asset_name = "red_bell_pepper_objaverse_robolab"
    initial_position_xyz = (0.00, 0.23, 0.24)
    initial_velocity_xyz = (-0.33, 0.29, 0.49)
    camera_position_xyz = (0.0, -1.00, 0.40)
    camera_target_xyz = (0.0, 0.0, 0.30)


class SmartphoneEnvironment(SingleObjectSceneBase):
    """Smartphone rising leftward toward the camera, |v|=0.42 m/s; viewed from a low camera."""

    scene_metadata = make_single_object_metadata("smartphone")
    asset_name = "smartphone_hot3d_robolab"
    initial_position_xyz = (-0.16, 0.20, 0.32)
    initial_velocity_xyz = (-0.21, -0.15, 0.33)
    camera_position_xyz = (0.0, -0.75, 0.25)
    camera_target_xyz = (0.0, 0.0, 0.30)


class KeyboardEnvironment(SingleObjectSceneBase):
    """Keyboard descending while sweeping left and away, |v|=0.60 m/s; high camera, wide FOV."""

    scene_metadata = make_single_object_metadata("keyboard")
    asset_name = "keyboard_hot3d_robolab"
    initial_position_xyz = (0.25, 0.48, 0.43)
    initial_velocity_xyz = (-0.34, 0.25, -0.43)
    camera_position_xyz = (0.0, -1.10, 1.00)
    camera_target_xyz = (0.0, 0.0, 0.15)
    focal_length_mm = 20.0


# ---------------------------------------------------------------------------
# Rotation-only variants (linear velocity = 0)
# Each class reuses the asset / position / camera of the matching translation
# scene; the object spins in place at the angular velocity below.
# Angular velocities are deterministic random samples (seed=42) with
# magnitudes in [1.0, 3.5] rad/s.
# ---------------------------------------------------------------------------


class LemonRotationEnvironment(LemonEnvironment):
    """Lemon spinning in place; linear velocity zero."""

    scene_metadata = dataclasses.replace(LemonEnvironment.scene_metadata, motion_type=MotionType.ROTATION)
    initial_velocity_xyz = (0.0, 0.0, 0.0)
    initial_angular_velocity_xyz = (-0.34, -0.40, -1.60)


class DrillRotationEnvironment(DrillEnvironment):
    """Drill spinning in place; linear velocity zero."""

    scene_metadata = dataclasses.replace(DrillEnvironment.scene_metadata, motion_type=MotionType.ROTATION)
    initial_velocity_xyz = (0.0, 0.0, 0.0)
    initial_angular_velocity_xyz = (0.40, 2.34, 1.27)


class MugRotationEnvironment(MugEnvironment):
    """Mug spinning in place; linear velocity zero."""

    scene_metadata = dataclasses.replace(MugEnvironment.scene_metadata, motion_type=MotionType.ROTATION)
    initial_velocity_xyz = (0.0, 0.0, 0.0)
    initial_angular_velocity_xyz = (0.90, -0.73, -1.70)


class SpatulaRotationEnvironment(SpatulaEnvironment):
    """Spatula spinning in place; linear velocity zero."""

    scene_metadata = dataclasses.replace(SpatulaEnvironment.scene_metadata, motion_type=MotionType.ROTATION)
    initial_velocity_xyz = (0.0, 0.0, 0.0)
    initial_angular_velocity_xyz = (1.84, 0.35, -1.27)


class SauceRotationEnvironment(SauceEnvironment):
    """Tomato-sauce can spinning in place; linear velocity zero."""

    scene_metadata = dataclasses.replace(SauceEnvironment.scene_metadata, motion_type=MotionType.ROTATION)
    initial_velocity_xyz = (0.0, 0.0, 0.0)
    initial_angular_velocity_xyz = (2.07, 0.35, -1.58)


class CheezitRotationEnvironment(CheezitEnvironment):
    """Cheez-It box spinning in place; linear velocity zero."""

    scene_metadata = dataclasses.replace(CheezitEnvironment.scene_metadata, motion_type=MotionType.ROTATION)
    initial_velocity_xyz = (0.0, 0.0, 0.0)
    initial_angular_velocity_xyz = (-1.97, -0.57, -1.38)


class PopcornRotationEnvironment(PopcornEnvironment):
    """Popcorn box spinning in place; linear velocity zero."""

    scene_metadata = dataclasses.replace(PopcornEnvironment.scene_metadata, motion_type=MotionType.ROTATION)
    initial_velocity_xyz = (0.0, 0.0, 0.0)
    initial_angular_velocity_xyz = (0.18, -0.45, -2.98)


class PepperRotationEnvironment(PepperEnvironment):
    """Pepper spinning in place; linear velocity zero."""

    scene_metadata = dataclasses.replace(PepperEnvironment.scene_metadata, motion_type=MotionType.ROTATION)
    initial_velocity_xyz = (0.0, 0.0, 0.0)
    initial_angular_velocity_xyz = (-0.42, -1.25, -0.44)


class SmartphoneRotationEnvironment(SmartphoneEnvironment):
    """Smartphone spinning in place; linear velocity zero."""

    scene_metadata = dataclasses.replace(SmartphoneEnvironment.scene_metadata, motion_type=MotionType.ROTATION)
    initial_velocity_xyz = (0.0, 0.0, 0.0)
    initial_angular_velocity_xyz = (1.12, -0.31, -0.40)


class KeyboardRotationEnvironment(KeyboardEnvironment):
    """Keyboard spinning in place; linear velocity zero."""

    scene_metadata = dataclasses.replace(KeyboardEnvironment.scene_metadata, motion_type=MotionType.ROTATION)
    initial_velocity_xyz = (0.0, 0.0, 0.0)
    initial_angular_velocity_xyz = (1.48, 1.03, 1.74)


# ---------------------------------------------------------------------------
# Translation + rotation variants (linear velocity reused from the matching
# translation scene; angular velocity is an independent random sample).
# ---------------------------------------------------------------------------


class LemonTranslationRotationEnvironment(LemonEnvironment):
    """Lemon: ``lemon``'s linear velocity plus an independent angular velocity."""

    scene_metadata = dataclasses.replace(LemonEnvironment.scene_metadata, motion_type=MotionType.TRANSLATION_ROTATION)
    initial_angular_velocity_xyz = (0.73, -1.95, 1.08)


class DrillTranslationRotationEnvironment(DrillEnvironment):
    """Drill: ``drill``'s linear velocity plus an independent angular velocity."""

    scene_metadata = dataclasses.replace(DrillEnvironment.scene_metadata, motion_type=MotionType.TRANSLATION_ROTATION)
    initial_angular_velocity_xyz = (2.28, -0.39, -0.58)


class MugTranslationRotationEnvironment(MugEnvironment):
    """Mug: ``mug``'s linear velocity plus an independent angular velocity."""

    scene_metadata = dataclasses.replace(MugEnvironment.scene_metadata, motion_type=MotionType.TRANSLATION_ROTATION)
    initial_angular_velocity_xyz = (1.47, -2.69, 0.75)


class SpatulaTranslationRotationEnvironment(SpatulaEnvironment):
    """Spatula: ``spatula``'s linear velocity plus an independent angular velocity."""

    scene_metadata = dataclasses.replace(SpatulaEnvironment.scene_metadata, motion_type=MotionType.TRANSLATION_ROTATION)
    initial_angular_velocity_xyz = (-0.90, -0.48, 0.46)


class SauceTranslationRotationEnvironment(SauceEnvironment):
    """Tomato-sauce can: ``sauce``'s linear velocity plus an independent angular velocity."""

    scene_metadata = dataclasses.replace(SauceEnvironment.scene_metadata, motion_type=MotionType.TRANSLATION_ROTATION)
    initial_angular_velocity_xyz = (0.15, 1.08, -0.51)


class CheezitTranslationRotationEnvironment(CheezitEnvironment):
    """Cheez-It box: ``cheezit``'s linear velocity plus an independent angular velocity."""

    scene_metadata = dataclasses.replace(CheezitEnvironment.scene_metadata, motion_type=MotionType.TRANSLATION_ROTATION)
    initial_angular_velocity_xyz = (0.11, 1.02, -1.35)


class PopcornTranslationRotationEnvironment(PopcornEnvironment):
    """Popcorn box: ``popcorn``'s linear velocity plus an independent angular velocity."""

    scene_metadata = dataclasses.replace(PopcornEnvironment.scene_metadata, motion_type=MotionType.TRANSLATION_ROTATION)
    initial_angular_velocity_xyz = (-1.22, -1.40, -0.52)


class PepperTranslationRotationEnvironment(PepperEnvironment):
    """Pepper: ``pepper``'s linear velocity plus an independent angular velocity."""

    scene_metadata = dataclasses.replace(PepperEnvironment.scene_metadata, motion_type=MotionType.TRANSLATION_ROTATION)
    initial_angular_velocity_xyz = (0.74, 2.86, -1.56)


class SmartphoneTranslationRotationEnvironment(SmartphoneEnvironment):
    """Smartphone: ``smartphone``'s linear velocity plus an independent angular velocity."""

    scene_metadata = dataclasses.replace(
        SmartphoneEnvironment.scene_metadata, motion_type=MotionType.TRANSLATION_ROTATION
    )
    initial_angular_velocity_xyz = (-0.83, -1.12, 0.31)


class KeyboardTranslationRotationEnvironment(KeyboardEnvironment):
    """Keyboard: ``keyboard``'s linear velocity plus an independent angular velocity."""

    scene_metadata = dataclasses.replace(
        KeyboardEnvironment.scene_metadata, motion_type=MotionType.TRANSLATION_ROTATION
    )
    initial_angular_velocity_xyz = (-0.19, -1.43, -1.31)

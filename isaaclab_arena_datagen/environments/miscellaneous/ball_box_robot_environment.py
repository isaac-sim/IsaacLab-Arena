# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import argparse
from typing import Any

from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.tasks.no_task import NoTask
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.velocity import Velocity
from isaaclab_arena_datagen.camera_trajectory import CameraViewTrajectory
from isaaclab_arena_datagen.scene_metadata import get_registry_name, make_reference_metadata
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase


class BallBoxRobotEnvironment(ExampleEnvironmentBase):
    """Ball, box, and robot environment for data generation."""

    scene_metadata = make_reference_metadata("ball_box_robot", num_objects=3)
    name: str = get_registry_name(scene_metadata)

    def get_env(self, args_cli: argparse.Namespace) -> Any:
        """Build and return the Isaac Lab Arena environment.

        Args:
            args_cli: Parsed CLI arguments including embodiment and object options.
        """
        background = self.asset_registry.get_asset_by_name("kitchen")()
        microwave = self.asset_registry.get_asset_by_name("microwave")()
        dynamic_object = self.asset_registry.get_asset_by_name("sphere")()
        cracker_box = self.asset_registry.get_asset_by_name("cracker_box")()
        assets = [background, microwave, dynamic_object, cracker_box]
        embodiment = self.asset_registry.get_asset_by_name("gr1_joint")(enable_cameras=args_cli.enable_cameras)
        embodiment.set_initial_pose(
            Pose(
                position_xyz=(-0.4, 0.0, 0.0),
                rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
        )

        # Put the microwave on the packing table.
        microwave_pose = Pose(
            position_xyz=(0.4, -0.00586, 0.22773),
            rotation_xyzw=(0, 0, -0.7071068, 0.7071068),
        )
        microwave.set_initial_pose(microwave_pose)

        sphere_pose = Pose(
            position_xyz=(0.466, -0.737, 0.4),
            rotation_xyzw=(-0.5, 0.5, -0.5, 0.5),
        )
        dynamic_object.set_initial_pose(sphere_pose)
        dynamic_object.set_initial_velocity(Velocity(linear_xyz=(-0.1, 1.0, -0.5)))
        # Disable gravity for the dynamic object
        dynamic_object.object_cfg.spawn.rigid_props.disable_gravity = True

        cracker_box_pose = Pose(
            position_xyz=(0.466, -0.437, 0.154),
            rotation_xyzw=(-0.5, 0.5, -0.5, 0.5),
        )
        cracker_box.set_initial_pose(cracker_box_pose)

        # Compose the scene
        scene = Scene(assets=assets)

        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=NoTask(),
            teleop_device=None,
        )

        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        """Add ball-box-robot-specific CLI arguments to the parser."""

    @staticmethod
    def get_default_cameras(num_steps: int) -> list[CameraViewTrajectory]:
        """Return default camera configurations for data generation.

        Args:
            num_steps: Number of simulation steps (used for dynamic trajectories).
        """
        x_start, x_end = -2.0, 0.4
        return [
            CameraViewTrajectory(
                position=(0.0, -0.737, 1.0),
                target=(0.466, -0.737, 0.4),
                focal_length_mm=24.0,
            ),
            CameraViewTrajectory(
                position=[
                    (
                        x_start + (x_end - x_start) * step_index / max(num_steps - 1, 1),
                        -0.337,
                        0.8,
                    )
                    for step_index in range(num_steps)
                ],
                target=(0.466, -0.737, 0.4),
                focal_length_mm=12.0,
            ),
        ]

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
from isaaclab_arena_datagen.camera_trajectory import CameraViewTrajectory
from isaaclab_arena_datagen.scene_metadata import get_registry_name, make_reference_metadata
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase


class SingleCrackerBoxEnvironment(ExampleEnvironmentBase):
    """Single cracker box environment with one camera for data generation."""

    scene_metadata = make_reference_metadata("single_cracker_box", num_objects=1)
    name: str = get_registry_name(scene_metadata)

    def get_env(self, _args_cli: argparse.Namespace) -> Any:
        """Build and return the Isaac Lab Arena environment.

        Args:
            args_cli: Parsed CLI arguments including embodiment and object options.
        """
        import isaaclab.sim as sim_utils

        cracker_box = self.asset_registry.get_asset_by_name("cracker_box")()

        cracker_box_pose = Pose(
            position_xyz=(0.466, -0.437, 0.154),
            rotation_xyzw=(-0.5, 0.5, -0.5, 0.5),
        )
        cracker_box.set_initial_pose(cracker_box_pose)

        # Dome light for uniform ambient fill; distant light for directional shadows and highlights.
        dome_cfg = sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0)
        light = self.asset_registry.get_asset_by_name("light")(spawner_cfg=dome_cfg)

        distant_cfg = sim_utils.DistantLightCfg(color=(1.0, 1.0, 1.0), intensity=3000.0, angle=0.53)
        distant_light = self.asset_registry.get_asset_by_name("light")(
            instance_name="distant_light",
            prim_path="/World/DistantLight",
            spawner_cfg=distant_cfg,
        )

        scene = Scene(assets=[cracker_box, light, distant_light])

        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=None,
            scene=scene,
            task=NoTask(),
            teleop_device=None,
        )

        return isaaclab_arena_environment

    @staticmethod
    def get_default_cameras(_num_steps: int) -> list[CameraViewTrajectory]:
        """Return default camera configurations for data generation.

        Args:
            num_steps: Number of simulation steps (used for dynamic trajectories).
        """
        return [
            CameraViewTrajectory(
                position=(0.0, -0.437, 0.8),
                target=(0.466, -0.437, 0.154),
                focal_length_mm=24.0,
            ),
        ]

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        """Add single-cracker-box-specific CLI arguments to the parser."""

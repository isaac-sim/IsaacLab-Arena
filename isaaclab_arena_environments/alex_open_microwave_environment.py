# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@register_environment
class AlexOpenMicrowaveEnvironment(ExampleEnvironmentBase):
    """Open-microwave task with the IHMC Alex V1 or V2 robot.

    For ability hands (default), mount the ihmc-alex-sdk root so both alex-models
    (must include ``alex_V1_description`` and ``alex_V2_description`` for V2)
    and ihmc_hands_ros2 are visible inside the container::

        ./docker/run_docker.sh -m /path/to/ihmc-alex-sdk

    For nubs forearms only, alex-models alone is sufficient::

        ./docker/run_docker.sh -m /path/to/ihmc-alex-sdk/alex-models

    V2 embodiments: ``alex_v2_pink``, ``alex_v2_ability_hands``,
    ``alex_v2_ability_hands_joint_pos``.

    Usage::

        python isaaclab_arena/scripts/imitation_learning/record_demos.py \\
            --device cpu --viz kit --enable_cameras \\
            --dataset_file /tmp/alex_demo.hdf5 \\
            --num_demos 1 --num_success_steps 2 \\
            alex_open_microwave \\
            --teleop_device openxr \\
            --embodiment alex_ability_hands

    Stereo ZED X Mini cameras (``zed_left_cam``, ``zed_right_cam``) mount on
    ``HEAD_LINK`` at the forehead bracket (50 mm baseline, 2.2 mm / 110° FOV lens).
    Pass ``--enable_cameras`` to add camera observations
    and HDF5 recordings (``camera_obs/zed_left_cam_rgb``, ``camera_obs/zed_right_cam_rgb``).

    Convert teleop HDF5 to LeRobot (place file under ``/datasets`` in Docker)::

        python isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py \\
            --yaml_file isaaclab_arena_gr00t/lerobot/config/alex_open_microwave_config.yaml
    """

    name: str = "alex_open_microwave"

    def get_env(self, args_cli: argparse.Namespace) -> IsaacLabArenaEnvironment:
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.open_door_task import OpenDoorTask
        from isaaclab_arena.utils.pose import Pose

        background = self.asset_registry.get_asset_by_name("kitchen")()
        microwave = self.asset_registry.get_asset_by_name("microwave")()
        assets = [background, microwave]

        assert args_cli.embodiment in [
            "alex_pink",
            "alex_ability_hands",
            "alex_ability_hands_joint_pos",
            "alex_v2_pink",
            "alex_v2_ability_hands",
            "alex_v2_ability_hands_joint_pos",
        ], "Invalid Alex embodiment {}".format(args_cli.embodiment)
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)
        # Alex stands ~0.15 m further back than GR1T2 to account for the longer arm reach.
        embodiment.set_initial_pose(Pose(position_xyz=(-0.40, -0.1, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        microwave.set_initial_pose(
            Pose(
                position_xyz=(0.4, -0.00586, 0.22773),
                rotation_xyzw=(0, 0, -0.7071068, 0.7071068),
            )
        )

        if args_cli.object is not None:
            obj = self.asset_registry.get_asset_by_name(args_cli.object)()
            obj.set_initial_pose(
                Pose(
                    position_xyz=(0.466, -0.437, 0.154),
                    rotation_xyzw=(-0.5, 0.5, -0.5, 0.5),
                )
            )
            assets.append(obj)

        scene = Scene(assets=assets)

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=OpenDoorTask(microwave, openness_threshold=0.8, reset_openness=0.2, episode_length_s=100 / 30),
            teleop_device=teleop_device,
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--object", type=str, default=None)
        parser.add_argument("--teleop_device", type=str, default=None)
        parser.add_argument("--embodiment", type=str, default="alex_ability_hands")

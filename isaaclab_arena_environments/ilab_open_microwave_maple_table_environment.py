# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import math
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@register_environment
class IlabOpenMicrowaveMapleTableEnvironment(ExampleEnvironmentBase):
    """Open the door of a microwave resting on the maple table.

    Shares the maple-table background used by the first ilab environment; the
    microwave is placed on the table and the task succeeds once its door swings
    past the openness threshold.
    """

    name: str = "ilab_open_microwave_maple_table"

    def get_env(self, args_cli: argparse.Namespace) -> IsaacLabArenaEnvironment:
        from isaaclab.envs.common import ViewerCfg

        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.relations import IsAnchor, On, RotateAroundSolution
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.open_door_task import OpenDoorTask
        from isaaclab_arena.utils.pose import Pose

        # Step 1: Retrieve assets from the registry
        background = self.asset_registry.get_asset_by_name("maple_table_robolab")()
        microwave = self.asset_registry.get_asset_by_name("microwave")()

        # Step 2: Describe spatial relationships. The microwave sits on the same
        # table the first ilab environment uses; the yaw rotation faces its door
        # toward the robot.
        table_reference = ObjectReference(
            name="table",
            prim_path="{ENV_REGEX_NS}/maple_table_robolab/table",
            parent_asset=background,
            object_type=ObjectType.RIGID,
        )
        table_reference.add_relation(IsAnchor())

        microwave.add_relation(On(table_reference))
        microwave.add_relation(RotateAroundSolution(yaw_rad=-math.pi / 2))

        # Step 3: Configure lighting
        light = self.asset_registry.get_asset_by_name("light")()
        if args_cli.hdr is not None:
            light.add_hdr(self.hdr_registry.get_hdr_by_name(args_cli.hdr)())
        light.set_intensity(2000.0)

        # Step 4: Select the embodiment
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
            enable_cameras=args_cli.enable_cameras,
        )
        embodiment.set_initial_pose(Pose(position_xyz=(-0.3, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

        # Step 5: Compose the scene
        scene = Scene(assets=[background, light, microwave, table_reference])

        # Step 6: Define the task as opening the microwave door
        task = OpenDoorTask(
            openable_object=microwave,
            openness_threshold=args_cli.openness_threshold,
            reset_openness=0.0,
            episode_length_s=20.0,
        )

        # Set viewport camera to match the robolab droid view
        def _set_viewer_cfg(env_cfg):
            env_cfg.viewer = ViewerCfg(eye=(1.5, 0.0, 1.0), lookat=(0.2, 0.0, 0.0))
            return env_cfg

        # Step 7: Assemble the environment
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            env_cfg_callback=_set_viewer_cfg,
        )
        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--embodiment", type=str, default="droid_abs_joint_pos")
        parser.add_argument("--openness_threshold", type=float, default=0.8)
        parser.add_argument("--hdr", type=str, default="home_office_robolab")

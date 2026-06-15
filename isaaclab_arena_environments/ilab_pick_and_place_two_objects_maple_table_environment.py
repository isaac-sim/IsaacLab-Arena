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
class IlabPickAndPlaceTwoObjectsMapleTableEnvironment(ExampleEnvironmentBase):
    """Pick-and-place of two objects into a destination on the maple table.

    Completion order does not matter; the composite task succeeds once both
    objects rest on the destination.
    """

    name: str = "ilab_pick_and_place_two_objects_maple_table"

    def get_env(self, args_cli: argparse.Namespace) -> IsaacLabArenaEnvironment:
        import isaaclab.sim as sim_utils
        from isaaclab.envs.common import ViewerCfg

        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.relations import IsAnchor, On
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask

        # Step 1: Retrieve assets from the registry
        background = self.asset_registry.get_asset_by_name("maple_table_robolab")()
        pick_up_object_1 = self.asset_registry.get_asset_by_name(args_cli.pick_up_object_1)()
        pick_up_object_2 = self.asset_registry.get_asset_by_name(args_cli.pick_up_object_2)()
        destination_location = self.asset_registry.get_asset_by_name(args_cli.destination_location)()

        # Step 2: Describe spatial relationships
        table_reference = ObjectReference(
            name="table",
            prim_path="{ENV_REGEX_NS}/maple_table_robolab/table",
            parent_asset=background,
            object_type=ObjectType.RIGID,
        )
        table_reference.add_relation(IsAnchor())

        pick_up_object_1.add_relation(On(table_reference))
        pick_up_object_2.add_relation(On(table_reference))
        destination_location.add_relation(On(table_reference))

        # Step 3: Configure lighting
        light = self.asset_registry.get_asset_by_name("light")(
            spawner_cfg=sim_utils.DomeLightCfg(intensity=args_cli.light_intensity),
        )
        if args_cli.hdr is not None:
            light.add_hdr(self.hdr_registry.get_hdr_by_name(args_cli.hdr)())

        # Step 4: Select the embodiment
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
            enable_cameras=args_cli.enable_cameras,
        )

        # Step 5: Compose the scene
        scene = Scene(
            assets=[background, light, pick_up_object_1, pick_up_object_2, destination_location, table_reference]
        )

        # Step 6: Define the task as a composite of two pick-and-place subtasks. Each subtask's contact
        # sensor is named after its pick-up object, so the combined scene config does not collide.
        pick_and_place_task_1 = PickAndPlaceTask(
            pick_up_object=pick_up_object_1,
            destination_location=destination_location,
            destination_object=destination_location,
            background_scene=background,
        )
        pick_and_place_task_2 = PickAndPlaceTask(
            pick_up_object=pick_up_object_2,
            destination_location=destination_location,
            destination_object=destination_location,
            background_scene=background,
        )
        task = CompositeTaskBase(
            subtasks=[pick_and_place_task_1, pick_and_place_task_2],
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
        parser.add_argument("--teleop_device", type=str, default=None)
        parser.add_argument("--hdr", type=str, default=None)
        parser.add_argument("--light_intensity", type=float, default=500.0)
        parser.add_argument("--pick_up_object_1", type=str, default="rubiks_cube_hot3d_robolab")
        parser.add_argument("--pick_up_object_2", type=str, default="mug_ycb_robolab")
        parser.add_argument("--destination_location", type=str, default="bowl_ycb_robolab")

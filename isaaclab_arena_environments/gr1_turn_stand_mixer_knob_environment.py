# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

# NOTE(alexmillane, 2025.09.04): There is an issue with type annotation in this file.
# We cannot annotate types which require the simulation app to be started in order to
# import, because this file is used to retrieve CLI arguments, so it must be imported
# before the simulation app is started.
# TODO(alexmillane, 2025.09.04): Fix this.


class Gr1TurnStandMixerKnobEnvironment(ExampleEnvironmentBase):

    name: str = "gr1_turn_stand_mixer_knob"

    def get_env(self, args_cli: argparse.Namespace):  # -> IsaacLabArenaEnvironment:
        import math

        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.relations import AtPosition, IsAnchor, NextTo, On, RotateAroundSolution, Side
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.turn_knob_task import TurnKnobTask
        from isaaclab_arena.utils.pose import Pose

        background = self.asset_registry.get_asset_by_name("kitchen")()
        table_top_reference = ObjectReference(
            name="table_top_reference",
            prim_path="{ENV_REGEX_NS}/kitchen/Kitchen_Counter/TRS_Base/TRS_Static/Counter_Top_A",
            parent_asset=background,
        )
        table_top_reference.add_relation(IsAnchor())
        assert args_cli.embodiment in ["gr1_pink", "gr1_joint"], "Invalid GR1T2 embodiment {}".format(
            args_cli.embodiment
        )
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)
        embodiment.set_initial_pose(Pose(position_xyz=(-0.4, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

        teleop_device = None
        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()

        stand_mixer = self.asset_registry.get_asset_by_name("stand_mixer")()
        stand_mixer.add_relation(On(table_top_reference, clearance_m=0.02))
        stand_mixer.add_relation(AtPosition(x=0.4, y=0.0))
        stand_mixer.add_relation(RotateAroundSolution(yaw_rad=-math.pi / 2))

        assets = [background, table_top_reference, stand_mixer]

        if args_cli.object is not None:
            object = self.asset_registry.get_asset_by_name(args_cli.object)()
            object.add_relation(On(table_top_reference, clearance_m=0.02))
            object.add_relation(NextTo(stand_mixer, side=Side.NEGATIVE_Y, distance_m=0.05))
            assets.append(object)

        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=Scene(assets=assets),
            task=TurnKnobTask(
                turnable_object=stand_mixer, target_level=args_cli.target_level, reset_level=args_cli.reset_level
            ),
            teleop_device=teleop_device,
        )

        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--object", type=str, default=None)
        # NOTE(alexmillane, 2025.09.04): We need a teleop device argument in order
        # to be used in the record_demos.py script.
        parser.add_argument("--teleop_device", type=str, default=None)
        # Note (xinjieyao, 2025.10.06): Add the embodiment argument for PINK IK EEF control or Joint positional control
        parser.add_argument("--embodiment", type=str, default="gr1_pink")
        parser.add_argument("--target_level", type=int, default=4)
        parser.add_argument("--reset_level", type=int, default=-1)

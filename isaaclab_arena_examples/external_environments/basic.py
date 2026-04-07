# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Simple example externally-defined environment.

This demonstrates how to define an external environment that can be loaded
into Isaac Lab Arena via the ``--external_environment_class_path`` CLI flag::

    python isaaclab_arena/evaluation/policy_runner.py \\
        --policy_type zero_action --num_steps 10 \\
        --external_environment_class_path isaaclab_arena_examples.external_environments.basic:ExternalFrankaTableEnvironment \\
        franka_table
"""

import argparse

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase


class ExternalFrankaTableEnvironment(ExampleEnvironmentBase):

    name: str = "franka_table"

    def get_env(self, args_cli: argparse.Namespace):  # -> IsaacLabArenaEnvironment:
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.relations import IsAnchor, On
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.no_task import NoTask

        # Grab some assets from the registry.
        background = self.asset_registry.get_asset_by_name("maple_table_robolab")()
        light = self.asset_registry.get_asset_by_name("light")()
        pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)()
        embodiment = self.asset_registry.get_asset_by_name("franka_ik")()

        # Position the assets
        table_reference = ObjectReference(
            name="table",
            prim_path="{ENV_REGEX_NS}/maple_table_robolab/table",
            parent_asset=background,
        )
        table_reference.add_relation(IsAnchor())
        pick_up_object.add_relation(On(table_reference))

        # Compose the scene
        scene = Scene(assets=[background, table_reference, pick_up_object, light])

        # Create the environment
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=NoTask(),
        )
        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--object", type=str, default="cracker_box")

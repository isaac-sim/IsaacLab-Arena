# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Simple example of an environment defined outside Arena.

This demonstrates how to define an environment outside Isaac Lab Arena. Like the
built-in environments, it is a config dataclass plus a factory with ``build(cfg)``.
Load it with the ``--external_environment_class_path`` CLI flag, which also registers
it; the config fields become flags after the environment name::

    python isaaclab_arena/evaluation/policy_runner.py \\
        --policy_type zero_action --num_steps 10 \\
        --external_environment_class_path isaaclab_arena_examples.external_environments.basic.ExternalFrankaTableEnvironment \\
        franka_table --object cracker_box
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@dataclass
class ExternalFrankaTableEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the external Franka table environment."""

    object: str = "cracker_box"
    """Registered name of the object placed on the table."""


class ExternalFrankaTableEnvironment(ArenaEnvironmentFactory[ExternalFrankaTableEnvironmentCfg]):

    name: str = "franka_table"

    def build(self, cfg: ExternalFrankaTableEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its config."""
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.relations import IsAnchor, On
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.no_task import NoTask

        # Grab some assets from the registry.
        background = self.asset_registry.get_asset_by_name("maple_table_robolab")()
        light = self.asset_registry.get_asset_by_name("light")()
        pick_up_object = self.asset_registry.get_asset_by_name(cfg.object)()
        embodiment = self.asset_registry.get_asset_by_name("franka_ik")(enable_cameras=cfg.enable_cameras)

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

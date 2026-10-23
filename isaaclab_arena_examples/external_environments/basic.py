# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Simple example externally-defined environment.

This demonstrates how to define an environment outside ``isaaclab_arena_environments``.
An Arena Experiment selects it by the dotted class path of its factory::

    runs:
      franka_table:
        environment:
          type: isaaclab_arena_examples.external_environments.basic.ExternalFrankaTableEnvironment
          object: tomato_soup_can
        policy:
          type: zero_action
        rollout_limit:
          num_steps: 50

``external_environments_experiment.yaml`` in this directory runs this environment with
``isaaclab_arena/evaluation/experiment_runner.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@dataclass
class ExternalFrankaTableEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the external Franka table environment."""

    object: str = "cracker_box"
    """Registered name of the object placed on the table."""


@register_environment
class ExternalFrankaTableEnvironment(ArenaEnvironmentFactory[ExternalFrankaTableEnvironmentCfg]):

    name: str = "franka_table"

    def build(self, cfg: ExternalFrankaTableEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.relations import IsAnchor, On
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.no_task import NoTask

        # Grab some assets from the registry.
        background = self.asset_registry.get_asset_by_name("maple_table_robolab")()
        light = self.asset_registry.get_asset_by_name("light")()
        pick_up_object = self.asset_registry.get_asset_by_name(cfg.object)()
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

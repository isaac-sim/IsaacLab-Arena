# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Arena example: Dexsuite Kuka Allegro **lift** task.

Matches the MDP of Isaac Lab ``Isaac-Dexsuite-Kuka-Allegro-Lift-v0`` (state observations, joint actions):
procedural lift cuboid + Dexsuite-style kinematic table, :class:`~isaaclab_arena.embodiments.kuka_allegro.kuka_allegro.KukaAllegroEmbodiment`.

The physics backend is selectable via the common ``--presets`` CLI flag
(same concept as Isaac Lab's ``presets=newton`` Hydra override).

**Play a checkpoint trained in Isaac Lab** (same PPO runner / obs groups)::

    ./isaaclab.sh -p isaaclab_arena/evaluation/policy_runner.py \\
      --policy_type rsl_rl_action \\
      dexsuite_lift \\
      --num_envs 1 --env_spacing 3 \\
      --checkpoint /path/to/logs/rsl_rl/dexsuite_kuka_allegro/<run>/model_<iter>.pt

    # Use Newton physics:
    ./isaaclab.sh -p isaaclab_arena/evaluation/policy_runner.py \\
      --presets newton --policy_type rsl_rl_action \\
      dexsuite_lift \\
      --num_envs 1 --env_spacing 3 \\
      --checkpoint /path/to/model.pt

Match Dexsuite training layout: use ``--env_spacing 3`` (Arena CLI defaults to a larger spacing).

The task subclasses :class:`~isaaclab_arena.tasks.lift_object_task.LiftObjectTask` and uses the same
look-at-lift-object viewer helper as other Arena lift examples (Isaac Lab's stock task uses a fixed
viewer pose instead).
"""

from __future__ import annotations

import argparse

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

# NOTE: Same pattern as other example envs — avoid heavy imports before AppLauncher.


class DexsuiteLiftEnvironment(ExampleEnvironmentBase):
    """Dexsuite Kuka Allegro lift task; RSL-RL config ``DexsuiteKukaAllegroPPORunnerCfg``."""

    name: str = "dexsuite_lift"

    def get_env(self, args_cli: argparse.Namespace):
        import math

        import isaaclab_tasks.manager_based.manipulation.dexsuite  # noqa: F401

        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.reinforcement_learning.frameworks import RLFramework
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.lift_object_task import DexsuiteLiftTask
        from isaaclab_arena.utils.pose import Pose, PoseRange

        dexsuite_table = self.asset_registry.get_asset_by_name("procedural_table")()
        dexsuite_table.set_initial_pose(Pose(position_xyz=(-0.55, 0.0, 0.235)))

        manip_object = self.asset_registry.get_asset_by_name("procedural_cube")()
        manip_object.set_initial_pose(
            PoseRange(
                position_xyz_min=(-0.75, -0.1, 0.35),
                position_xyz_max=(-0.35, 0.3, 0.75),
                rpy_min=(-math.pi, -math.pi, -math.pi),
                rpy_max=(math.pi, math.pi, math.pi),
            )
        )

        ground_plane = self.asset_registry.get_asset_by_name("ground_plane")()
        light = self.asset_registry.get_asset_by_name("light")()

        embodiment = self.asset_registry.get_asset_by_name("kuka_allegro")()

        scene = Scene(assets=[dexsuite_table, manip_object, ground_plane, light])
        task = DexsuiteLiftTask(lift_object=manip_object, background_scene=dexsuite_table)

        dexsuite_rl_cfg_entry = (
            "isaaclab_tasks.manager_based.manipulation.dexsuite.config.kuka_allegro.agents."
            "rsl_rl_ppo_cfg:DexsuiteKukaAllegroPPORunnerCfg"
        )

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=None,
            rl_framework=RLFramework.RSL_RL,
            rl_policy_cfg=dexsuite_rl_cfg_entry,
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        pass

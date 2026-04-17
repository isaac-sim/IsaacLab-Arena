# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import argparse

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

# NOTE: Same pattern as other example envs — avoid heavy imports before AppLauncher.


@register_environment
class DexsuiteLiftEnvironment(ExampleEnvironmentBase):
    """
    Dexsuite Kuka Allegro lift task; RSL-RL config ``DexsuiteKukaAllegroPPORunnerCfg``.
    The robot picks up a cube and lifts it to a target position.
    """

    name: str = "dexsuite_lift"

    def get_env(self, args_cli: argparse.Namespace):
        import math

        import isaaclab_tasks.manager_based.manipulation.dexsuite  # noqa: F401

        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
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
            rl_framework_entry_point="rsl_rl_cfg_entry_point",
            rl_policy_cfg=dexsuite_rl_cfg_entry,
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        pass

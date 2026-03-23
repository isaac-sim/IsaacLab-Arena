# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Arena example: Dexsuite Kuka Allegro **lift** with **Newton** physics.

Matches the MDP of Isaac Lab ``Isaac-Dexsuite-Kuka-Allegro-Lift-v0`` (state observations, joint actions):
procedural lift cuboid + Dexsuite-style kinematic table, :class:`~isaaclab_arena.embodiments.kuka_allegro.kuka_allegro.KukaAllegroDexsuiteEmbodiment` with ``physics_preset="newton"``.

**Play a checkpoint trained in Isaac Lab** (same PPO runner / obs groups)::

    ./isaaclab.sh -p isaaclab_arena/scripts/reinforcement_learning/play.py \\
      kuka_allegro_dexsuite_lift \\
      --num_envs 1 --env_spacing 3 \\
      --checkpoint /path/to/logs/rsl_rl/dexsuite_kuka_allegro/<run>/model_<iter>.pt

Match Dexsuite training layout: use ``--env_spacing 3`` (Arena CLI defaults to a larger spacing).

The task subclasses :class:`~isaaclab_arena.tasks.lift_object_task.LiftObjectTask` and uses the same
look-at-lift-object viewer helper as other Arena lift examples (Isaac Lab's stock task uses a fixed
viewer pose instead).

Use ``--enable_cameras`` (and optionally ``--duo_cameras``) if the policy was trained with vision presets.

**Note:** Checkpoints from the stock Isaac Lab task are usually trained with **PhysX**; this example uses **Newton**,
so replay quality may differ unless you train or fine-tune with Newton as well.
"""

from __future__ import annotations

import argparse

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

# NOTE: Same pattern as other example envs — avoid heavy imports before AppLauncher.


class KukaAllegroDexsuiteLiftEnvironment(ExampleEnvironmentBase):
    """Dexsuite Kuka Allegro lift task; Newton backend; RSL-RL config ``DexsuiteKukaAllegroPPORunnerCfg``."""

    name: str = "kuka_allegro_dexsuite_lift"

    def get_env(self, args_cli: argparse.Namespace):
        import isaaclab_tasks.manager_based.manipulation.dexsuite  # noqa: F401

        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.reinforcement_learning.frameworks import RLFramework
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.dexsuite_kuka_allegro_lift_task import DexsuiteKukaAllegroLiftTask

        dexsuite_table = self.asset_registry.get_asset_by_name("dexsuite_manip_table")()
        manip_object = self.asset_registry.get_asset_by_name("dexsuite_lift_object")()
        ground_plane = self.asset_registry.get_asset_by_name("ground_plane")()
        light = self.asset_registry.get_asset_by_name("light")()

        enable_cameras = getattr(args_cli, "enable_cameras", False)
        duo_cameras = getattr(args_cli, "duo_cameras", False)

        embodiment = self.asset_registry.get_asset_by_name("kuka_allegro_dexsuite")(
            physics_preset="newton",
            enable_cameras=enable_cameras,
            duo_cameras=duo_cameras,
        )

        scene = Scene(assets=[dexsuite_table, manip_object, ground_plane, light])
        task = DexsuiteKukaAllegroLiftTask(lift_object=manip_object, background_scene=dexsuite_table)

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
        parser.add_argument(
            "--duo_cameras",
            action="store_true",
            default=False,
            help="Use base+wrist cameras (Dexsuite duo layout). Match the checkpoint training preset.",
        )

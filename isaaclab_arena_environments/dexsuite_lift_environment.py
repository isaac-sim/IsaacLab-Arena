# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory
from isaaclab_arena.utils.physics_backend import PhysicsBackend

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


def _match_isaac_lab_lift_cfg(env_cfg):
    """Match Isaac Lab's Kuka-Allegro control rate and Newton configuration."""
    from isaaclab_newton.physics import NewtonCfg
    from isaaclab_tasks.core.lift import lift_env_cfg as lift

    env_cfg.sim.dt = 1 / 120
    env_cfg.sim.render_interval = 4
    env_cfg.decimation = 4
    if isinstance(env_cfg.sim.physics, NewtonCfg):
        env_cfg.sim.physics = deepcopy(lift.PhysicsCfg().newton_mjwarp)
    return env_cfg


@dataclass
class DexsuiteLiftEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the Dexsuite lift environment."""


@register_environment
class DexsuiteLiftEnvironment(ArenaEnvironmentFactory[DexsuiteLiftEnvironmentCfg]):
    """
    Dexsuite Kuka Allegro lift task; RSL-RL config ``KukaAllegroPPORunnerCfg``.
    The robot picks up a cube and lifts it to a target position.
    """

    name: str = "dexsuite_lift"
    _legacy_argparse_cfg_type = DexsuiteLiftEnvironmentCfg

    def build(self, cfg: DexsuiteLiftEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        import isaaclab_tasks.core.lift  # noqa: F401
        from isaaclab_tasks.core.lift import lift_env_cfg as lift

        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_type import ObjectType
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.lift_object_task import DexsuiteLiftTask
        from isaaclab_arena.utils.pose import Pose

        dexsuite_table = Object(
            name="table",
            prim_path="{ENV_REGEX_NS}/table",
            object_type=ObjectType.RIGID,
            spawner_cfg=deepcopy(lift.TABLE_SPAWN_CFG),
            initial_pose=Pose(position_xyz=(-0.55, 0.0, 0.235)),
        )
        dexsuite_table.object_min_z = 0.0
        dexsuite_table.disable_reset_pose()

        manip_object = Object(
            name="object",
            prim_path="{ENV_REGEX_NS}/Object",
            object_type=ObjectType.RIGID,
            spawner_cfg=deepcopy(lift.ObjectCfg().default),
            initial_pose=Pose(position_xyz=(-0.55, 0.1, 0.35)),
        )
        manip_object.disable_reset_pose()

        ground_plane = self.asset_registry.get_asset_by_name("ground_plane")()
        light = self.asset_registry.get_asset_by_name("light")()

        embodiment = self.asset_registry.get_asset_by_name("kuka_allegro")(enable_cameras=cfg.enable_cameras)

        scene = Scene(assets=[dexsuite_table, manip_object, ground_plane, light])
        task = DexsuiteLiftTask(lift_object=manip_object, background_scene=dexsuite_table)

        dexsuite_rl_cfg_entry = (
            "isaaclab_tasks.core.lift.config.kuka_allegro.agents.rsl_rl_ppo_cfg:KukaAllegroPPORunnerCfg"
        )

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=None,
            rl_framework_entry_point="rsl_rl_cfg_entry_point",
            rl_policy_cfg=dexsuite_rl_cfg_entry,
            default_physics_backend=PhysicsBackend.NEWTON,
            env_cfg_callback=_match_isaac_lab_lift_cfg,
        )

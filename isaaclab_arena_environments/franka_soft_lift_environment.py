# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory
from isaaclab_arena.utils.pose import Pose

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@dataclass
class FrankaSoftLiftEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the Franka soft-lift evaluation scene."""

    enable_cameras: bool = False
    pose_variation_enabled: bool = True


@register_environment
class FrankaSoftLiftEnvironment(ArenaEnvironmentFactory[FrankaSoftLiftEnvironmentCfg]):
    """Registered provider for the Isaac-Lift-Soft-Franka evaluation scene."""

    name: str = "franka_soft_lift"
    _legacy_argparse_cfg_type = FrankaSoftLiftEnvironmentCfg

    def build(self, cfg: FrankaSoftLiftEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.franka_soft_lift_task import FrankaSoftLiftTask

        deformable = self.asset_registry.get_asset_by_name("franka_soft_lift_block")(instance_name="deformable")
        deformable.disable_reset_pose()
        initial_pose_variation = deformable.get_variation("initial_pose")
        if cfg.pose_variation_enabled:
            initial_pose_variation.enable()
        else:
            initial_pose_variation.disable()

        table = self.asset_registry.get_asset_by_name("franka_soft_lift_table")(instance_name="table")
        ground = self.asset_registry.get_asset_by_name("ground_plane")(
            instance_name="ground",
            prim_path="/World/GroundPlane",
            initial_pose=Pose(position_xyz=(0.0, 0.0, -1.05)),
        )
        sky_light = self.asset_registry.get_asset_by_name("light")(
            instance_name="sky_light",
            prim_path="/World/skyLight",
            spawner_cfg=sim_utils.DomeLightCfg(
                intensity=750.0,
                texture_file=(
                    f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr"
                ),
            ),
        )
        embodiment = self.asset_registry.get_asset_by_name("franka_soft_lift_panda")(enable_cameras=cfg.enable_cameras)

        scene = Scene(assets=[table, deformable, ground, sky_light])
        task = FrankaSoftLiftTask(deformable=deformable, table=table)

        def _set_sim_cfg(env_cfg):
            env_cfg.decimation = 1
            env_cfg.sim.dt = 1.0 / 60.0
            env_cfg.sim.render_interval = env_cfg.decimation
            env_cfg.sim.gravity = (0.0, 0.0, -9.81)
            env_cfg.sync_deformable_visual_meshes_from_sim = True
            return env_cfg

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            env_cfg_callback=_set_sim_cfg,
            default_physics_preset="newton_mjwarp_vbd_proxy",
        )

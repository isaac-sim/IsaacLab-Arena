# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Literal

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

DisplayPortProfile = Literal["insertion", "passive_drop_test"]
CurriculumMode = Literal[
    "disabled",
    "fixed80",
    "anneal_80_0_500",
    "anneal_80_0_1000",
    "anneal_80_20_500",
    "anneal_80_20_1000",
]


@dataclass
class DisplayPortInsertionSimEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the embodiment-free DisplayPort insertion simulation scene."""

    profile: DisplayPortProfile = "insertion"
    light_intensity: float = 2500.0
    socket_pos_range: list[float] = field(default_factory=lambda: [0.01, 0.01, 0.02])
    socket_orn_deg: float = 2.0
    curriculum_mode: CurriculumMode = "anneal_80_0_500"
    at_goal_depth_range: list[float] = field(default_factory=lambda: [0.0, 0.015])
    approach_depth_range: list[float] = field(default_factory=lambda: [0.02, 0.06])
    success_pos_threshold: float = 0.003
    keypoint_scale: float = 0.15
    exp_reward_weight: float = 1.5
    episode_length_s: float | None = None


def displayport_insertion_env_cfg_callback(env_cfg, profile: DisplayPortProfile = "insertion"):
    """Apply the source DisplayPort physics and stepping settings to an Arena env cfg."""
    import isaaclab.sim as sim_utils
    from isaaclab_physx.physics import PhysxCfg

    env_cfg.scene.replicate_physics = True
    env_cfg.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
    )
    env_cfg.sim.physics = PhysxCfg(
        bounce_threshold_velocity=0.2,
        friction_offset_threshold=0.01,
        friction_correlation_distance=0.00625,
        gpu_collision_stack_size=2**30,
        gpu_max_rigid_contact_count=2**23,
        gpu_max_rigid_patch_count=2**23,
    )
    env_cfg.sim.dt = 1.0 / 240.0
    env_cfg.decimation = 1 if profile == "passive_drop_test" else 8
    env_cfg.sim.render_interval = env_cfg.decimation
    return env_cfg


@register_environment
class DisplayPortInsertionSimEnvironment(ArenaEnvironmentFactory[DisplayPortInsertionSimEnvironmentCfg]):
    """Registered provider for DisplayPort plug/socket simulation without an embodiment."""

    name: str = "displayport_insertion_sim"
    _legacy_argparse_cfg_type = DisplayPortInsertionSimEnvironmentCfg

    def build(self, cfg: DisplayPortInsertionSimEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        import isaaclab.sim as sim_utils

        from isaaclab_arena.assets import displayport_insertion_geometry as geometry
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.displayport_insertion_task import (
            DisplayPortInsertionTask,
            make_displayport_episode_recorder_term,
        )
        from isaaclab_arena.utils.pose import Pose

        ground = self.asset_registry.get_asset_by_name("ground_plane")()
        ground_z = 0.0 if cfg.profile == "passive_drop_test" else -1.05
        ground.set_initial_pose(Pose(position_xyz=(0.0, 0.0, ground_z)), create_reset_event=False)

        light_spawner_cfg = sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=cfg.light_intensity)
        light = self.asset_registry.get_asset_by_name("light")(spawner_cfg=light_spawner_cfg)

        plug = self.asset_registry.get_asset_by_name("displayport_plug")()
        socket = self.asset_registry.get_asset_by_name("displayport_socket")()
        if cfg.profile == "passive_drop_test":
            plug.set_initial_pose(
                Pose(
                    position_xyz=geometry.PASSIVE_DROP_PLUG_POS,
                    rotation_xyzw=geometry.PASSIVE_DROP_PLUG_ROT,
                ),
                create_reset_event=False,
            )
            socket.set_initial_pose(
                Pose(
                    position_xyz=geometry.PASSIVE_DROP_SOCKET_POS,
                    rotation_xyzw=geometry.PASSIVE_DROP_SOCKET_ROT,
                ),
                create_reset_event=False,
            )

        scene = Scene(assets=[ground, light, plug, socket])
        task = DisplayPortInsertionTask(
            plug=plug,
            socket=socket,
            profile=cfg.profile,
            socket_pos_range=tuple(cfg.socket_pos_range),
            socket_orn_deg=cfg.socket_orn_deg,
            curriculum_mode=cfg.curriculum_mode,
            at_goal_depth_range=tuple(cfg.at_goal_depth_range),
            approach_depth_range=tuple(cfg.approach_depth_range),
            success_pos_threshold=cfg.success_pos_threshold,
            keypoint_scale=cfg.keypoint_scale,
            exp_reward_weight=cfg.exp_reward_weight,
            episode_length_s=cfg.episode_length_s,
        )

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=None,
            scene=scene,
            task=task,
            env_cfg_callback=partial(displayport_insertion_env_cfg_callback, profile=cfg.profile),
            episode_recorder_terms={"displayport_insertion": make_displayport_episode_recorder_term(task)},
            allow_physics_presets=False,
        )

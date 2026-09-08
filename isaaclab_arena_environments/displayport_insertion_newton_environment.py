# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Newton environment for inserting a DisplayPort plug into its socket."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


SOCKET_INITIAL_POSE = (
    (0.475, 0.125, 0.0),
    (0.5, 0.5, 0.5, -0.5),
)
"""Socket pose calibrated to place its insertion target above the table."""

PLUG_INITIAL_POSE = (
    (0.475, 0.125, 0.0746004),
    (0.7071068, 0.7071068, 0.0, 0.0),
)
"""Plug pose 15 mm outward from the fully inserted target."""

RIZON_DISPLAYPORT_INITIAL_JOINT_POSITIONS = {
    "joint1": math.radians(34.804134),
    "joint2": math.radians(-20.901142),
    "joint3": math.radians(-6.256266),
    "joint4": math.radians(129.348470),
    "joint5": math.radians(4.467969),
    "joint6": math.radians(60.060333),
    "joint7": math.radians(115.082882),
}
"""Fixed Rizon reset pose above the DisplayPort plug."""


@dataclass
class DisplayPortInsertionNewtonEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the Newton Rizon DisplayPort-insertion environment."""

    teleop_device: str | None = None
    """Arena teleoperation device, either ``keyboard`` or ``spacemouse``."""

    episode_length_s: float = 120.0
    """Maximum episode duration."""

    def __post_init__(self) -> None:
        assert self.teleop_device in (
            None,
            "keyboard",
            "spacemouse",
        ), "teleop_device must be None, 'keyboard', or 'spacemouse'"
        assert self.episode_length_s > 0.0, "episode_length_s must be positive"


@register_environment
class DisplayPortInsertionNewtonEnvironment(ArenaEnvironmentFactory[DisplayPortInsertionNewtonEnvironmentCfg]):
    """Build a Rizon task for inserting a DisplayPort plug into its socket."""

    name = "displayport_insertion_newton"
    _legacy_argparse_cfg_type = DisplayPortInsertionNewtonEnvironmentCfg

    def build(self, cfg: DisplayPortInsertionNewtonEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from registered Arena components."""
        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.embodiments.rizon.rizon import InitiallyClosedKeyboardCfg, InitiallyClosedSpaceMouseCfg
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.displayport_insertion_task import DisplayPortInsertionTask
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena_environments import mdp

        table = self.asset_registry.get_asset_by_name("table")()
        table.set_initial_pose(
            Pose(
                position_xyz=(0.5, 0.0, 0.0),
                rotation_xyzw=(0.0, 0.0, 0.7071068, 0.7071068),
            )
        )
        socket = self.asset_registry.get_asset_by_name("displayport_socket")(
            initial_pose=Pose(position_xyz=SOCKET_INITIAL_POSE[0], rotation_xyzw=SOCKET_INITIAL_POSE[1])
        )
        plug = self.asset_registry.get_asset_by_name("displayport_plug")(
            initial_pose=Pose(position_xyz=PLUG_INITIAL_POSE[0], rotation_xyzw=PLUG_INITIAL_POSE[1])
        )
        insertion_target = ObjectReference(
            name="displayport_insertion_target",
            prim_path=f"{socket.get_prim_path()}/insertion_target",
            parent_asset=socket,
            object_type=ObjectType.BASE,
        )
        ground = self.asset_registry.get_asset_by_name("ground_plane")(
            initial_pose=Pose(position_xyz=(0.0, 0.0, -1.05))
        )
        light = self.asset_registry.get_asset_by_name("light")()
        light.set_intensity(2500.0)

        embodiment = self.asset_registry.get_asset_by_name("rizon4s_grav_differential_ik_newton")(
            enable_cameras=cfg.enable_cameras
        )
        embodiment.set_joint_initial_pos(RIZON_DISPLAYPORT_INITIAL_JOINT_POSITIONS)

        teleop_device = None
        if cfg.teleop_device == "keyboard":
            teleop_device = InitiallyClosedKeyboardCfg(pos_sensitivity=0.15, rot_sensitivity=0.3)
        elif cfg.teleop_device == "spacemouse":
            teleop_device = InitiallyClosedSpaceMouseCfg(pos_sensitivity=0.15, rot_sensitivity=0.3)

        scene = Scene(assets=[ground, table, socket, plug, insertion_target, light])
        task = DisplayPortInsertionTask(
            socket=socket,
            plug=plug,
            insertion_target=insertion_target,
            background_scene=table,
            episode_length_s=cfg.episode_length_s,
            enable_cameras=cfg.enable_cameras,
        )
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
            env_cfg_callback=mdp.displayport_insertion_newton_env_cfg_callback,
        )

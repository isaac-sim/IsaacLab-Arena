# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Barebones DisplayPort connector-insertion environment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import (
    ArenaEnvironmentCfg,
    ArenaEnvironmentFactory,
)

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import (
        IsaacLabArenaEnvironment,
    )
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import (
        IsaacLabArenaManagerBasedRLEnvCfg,
    )


_DEFAULT_ASSET_DIRECTORY = Path(__file__).resolve().parents[1] / "local_stuff" / "connector_insertion" / "assets"
_PLUG_USD = "display_port_plug_fixed_sdf.usd"
_SOCKET_USD = "display_port_socket_fixed_sdf_noprotrusions.usd"
_FRANKA_CUBE_STACK_JOINT_POS = {
    "panda_joint1": 0.0444,
    "panda_joint2": -0.1894,
    "panda_joint3": -0.1107,
    "panda_joint4": -2.5148,
    "panda_joint5": 0.0044,
    "panda_joint6": 2.3775,
    "panda_joint7": 0.6952,
    "panda_finger_joint1": 0.0400,
    "panda_finger_joint2": 0.0400,
}


@dataclass
class ConnectorInsertionEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the barebones DisplayPort connector-insertion environment."""

    asset_directory: str | None = None
    """Directory containing the DisplayPort plug and socket USD files."""

    background: str = "table"
    embodiment: str = "franka_ik"
    teleop_device: str | None = "spacemouse"


def _configure_connector_insertion_physics(
    env_cfg: IsaacLabArenaManagerBasedRLEnvCfg,
) -> IsaacLabArenaManagerBasedRLEnvCfg:
    """Apply stable 20 Hz Franka control settings for PhysX and Newton."""
    from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg

    # Preserve the cube-stacking control rate while giving Newton enough integration frequency to
    # keep the high-PD Franka stable during task-space motion.
    env_cfg.sim.dt = 0.005
    env_cfg.sim.render_interval = 10
    env_cfg.sim.physics = PhysxCfg(
        bounce_threshold_velocity=0.01,
        gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
        gpu_total_aggregate_pairs_capacity=2**21,
        friction_correlation_distance=0.00625,
    )
    # Newton's native actuator path keeps the high-PD Franka stable during task-space motion. This is
    # also safe with PhysX: implicit actuators continue to use their configured simulation gains.
    env_cfg.sim.use_newton_actuators = True
    env_cfg.decimation = 10
    return env_cfg


@register_environment
class ConnectorInsertionEnvironment(ArenaEnvironmentFactory[ConnectorInsertionEnvironmentCfg]):
    """Build a Franka/Panda scene for manipulating a DisplayPort plug into its socket."""

    name: str = "connector_insertion"
    _legacy_argparse_cfg_type = ConnectorInsertionEnvironmentCfg

    def build(self, cfg: ConnectorInsertionEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        import isaaclab.sim as sim_utils

        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_type import ObjectType
        from isaaclab_arena.environments.isaaclab_arena_environment import (
            IsaacLabArenaEnvironment,
        )
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.goal_pose_task import GoalPoseTask
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena_environments import mdp

        asset_directory = Path(cfg.asset_directory) if cfg.asset_directory is not None else _DEFAULT_ASSET_DIRECTORY
        plug_usd_path = asset_directory / _PLUG_USD
        socket_usd_path = asset_directory / _SOCKET_USD
        assert plug_usd_path.is_file(), f"DisplayPort plug USD not found: {plug_usd_path}"
        assert socket_usd_path.is_file(), f"DisplayPort socket USD not found: {socket_usd_path}"

        background = self.asset_registry.get_asset_by_name(cfg.background)()
        background.set_initial_pose(Pose(position_xyz=(0.55, 0.0, 0.0), rotation_xyzw=(0, 0, 0.707, 0.707)))

        plug = Object(
            name="displayport_plug",
            object_type=ObjectType.RIGID,
            usd_path=str(plug_usd_path),
            initial_pose=Pose(
                position_xyz=(0.50, -0.15, 0.04),
                rotation_xyzw=(0.70710678, 0.70710678, 0.0, 0.0),
            ),
            spawn_cfg_addon={
                "rigid_props": sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_contact_impulse=1e32,
                ),
                "collision_props": sim_utils.CollisionPropertiesCfg(
                    contact_offset=1e-5,
                    rest_offset=-5e-5,
                ),
                "mass_props": sim_utils.MassPropertiesCfg(mass=0.03),
            },
        )

        socket = Object(
            name="displayport_socket",
            object_type=ObjectType.RIGID,
            usd_path=str(socket_usd_path),
            initial_pose=Pose(
                position_xyz=(0.55, 0.15, 0.0),
                rotation_xyzw=(0.5, 0.5, 0.5, -0.5),
            ),
            spawn_cfg_addon={
                "rigid_props": sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                ),
                "collision_props": sim_utils.CollisionPropertiesCfg(
                    contact_offset=1e-4,
                    rest_offset=-1e-4,
                ),
            },
        )

        light_spawner_cfg = sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1500.0)
        light = self.asset_registry.get_asset_by_name("light")(spawner_cfg=light_spawner_cfg)

        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(enable_cameras=cfg.enable_cameras)
        robot_cfg = mdp.FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # PhysX honors ``disable_gravity`` per rigid body, but Newton does not. Enable MuJoCo's
        # body and actuator gravity compensation so the Franka holds its commanded pose while
        # leaving world gravity active for the loose connector.
        robot_cfg.spawn.joint_drive_props = sim_utils.MujocoJointDrivePropertiesCfg(actuatorgravcomp=True)
        # Match the stable high-PD gains used by Isaac Lab's Franka cube-stacking IK task. The lower
        # assembly-task gains eventually drive Newton's joint state to NaN, even with zero actions.
        for actuator_name in ("panda_shoulder", "panda_forearm"):
            robot_cfg.actuators[actuator_name].stiffness = 400.0
            robot_cfg.actuators[actuator_name].damping = 80.0
        robot_cfg.actuators["panda_hand"].stiffness = 2000.0
        robot_cfg.actuators["panda_hand"].damping = 100.0
        embodiment.scene_config.robot = robot_cfg
        embodiment.scene_config.robot.init_state.joint_pos = _FRANKA_CUBE_STACK_JOINT_POS.copy()
        embodiment.set_initial_joint_pose(list(_FRANKA_CUBE_STACK_JOINT_POS.values()))

        teleop_device = (
            self.device_registry.get_device_by_name(cfg.teleop_device)() if cfg.teleop_device is not None else None
        )
        if teleop_device is not None:
            teleop_device.pos_sensitivity = 0.15
            teleop_device.rot_sensitivity = 0.3

        scene = Scene(assets=[background, plug, socket, light])
        task = GoalPoseTask(object=plug, episode_length_s=120.0)

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
            env_cfg_callback=_configure_connector_insertion_physics,
        )

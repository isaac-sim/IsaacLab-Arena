# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""NIST assembled board gear-mesh environment with operational-space torque control."""

from __future__ import annotations

import argparse

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase


class NISTAssembledGearMeshOSCEnvironment(ExampleEnvironmentBase):
    """NIST gear insertion using OSC torque control and assembly-style observations."""

    name: str = "nist_assembled_gear_mesh_osc"

    def get_env(self, args_cli: argparse.Namespace):
        import isaaclab.sim as sim_utils
        from isaaclab.controllers import OperationalSpaceControllerCfg
        from isaaclab.envs.mdp.actions import BinaryJointPositionActionCfg
        from isaaclab.managers import ObservationTermCfg as ObsTerm
        from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg

        import isaaclab_arena_environments.mdp as mdp
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.nist_gear_insertion_task import GraspConfig, NistGearInsertionTask
        from isaaclab_arena.tasks.observations.gear_insertion_observations import NistGearInsertionPolicyObservations
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena_environments.mdp.nist_gear_insertion_osc_action import NistGearInsertionOscActionCfg

        peg_tip_offset = (0.02025, 0.0, 0.025)
        peg_base_offset = (0.02025, 0.0, 0.0)
        success_z_fraction = 0.20
        xy_threshold = 0.0025
        episode_length_s = 15.0

        table = self.asset_registry.get_asset_by_name("table")()
        assembled_board = self.asset_registry.get_asset_by_name("nist_board_assembled")()
        gears_and_base = self.asset_registry.get_asset_by_name("gears_and_base")()
        medium_gear = self.asset_registry.get_asset_by_name("medium_nist_gear")()
        light_spawner_cfg = sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1500.0)
        light = self.asset_registry.get_asset_by_name("light")(spawner_cfg=light_spawner_cfg)

        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
            enable_cameras=args_cli.enable_cameras,
            concatenate_observation_terms=True,
        )
        embodiment.scene_config.robot = mdp.FRANKA_MIMIC_OSC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        embodiment.scene_config.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_fingertip_centered",
                    name="end_effector",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
                ),
            ],
        )
        embodiment.set_initial_joint_pose([
            0.561824,
            0.287201,
            -0.543103,
            -2.410188,
            0.507908,
            2.847644,
            0.454298,
            0.04,
            0.04,
        ])
        embodiment.action_config.arm_action = NistGearInsertionOscActionCfg(
            asset_name="robot",
            joint_names=["panda_joint[1-7]"],
            body_name="panda_fingertip_centered",
            controller_cfg=OperationalSpaceControllerCfg(
                target_types=["pose_abs"],
                impedance_mode="fixed",
                inertial_dynamics_decoupling=True,
                partial_inertial_dynamics_decoupling=False,
                gravity_compensation=False,
                motion_stiffness_task=[565.0, 565.0, 565.0, 28.0, 28.0, 28.0],
                motion_damping_ratio_task=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                nullspace_control="position",
                nullspace_stiffness=10.0,
                nullspace_damping_ratio=1.0,
            ),
            position_scale=1.0,
            orientation_scale=1.0,
            nullspace_joint_pos_target="default",
            fixed_asset_name=gears_and_base.name,
            peg_offset=peg_tip_offset,
        )
        embodiment.action_config.gripper_action = BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger_joint[1-2]"],
            open_command_expr={"panda_finger_joint1": 0.0, "panda_finger_joint2": 0.0},
            close_command_expr={"panda_finger_joint1": 0.0, "panda_finger_joint2": 0.0},
        )

        # OSC insertion requires a specialized 24-D policy observation (peg-relative
        # EE pose, force feedback, prev actions) instead of the default embodiment obs.
        # The task_obs group adds privileged state for the critic separately.
        embodiment.observation_config.policy.actions = None
        embodiment.observation_config.policy.gripper_pos = None
        embodiment.observation_config.policy.eef_pos = None
        embodiment.observation_config.policy.eef_quat = None
        embodiment.observation_config.policy.joint_pos = None
        embodiment.observation_config.policy.joint_vel = None
        embodiment.observation_config.policy.nist_gear_policy_obs = ObsTerm(
            func=NistGearInsertionPolicyObservations,
            params={
                "robot_name": "robot",
                "board_name": gears_and_base.name,
                "peg_offset": list(peg_tip_offset),
                "fingertip_body_name": "panda_fingertip_centered",
                "force_body_name": "force_sensor",
                "pos_noise_level": 0.0,
                "rot_noise_level_deg": 0.0,
                "force_noise_level": 0.0,
            },
        )
        # OSC torque control uses task-specific penalties (action magnitude, jerk,
        # contact force) instead of the default joint-level regularizers.
        embodiment.reward_config.action_rate = None
        embodiment.reward_config.joint_vel = None

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        table.set_initial_pose(Pose(position_xyz=(0.55, 0.0, -0.009), rotation_xyzw=(0.0, 0.0, 0.707, 0.707)))
        assembled_board.set_initial_pose(
            Pose(position_xyz=(0.88, 0.15, -0.009), rotation_xyzw=(0.0, 0.0, -0.7071, 0.7071))
        )
        medium_gear.set_initial_pose(Pose(position_xyz=(0.5462, -0.02386, 0.12858), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
        gears_and_base.set_initial_pose(
            Pose(position_xyz=(0.585, -0.074, 0.0), rotation_xyzw=(0.0, 0.0, 0.9239, 0.3827))
        )
        scene = Scene(assets=[table, assembled_board, medium_gear, gears_and_base, light])

        grasp_cfg = GraspConfig(
            num_arm_joints=7,
            hand_grasp_width=0.03,
            hand_close_width=0.0,
            gripper_joint_setter_func=mdp.franka_gripper_joint_setter,
            end_effector_body_name="panda_hand",
            grasp_rot_offset=[1.0, 0.0, 0.0, 0.0],
            grasp_offset=[0.02, 0.0, -0.128],
            arm_joint_names="panda_joint[1-7]",
            finger_body_names=".*finger",
        )

        task = NistGearInsertionTask(
            assembled_board=assembled_board,
            held_gear=medium_gear,
            background_scene=table,
            peg_offset_from_board=list(peg_base_offset),
            peg_offset_for_obs=list(peg_tip_offset),
            gear_base_asset=gears_and_base,
            success_z_fraction=success_z_fraction,
            xy_threshold=xy_threshold,
            episode_length_s=episode_length_s,
            grasp_cfg=grasp_cfg,
            enable_randomization=True,
            rl_training_mode=args_cli.rl_training_mode,
        )

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
            env_cfg_callback=mdp.assembly_env_cfg_callback,
            rl_framework_entry_point="rl_games_cfg_entry_point",
            rl_policy_cfg="isaaclab_arena_examples.policy:nist_gear_insertion_osc_rl_games.yaml",
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--embodiment", type=str, default="franka_ik", help="Robot embodiment")
        parser.add_argument(
            "--teleop_device", type=str, default=None, help="Teleoperation device (e.g., keyboard, spacemouse)"
        )
        parser.add_argument(
            "--rl_training_mode",
            action="store_true",
            help="Disable success termination (use when training with RL-Games).",
        )

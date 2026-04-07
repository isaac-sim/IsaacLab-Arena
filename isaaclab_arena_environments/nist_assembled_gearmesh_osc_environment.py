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
        import isaaclab.envs.mdp as mdp_isaac_lab
        import isaaclab.sim as sim_utils
        from isaaclab.controllers import OperationalSpaceControllerCfg
        from isaaclab.envs.mdp.actions import BinaryJointPositionActionCfg
        from isaaclab.managers import EventTermCfg, ObservationTermCfg as ObsTerm, SceneEntityCfg
        from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg

        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.nist_gear_insertion_task import NistGearInsertionTask
        from isaaclab_arena.utils.pose import Pose
        import isaaclab_arena_environments.mdp as mdp

        peg_tip_offset = (0.02025, 0.0, 0.025)
        peg_base_offset = (0.02025, 0.0, 0.0)
        success_z_fraction = 0.05
        xy_threshold = 0.0025
        episode_length_s = 15.0
        pos_action_threshold = 0.02
        rot_action_threshold = 0.097

        table = self.asset_registry.get_asset_by_name("table")()
        assembled_board = self.asset_registry.get_asset_by_name("nist_assembled_board")()
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
        embodiment.set_initial_joint_pose(
            [
                0.561824,
                0.287201,
                -0.543103,
                -2.410188,
                0.507908,
                2.847644,
                0.454298,
                0.04,
                0.04,
            ]
        )
        embodiment.action_config.arm_action = mdp.NistGearInsertionOscActionCfg(
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

        embodiment.observation_config.policy.actions = None
        embodiment.observation_config.policy.gripper_pos = None
        embodiment.observation_config.policy.eef_pos = None
        embodiment.observation_config.policy.eef_quat = None
        embodiment.observation_config.policy.joint_pos = None
        embodiment.observation_config.policy.joint_vel = None
        embodiment.observation_config.policy.nist_gear_policy_obs = ObsTerm(
            func=mdp.NistGearInsertionPolicyObservations,
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
        embodiment.reward_config.action_rate = None
        embodiment.reward_config.joint_vel = None

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        table.set_initial_pose(Pose(position_xyz=(0.55, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.707, 0.707)))
        assembled_board.set_initial_pose(Pose(position_xyz=(0.71, -0.005, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
        medium_gear.set_initial_pose(Pose(position_xyz=(0.5462, -0.02386, 0.12858), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
        gears_and_base.set_initial_pose(
            Pose(position_xyz=(0.585, -0.074, 0.0), rotation_xyzw=(0.0, 0.0, 0.9239, 0.3827))
        )
        scene = Scene(assets=[table, assembled_board, medium_gear, gears_and_base, light])

        nist_randomization_events = {
            "held_physics_material": EventTermCfg(
                func=mdp_isaac_lab.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg(medium_gear.name),
                    "static_friction_range": (0.75, 0.75),
                    "dynamic_friction_range": (0.75, 0.75),
                    "restitution_range": (0.0, 0.0),
                    "num_buckets": 1,
                },
            ),
            "robot_physics_material": EventTermCfg(
                func=mdp_isaac_lab.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "static_friction_range": (0.75, 0.75),
                    "dynamic_friction_range": (0.75, 0.75),
                    "restitution_range": (0.0, 0.0),
                    "num_buckets": 1,
                },
            ),
            "fixed_physics_material": EventTermCfg(
                func=mdp_isaac_lab.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg(gears_and_base.name),
                    "static_friction_range": (0.25, 1.25),
                    "dynamic_friction_range": (0.25, 0.25),
                    "restitution_range": (0.0, 0.0),
                    "num_buckets": 128,
                },
            ),
            "held_object_mass": EventTermCfg(
                func=mdp_isaac_lab.randomize_rigid_body_mass,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg(medium_gear.name),
                    "mass_distribution_params": (-0.005, 0.005),
                    "operation": "add",
                    "distribution": "uniform",
                },
            ),
            "fixed_asset_pose": EventTermCfg(
                func=mdp_isaac_lab.reset_root_state_uniform,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg(gears_and_base.name),
                    "pose_range": {
                        "x": (0.0, 0.0),
                        "y": (0.0, 0.0),
                        "z": (0.0, 0.0),
                        "roll": (0.0, 0.0),
                        "pitch": (0.0, 0.0),
                        "yaw": (0.0, 0.2617993877991494),
                    },
                    "velocity_range": {},
                },
            ),
        }

        task = NistGearInsertionTask(
            assembled_board=assembled_board,
            held_gear=medium_gear,
            background_scene=table,
            peg_offset_from_board=list(peg_base_offset),
            success_z_fraction=success_z_fraction,
            xy_threshold=xy_threshold,
            start_in_gripper=True,
            num_arm_joints=7,
            hand_grasp_width=0.03,
            hand_close_width=0.0,
            gripper_joint_setter_func=mdp.franka_gripper_joint_setter,
            end_effector_body_name="panda_hand",
            grasp_rot_offset=[0.0, 1.0, 0.0, 0.0],
            grasp_offset=[0.02, 0.0, -0.128],
            episode_length_s=episode_length_s,
            enable_randomization=True,
            arm_joint_names="panda_joint[1-7]",
            finger_body_names=".*finger",
            peg_offset_xy_noise=0.005,
            gear_base_asset=gears_and_base,
            peg_offset_for_obs=list(peg_tip_offset),
            include_insertion_regularizers=True,
            pos_action_threshold=pos_action_threshold,
            rot_action_threshold=rot_action_threshold,
            engagement_xy_threshold=xy_threshold,
            success_bonus_weight=1.0,
            disable_drop_terminations=True,
            extra_event_terms=nist_randomization_events,
        )

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
            env_cfg_callback=mdp.assembly_env_cfg_callback,
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--embodiment", type=str, default="franka_ik", help="Robot embodiment")
        parser.add_argument(
            "--teleop_device", type=str, default=None, help="Teleoperation device (e.g., keyboard, spacemouse)"
        )

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

import tempfile

from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.fourier import GR1T2_CFG

# from isaaclab_tasks.manager_based.manipulation.pick_and_place_task.pickplace_gr1t2_env_cfg import ActionsCfg as GR1T2ActionsCfg
from isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_gr1t2_env_cfg import ActionsCfg as GR1T2ActionsCfg

# from isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_gr1t2_env_cfg import ObservationsCfg as GR1T2ObservationsCfg
# from isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_gr1t2_env_cfg import EventCfg as GR1T2EventCfg
import isaaclab.controllers.utils as ControllerUtils


from isaac_arena.embodiments.embodiment_base import EmbodimentBase


class GR1T2Embodiment(EmbodimentBase):
    def __init__(self):
        # Configuration structs
        self.scene_config = GR1T2SceneCfg()
        self.action_config = GR1T2ActionsCfg()
        self.observation_config = GR1T2ObservationsCfg()
        self.event_config = GR1T2EventCfg()
        # Link the controller to the robot
        # Convert USD to URDF and change revolute joints to fixed
        self.temp_urdf_dir = tempfile.gettempdir()
        temp_urdf_output_path, temp_urdf_meshes_output_path = ControllerUtils.convert_usd_to_urdf(
            self.scene_config.robot.spawn.usd_path, self.temp_urdf_dir, force_conversion=True
        )
        ControllerUtils.change_revolute_to_fixed(
            temp_urdf_output_path, self.action_config.pink_ik_cfg.ik_urdf_fixed_joint_names
        )
        # Set the URDF and mesh paths for the IK controller
        self.action_config.pink_ik_cfg.controller.urdf_path = temp_urdf_output_path
        self.action_config.pink_ik_cfg.controller.mesh_path = temp_urdf_meshes_output_path

    # def __post_init__(self):
    #     import isaaclab.controllers.utils as ControllerUtils

    #     # Convert USD to URDF and change revolute joints to fixed
    #     # self.temp_urdf_dir = tempfile.gettempdir()
    #     temp_urdf_output_path, temp_urdf_meshes_output_path = ControllerUtils.convert_usd_to_urdf(
    #         self.robot.spawn.usd_path, self.temp_urdf_dir, force_conversion=True
    #     )
    #     ControllerUtils.change_revolute_to_fixed(
    #         temp_urdf_output_path, self.actions.pink_ik_cfg.ik_urdf_fixed_joint_names
    #     )
    #     # Set the URDF and mesh paths for the IK controller
    #     self.actions.pink_ik_cfg.controller.urdf_path = temp_urdf_output_path
    #     self.actions.pink_ik_cfg.controller.mesh_path = temp_urdf_meshes_output_path


# NOTE(alexmillane, 2025.07.25): This is partially copied from pickplace_gr1t2_env_cfg.py
# The SceneCfg definition in that file contains both the robot and the scene. So here
# we copy out just the robot to allow composition with other scenes.
@configclass
class GR1T2SceneCfg:

    # Humanoid robot w/ arms higher
    robot: ArticulationCfg = GR1T2_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0.93),
            rot=(0.7071, 0, 0, 0.7071),
            joint_pos={
                # right-arm
                "right_shoulder_pitch_joint": 0.0,
                "right_shoulder_roll_joint": 0.0,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_pitch_joint": -1.5708,
                "right_wrist_yaw_joint": 0.0,
                "right_wrist_roll_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
                # left-arm
                "left_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": 0.0,
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_pitch_joint": -1.5708,
                "left_wrist_yaw_joint": 0.0,
                "left_wrist_roll_joint": 0.0,
                "left_wrist_pitch_joint": 0.0,
                # --
                "head_.*": 0.0,
                "waist_.*": 0.0,
                ".*_hip_.*": 0.0,
                ".*_knee_.*": 0.0,
                ".*_ankle_.*": 0.0,
                "R_.*": 0.0,
                "L_.*": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
    )

    # MOVE IF POSSIBLE.
    # temp_urdf_dir = tempfile.gettempdir()

    # def __post_init__(self):

    #     import isaaclab.controllers.utils as ControllerUtils

    #     # Convert USD to URDF and change revolute joints to fixed
    #     # self.temp_urdf_dir = tempfile.gettempdir()
    #     temp_urdf_output_path, temp_urdf_meshes_output_path = ControllerUtils.convert_usd_to_urdf(
    #         self.robot.spawn.usd_path, self.temp_urdf_dir, force_conversion=True
    #     )
    #     ControllerUtils.change_revolute_to_fixed(
    #         temp_urdf_output_path, self.actions.pink_ik_cfg.ik_urdf_fixed_joint_names
    #     )
    #     # Set the URDF and mesh paths for the IK controller
    #     self.actions.pink_ik_cfg.controller.urdf_path = temp_urdf_output_path
    #     self.actions.pink_ik_cfg.controller.mesh_path = temp_urdf_meshes_output_path


# MOVE
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as base_mdp

import isaaclab_tasks.manager_based.manipulation.pick_place.mdp as mdp


# NOTE(alexmillane, 2025.07.25): This is partially copied from pickplace_gr1t2_env_cfg.py
# The ObservationsCfg definition in that file contains observations from the robot and
# the scene e.g. object positions. So here we copy out just the robot observations
# to allow composition with other scenes.
@configclass
class GR1T2ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})
        # object_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
        # object_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object")})
        robot_links_state = ObsTerm(func=mdp.get_all_robot_link_state)

        left_eef_pos = ObsTerm(func=mdp.get_left_eef_pos)
        left_eef_quat = ObsTerm(func=mdp.get_left_eef_quat)
        right_eef_pos = ObsTerm(func=mdp.get_right_eef_pos)
        right_eef_quat = ObsTerm(func=mdp.get_right_eef_quat)

        hand_joint_state = ObsTerm(func=mdp.get_hand_state)
        head_joint_state = ObsTerm(func=mdp.get_head_state)

        # object = ObsTerm(func=mdp.object_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


from isaaclab.managers import EventTermCfg as EventTerm


# DOX
@configclass
class GR1T2EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # reset_object = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {
    #             "x": [-0.05, 0.0],
    #             "y": [0.0, 0.05],
    #         },
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("object"),
    #     },
    # )

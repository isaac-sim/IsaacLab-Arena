# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tempfile
import torch
from collections.abc import Sequence

import isaaclab.controllers.utils as ControllerUtils
import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils  # noqa: F401
import isaaclab.utils.math as PoseUtils
import isaaclab_tasks.manager_based.manipulation.pick_place.mdp as mdp
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg  # noqa: F401
from isaaclab.utils import configclass
from isaaclab_assets.robots.fourier import GR1T2_CFG
from isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_gr1t2_env_cfg import ActionsCfg as GR1T2ActionsCfg

from isaac_arena.assets.register import register_asset
from isaac_arena.embodiments.embodiment_base import EmbodimentBase
from isaac_arena.geometry.pose import Pose
from isaac_arena.isaaclab_utils.resets import reset_all_articulation_joints


@register_asset
class GR1T2Embodiment(EmbodimentBase):
    """Embodiment for the GR1T2 robot."""

    name = "gr1"

    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None):
        super().__init__(enable_cameras, initial_pose)
        # Configuration structs
        self.scene_config = GR1T2SceneCfg()
        self.camera_config = GR1T2CameraCfg()
        self.action_config = GR1T2ActionsCfg()
        self.observation_config = GR1T2ObservationsCfg()
        self.event_config = GR1T2EventCfg()
        self.mimic_env = GR1T2MimicEnv

        # XR settings
        # This unfortunately works wrt to global coordinates, so its ideal if the robot is at the origin.
        self.xr: XrCfg = XrCfg(
            anchor_pos=(0.0, 0.0, -1.0),
            anchor_rot=(0.70711, 0.0, 0.0, -0.70711),
        )

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


# NOTE(alexmillane, 2025.07.25): This is partially copied from pickplace_gr1t2_env_cfg.py
# The SceneCfg definition in that file contains both the robot and the scene. So here
# we copy out just the robot to allow composition with other scenes.
@configclass
class GR1T2SceneCfg:

    # Humanoid robot w/ arms higher
    robot: ArticulationCfg = GR1T2_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
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


@configclass
class GR1T2CameraCfg:
    """Configuration for cameras."""

    robot_pov_cam: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/head_yaw_link/RobotPOVCam",
        update_period=0.0,
        height=512,
        width=512,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.15, clipping_range=(0.01, 1.0e5)),
        offset=CameraCfg.OffsetCfg(
            pos=(0.12515, 0.0, 0.06776),
            rot=(0.62, 0.32, -0.32, -0.63),
            convention="opengl",
        ),
    )


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
        robot_links_state = ObsTerm(func=mdp.get_all_robot_link_state)

        left_eef_pos = ObsTerm(func=mdp.get_left_eef_pos)
        left_eef_quat = ObsTerm(func=mdp.get_left_eef_quat)
        right_eef_pos = ObsTerm(func=mdp.get_right_eef_pos)
        right_eef_quat = ObsTerm(func=mdp.get_right_eef_quat)

        hand_joint_state = ObsTerm(func=mdp.get_hand_state)
        head_joint_state = ObsTerm(func=mdp.get_head_state)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


# NOTE(alexmillane, 2025.07.25): This is partially copied from pickplace_gr1t2_env_cfg.py
# The EventCfg definition in that file contains events from the robot and
# the scene e.g. object randomization. So here we copy out just the robot events
# to allow composition with other scenes.
@configclass
class GR1T2EventCfg:
    """Configuration for events."""

    # NOTE(alexmillane, 2025-07-28): I removed this event term because it was resetting
    # elements of the scene not related to the robot. However, this causes the humanoid
    # to not go to it's initial pose... Need to figure out what's going on here.
    reset_all = EventTerm(func=reset_all_articulation_joints, mode="reset")


class GR1T2MimicEnv(ManagerBasedRLMimicEnv):
    """Configuration for GR1T2 Mimic."""

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """
        Get current robot end effector pose. Should be the same frame as used by the robot end-effector controller.

        Args:
            eef_name: Name of the end effector.
            env_ids: Environment indices to get the pose for. If None, all envs are considered.

        Returns:
            A torch.Tensor eef pose matrix. Shape is (len(env_ids), 4, 4)
        """
        if env_ids is None:
            env_ids = slice(None)

        eef_pos_name = f"{eef_name}_eef_pos"
        eef_quat_name = f"{eef_name}_eef_quat"

        target_wrist_position = self.obs_buf["policy"][eef_pos_name][env_ids]
        target_rot_mat = PoseUtils.matrix_from_quat(self.obs_buf["policy"][eef_quat_name][env_ids])

        return PoseUtils.make_pose(target_wrist_position, target_rot_mat)

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """
        Takes a target pose and gripper action for the end effector controller and returns an action
        (usually a normalized delta pose action) to try and achieve that target pose.
        Noise is added to the target pose action if specified.

        Args:
            target_eef_pose_dict: Dictionary of 4x4 target eef pose for each end-effector.
            gripper_action_dict: Dictionary of gripper actions for each end-effector.
            noise: Noise to add to the action. If None, no noise is added.
            env_id: Environment index to get the action for.

        Returns:
            An action torch.Tensor that's compatible with env.step().
        """

        # target position and rotation
        target_left_eef_pos, left_target_rot = PoseUtils.unmake_pose(target_eef_pose_dict["left"])
        target_right_eef_pos, right_target_rot = PoseUtils.unmake_pose(target_eef_pose_dict["right"])

        target_left_eef_rot_quat = PoseUtils.quat_from_matrix(left_target_rot)
        target_right_eef_rot_quat = PoseUtils.quat_from_matrix(right_target_rot)

        # gripper actions
        left_gripper_action = gripper_action_dict["left"]
        right_gripper_action = gripper_action_dict["right"]

        if action_noise_dict is not None:
            pos_noise_left = action_noise_dict["left"] * torch.randn_like(target_left_eef_pos)
            pos_noise_right = action_noise_dict["right"] * torch.randn_like(target_right_eef_pos)
            quat_noise_left = action_noise_dict["left"] * torch.randn_like(target_left_eef_rot_quat)
            quat_noise_right = action_noise_dict["right"] * torch.randn_like(target_right_eef_rot_quat)

            target_left_eef_pos += pos_noise_left
            target_right_eef_pos += pos_noise_right
            target_left_eef_rot_quat += quat_noise_left
            target_right_eef_rot_quat += quat_noise_right

        return torch.cat(
            (
                target_left_eef_pos,
                target_left_eef_rot_quat,
                target_right_eef_pos,
                target_right_eef_rot_quat,
                left_gripper_action,
                right_gripper_action,
            ),
            dim=0,
        )

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Converts action (compatible with env.step) to a target pose for the end effector controller.
        Inverse of @target_eef_pose_to_action. Usually used to infer a sequence of target controller poses
        from a demonstration trajectory using the recorded actions.

        Args:
            action: Environment action. Shape is (num_envs, action_dim).

        Returns:
            A dictionary of eef pose torch.Tensor that @action corresponds to.
        """
        target_poses = {}

        target_left_wrist_position = action[:, 0:3]
        target_left_rot_mat = PoseUtils.matrix_from_quat(action[:, 3:7])
        target_pose_left = PoseUtils.make_pose(target_left_wrist_position, target_left_rot_mat)
        target_poses["left"] = target_pose_left

        target_right_wrist_position = action[:, 0:3]
        target_right_rot_mat = PoseUtils.matrix_from_quat(action[:, 3:7])
        target_pose_right = PoseUtils.make_pose(target_right_wrist_position, target_right_rot_mat)
        target_poses["right"] = target_pose_right

        return target_poses

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Extracts the gripper actuation part from a sequence of env actions (compatible with env.step).

        Args:
            actions: environment actions. The shape is (num_envs, num steps in a demo, action_dim).

        Returns:
            A dictionary of torch.Tensor gripper actions. Key to each dict is an eef_name.
        """
        return {"left": actions[:, 14:25], "right": actions[:, 25:]}

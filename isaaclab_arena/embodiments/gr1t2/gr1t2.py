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
from dataclasses import MISSING

import isaaclab.controllers.utils as ControllerUtils
import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils  # noqa: F401
import isaaclab.utils.math as PoseUtils
import isaaclab_tasks.manager_based.manipulation.pick_place.mdp as mdp
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, TiledCameraCfg  # noqa: F401
from isaaclab.utils import configclass
from isaaclab_assets.robots.fourier import GR1T2_CFG
from isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_gr1t2_env_cfg import ActionsCfg as GR1T2ActionsCfg

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.mimic_utils import get_rigid_and_articulated_object_poses
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.utils.isaaclab_utils.resets import reset_all_articulation_joints
from isaaclab_arena.utils.pose import Pose

ARM_JOINT_NAMES_LIST = [
    # arm joint
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "right_elbow_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    # hand joints
    "L_index_proximal_joint",
    "L_middle_proximal_joint",
    "L_pinky_proximal_joint",
    "L_ring_proximal_joint",
    "L_thumb_proximal_yaw_joint",
    "R_index_proximal_joint",
    "R_middle_proximal_joint",
    "R_pinky_proximal_joint",
    "R_ring_proximal_joint",
    "R_thumb_proximal_yaw_joint",
    "L_index_intermediate_joint",
    "L_middle_intermediate_joint",
    "L_pinky_intermediate_joint",
    "L_ring_intermediate_joint",
    "L_thumb_proximal_pitch_joint",
    "R_index_intermediate_joint",
    "R_middle_intermediate_joint",
    "R_pinky_intermediate_joint",
    "R_ring_intermediate_joint",
    "R_thumb_proximal_pitch_joint",
    "L_thumb_distal_joint",
    "R_thumb_distal_joint",
]

# Default camera offset pose
_DEFAULT_CAMERA_OFFSET = Pose(position_xyz=(0.12515, 0.0, 0.06776), rotation_wxyz=(0.62, 0.32, -0.32, -0.63))


@register_asset
class GR1T2EmbodimentBase(EmbodimentBase):
    """Embodiment for the GR1T2 robot."""

    name = "gr1"

    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None):
        super().__init__(enable_cameras, initial_pose)
        # Configuration structs
        self.scene_config = GR1T2SceneCfg()
        self.observation_config = GR1T2ObservationsCfg()
        self.event_config = GR1T2EventCfg()
        self.mimic_env = GR1T2MimicEnv
        self.action_config = MISSING
        self.camera_config = GR1T2CameraCfg()

        # XR settings
        # This unfortunately works wrt to global coordinates, so its ideal if the robot is at the origin.
        self.xr: XrCfg = XrCfg(
            anchor_pos=(0.0, 0.0, -1.0),
            anchor_rot=(0.70711, 0.0, 0.0, -0.70711),
        )


@register_asset
class GR1T2JointEmbodiment(GR1T2EmbodimentBase):
    """Embodiment for the GR1T2 robot with joint position control.

    By default uses tiled camera for efficient parallel evaluation.
    """

    name = "gr1_joint"

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        camera_offset: Pose | None = _DEFAULT_CAMERA_OFFSET,
        use_tiled_camera: bool = True,  # Default to tiled for parallel evaluation
    ):
        super().__init__(enable_cameras, initial_pose)
        # Joint positional control
        self.action_config = GR1T2JointPositionActionCfg()
        # Tuned arm joints pd gains, smoother motions and less oscillations
        self.scene_config = GR1T2HighPDSceneCfg()
        # Create camera config with private attributes to avoid scene parser issues
        self.camera_config._is_tiled_camera = use_tiled_camera
        self.camera_config._camera_offset = camera_offset


@register_asset
class GR1T2PinkEmbodiment(GR1T2EmbodimentBase):
    """Embodiment for the GR1T2 robot with PINK IK end-effector control.

    By default uses regular camera for single-environment applications.
    """

    name = "gr1_pink"

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        camera_offset: Pose | None = _DEFAULT_CAMERA_OFFSET,
        use_tiled_camera: bool = False,  # Default to regular for single env
    ):
        super().__init__(enable_cameras, initial_pose)
        # Pink IK EEF control
        self.action_config = GR1T2ActionsCfg()
        # Create camera config with private attributes to avoid scene parser issues
        self.camera_config._is_tiled_camera = use_tiled_camera
        self.camera_config._camera_offset = camera_offset

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


@configclass
class GR1T2JointPositionActionCfg:
    """Configuration for the arm joint position action."""

    joint_pos = JointPositionActionCfg(
        asset_name="robot", joint_names=ARM_JOINT_NAMES_LIST, scale=1.0, use_default_offset=False
    )


# NOTE(alexmillane, 2025.07.25): This is partially copied from pickplace_gr1t2_env_cfg.py
# The SceneCfg definition in that file contains both the robot and the scene. So here
# we copy out just the robot to allow composition with other scenes.
@configclass
class GR1T2SceneCfg:

    # Humanoid robot w/ arms higher
    # Note (xinjieyao, 2025.10.06): This is the default robot pd gains, compatible with PINK IK EEF control
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
class GR1T2HighPDSceneCfg:
    """GR1T2 Robot with tuned high PD gains on arm joints, reducing joint oscillation when using joint positional controller."""

    # Tune PD gains for the arm joints only, others kept as default
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
        actuators={
            "head": ImplicitActuatorCfg(
                joint_names_expr=[
                    "head_.*",
                ],
                effort_limit=None,
                velocity_limit=None,
                stiffness=None,
                damping=None,
            ),
            "trunk": ImplicitActuatorCfg(
                joint_names_expr=[
                    "waist_.*",
                ],
                effort_limit=None,
                velocity_limit=None,
                stiffness=None,
                damping=None,
            ),
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_hip_.*",
                    ".*_knee_.*",
                    ".*_ankle_.*",
                ],
                effort_limit=None,
                velocity_limit=None,
                stiffness=None,
                damping=None,
            ),
            "right-arm": ImplicitActuatorCfg(
                joint_names_expr=[
                    "right_shoulder_.*",
                    "right_elbow_.*",
                    "right_wrist_.*",
                ],
                effort_limit=torch.inf,
                velocity_limit=torch.inf,
                stiffness=3000,
                damping=100,
                armature=0.0,
            ),
            "left-arm": ImplicitActuatorCfg(
                joint_names_expr=[
                    "left_shoulder_.*",
                    "left_elbow_.*",
                    "left_wrist_.*",
                ],
                effort_limit=torch.inf,
                velocity_limit=torch.inf,
                stiffness=3000,
                damping=100,
                armature=0.0,
            ),
            "right-hand": ImplicitActuatorCfg(
                joint_names_expr=[
                    "R_.*",
                ],
                effort_limit=None,
                velocity_limit=None,
                stiffness=None,
                damping=None,
            ),
            "left-hand": ImplicitActuatorCfg(
                joint_names_expr=[
                    "L_.*",
                ],
                effort_limit=None,
                velocity_limit=None,
                stiffness=None,
                damping=None,
            ),
        },
    )


@configclass
class GR1T2CameraCfg:
    """Configuration for cameras."""

    robot_pov_cam: CameraCfg | TiledCameraCfg = MISSING

    def __post_init__(self):
        # Get configuration from private attributes set by embodiment constructor
        # These use getattr with defaults to avoid scene parser treating them as assets
        is_tiled_camera = getattr(self, "_is_tiled_camera", True)
        camera_offset = getattr(self, "_camera_offset", _DEFAULT_CAMERA_OFFSET)

        CameraClass = TiledCameraCfg if is_tiled_camera else CameraCfg
        OffsetClass = CameraClass.OffsetCfg

        common_kwargs = dict(
            prim_path="{ENV_REGEX_NS}/Robot/head_yaw_link/RobotPOVCam",
            update_period=0.0,
            height=512,
            width=512,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.15, clipping_range=(0.01, 1.0e5)),
        )
        offset = OffsetClass(
            pos=camera_offset.position_xyz,
            rot=camera_offset.rotation_wxyz,
            convention="opengl",
        )

        self.robot_pov_cam = CameraClass(offset=offset, **common_kwargs)


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

        target_right_wrist_position = action[:, 7:10]
        target_right_rot_mat = PoseUtils.matrix_from_quat(action[:, 10:14])
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

    # Implemented this to consider articulated objects as well
    def get_object_poses(self, env_ids: Sequence[int] | None = None):
        """
        Gets the pose of each object(rigid and articulated) in the current scene.
        Args:
            env_ids: Environment indices to get the pose for. If None, all envs are considered.
        Returns:
            A dictionary that maps object names to object pose matrix (4x4 torch.Tensor)
        """
        if env_ids is None:
            env_ids = slice(None)

        state = self.scene.get_state(is_relative=True)

        object_pose_matrix = get_rigid_and_articulated_object_poses(state, env_ids)

        return object_pose_matrix

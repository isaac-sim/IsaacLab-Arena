# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from collections.abc import Sequence
from dataclasses import MISSING

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils  # noqa: F401
import isaaclab.utils.math as PoseUtils
import isaaclab_tasks.manager_based.manipulation.pick_place.mdp as mdp
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedRLMimicEnv  # noqa: F401
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.action_manager import ActionTermCfg
from isaaclab.sensors import CameraCfg, TiledCameraCfg  # noqa: F401
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import isaaclab_arena.terms.transforms as transforms_terms
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.utils.isaaclab_utils.resets import reset_all_articulation_joints
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena_g1.g1_env.mdp import g1_events as g1_events_mdp
from isaaclab_arena_g1.g1_env.mdp import g1_observations as g1_observations_mdp
from isaaclab_arena_g1.g1_env.mdp.actions.g1_decoupled_wbc_joint_action_cfg import G1DecoupledWBCJointActionCfg
from isaaclab_arena_g1.g1_env.mdp.actions.g1_decoupled_wbc_pink_action_cfg import G1DecoupledWBCPinkActionCfg


class G1EmbodimentBase(EmbodimentBase):
    """Embodiment for the G1 robot."""

    name = "g1"

    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None):
        super().__init__(enable_cameras, initial_pose)
        # Configuration structs
        self.scene_config = G1SceneCfg()
        self.camera_config = G1CameraCfg()
        self.action_config = MISSING
        self.observation_config = MISSING
        self.event_config = MISSING
        self.mimic_env = G1MimicEnv

        # XR settings
        # This unfortunately works wrt to global coordinates, so its ideal if the robot is at the origin.
        # NOTE(xinjie.yao, 2025.09.09): Copied from GR1T2.py
        self.xr: XrCfg = XrCfg(
            anchor_pos=(0.0, 0.0, -1.0),
            anchor_rot=(0.70711, 0.0, 0.0, -0.70711),
        )


# Default camera offset pose
_DEFAULT_G1_CAMERA_OFFSET = Pose(
    position_xyz=(0.04485, 0.0, 0.35325), rotation_wxyz=(0.32651, -0.62721, 0.62721, -0.32651)
)


@register_asset
class G1WBCJointEmbodiment(G1EmbodimentBase):
    """Embodiment for the G1 robot with WBC policy and direct joint upperbody control.

    By default uses tiled camera for efficient parallel evaluation.
    """

    name = "g1_wbc_joint"

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        camera_offset: Pose | None = _DEFAULT_G1_CAMERA_OFFSET,
        use_tiled_camera: bool = True,  # Default to tiled for parallel evaluation
    ):
        super().__init__(enable_cameras, initial_pose)
        self.action_config = G1WBCJointActionCfg()
        self.observation_config = G1WBCJointObservationsCfg()
        self.event_config = G1WBCJointEventCfg()
        # Create camera config with private attributes to avoid scene parser issues
        self.camera_config._is_tiled_camera = use_tiled_camera
        self.camera_config._camera_offset = camera_offset


@register_asset
class G1WBCPinkEmbodiment(G1EmbodimentBase):
    """Embodiment for the G1 robot with WBC policy and PINK IK upperbody control.

    By default uses regular camera for single-environment applications.
    """

    name = "g1_wbc_pink"

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        camera_offset: Pose | None = _DEFAULT_G1_CAMERA_OFFSET,
        use_tiled_camera: bool = False,  # Default to regular for single env
    ):
        super().__init__(enable_cameras, initial_pose)
        self.action_config = G1WBCPinkActionCfg()
        self.observation_config = G1WBCPinkObservationsCfg()
        self.event_config = G1WBCPinkEventCfg()
        # Create camera config with private attributes to avoid scene parser issues
        self.camera_config._is_tiled_camera = use_tiled_camera
        self.camera_config._camera_offset = camera_offset


@configclass
class G1SceneCfg:

    # Gear'WBC G1 config, used in WBC training
    robot: ArticulationCfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Samples/Groot/Robots/g1_29dof_with_hand_rev_1_0.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
            ),
        ),
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.8, -1.38, 0.78),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                # target angles [rad]
                "left_hip_pitch_joint": -0.1,
                "left_hip_roll_joint": 0.0,
                "left_hip_yaw_joint": 0.0,
                "left_knee_joint": 0.3,
                "left_ankle_pitch_joint": -0.2,
                "left_ankle_roll_joint": 0.0,
                "right_hip_pitch_joint": -0.1,
                "right_hip_roll_joint": 0.0,
                "right_hip_yaw_joint": 0.0,
                "right_knee_joint": 0.3,
                "right_ankle_pitch_joint": -0.2,
                "right_ankle_roll_joint": 0.0,
                "waist_yaw_joint": 0.0,
                "waist_roll_joint": 0.0,
                "waist_pitch_joint": 0.0,
                "left_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": 0.0,
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_joint": 0.0,
                "right_shoulder_pitch_joint": 0.0,
                "right_shoulder_roll_joint": 0,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_joint": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "legs": IdealPDActuatorCfg(
                joint_names_expr=[
                    ".*_hip_yaw_joint",
                    ".*_hip_roll_joint",
                    ".*_hip_pitch_joint",
                    ".*_knee_joint",
                ],
                effort_limit={
                    ".*_hip_yaw_joint": 88.0,
                    ".*_hip_roll_joint": 88.0,
                    ".*_hip_pitch_joint": 88.0,
                    ".*_knee_joint": 139.0,
                },
                velocity_limit={
                    ".*_hip_yaw_joint": 32.0,
                    ".*_hip_roll_joint": 32.0,
                    ".*_hip_pitch_joint": 32.0,
                    ".*_knee_joint": 20.0,
                },
                stiffness={
                    ".*_hip_yaw_joint": 150.0,
                    ".*_hip_roll_joint": 150.0,
                    ".*_hip_pitch_joint": 150.0,
                    ".*_knee_joint": 300.0,
                },
                damping={
                    ".*_hip_yaw_joint": 2.0,
                    ".*_hip_roll_joint": 2.0,
                    ".*_hip_pitch_joint": 2.0,
                    ".*_knee_joint": 4.0,
                },
                armature={
                    ".*_hip_.*": 0.03,
                    ".*_knee_joint": 0.03,
                },
            ),
            "feet": IdealPDActuatorCfg(
                joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
                stiffness={
                    ".*_ankle_pitch_joint": 40.0,
                    ".*_ankle_roll_joint": 40.0,
                },
                damping={
                    ".*_ankle_pitch_joint": 2,
                    ".*_ankle_roll_joint": 2,
                },
                effort_limit={
                    ".*_ankle_pitch_joint": 50.0,
                    ".*_ankle_roll_joint": 50.0,
                },
                velocity_limit={
                    ".*_ankle_pitch_joint": 37.0,
                    ".*_ankle_roll_joint": 37.0,
                },
                armature=0.03,
                friction=0.03,
            ),
            "waist": IdealPDActuatorCfg(
                joint_names_expr=[
                    "waist_.*_joint",
                ],
                effort_limit={
                    "waist_yaw_joint": 88.0,
                    "waist_roll_joint": 50.0,
                    "waist_pitch_joint": 50.0,
                },
                velocity_limit={
                    "waist_yaw_joint": 32.0,
                    "waist_roll_joint": 37.0,
                    "waist_pitch_joint": 37.0,
                },
                stiffness={
                    "waist_yaw_joint": 250.0,
                    "waist_roll_joint": 250.0,
                    "waist_pitch_joint": 250.0,
                },
                damping={
                    "waist_yaw_joint": 5.0,
                    "waist_roll_joint": 5.0,
                    "waist_pitch_joint": 5.0,
                },
                armature=0.03,
                friction=0.03,
            ),
            "arms": IdealPDActuatorCfg(
                joint_names_expr=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*_joint",
                ],
                effort_limit={
                    ".*_shoulder_pitch_joint": 25.0,
                    ".*_shoulder_roll_joint": 25.0,
                    ".*_shoulder_yaw_joint": 25.0,
                    ".*_elbow_joint": 25.0,
                    ".*_wrist_roll_joint": 25.0,
                    ".*_wrist_pitch_joint": 5.0,
                    ".*_wrist_yaw_joint": 5.0,
                },
                velocity_limit={
                    ".*_shoulder_pitch_joint": 37.0,
                    ".*_shoulder_roll_joint": 37.0,
                    ".*_shoulder_yaw_joint": 37.0,
                    ".*_elbow_joint": 37.0,
                    ".*_wrist_roll_joint": 37.0,
                    ".*_wrist_pitch_joint": 22.0,
                    ".*_wrist_yaw_joint": 22.0,
                },
                stiffness={
                    ".*_shoulder_pitch_joint": 100.0,
                    ".*_shoulder_roll_joint": 100.0,
                    ".*_shoulder_yaw_joint": 40.0,
                    ".*_elbow_joint": 40.0,
                    ".*_wrist_.*_joint": 20.0,
                },
                damping={
                    ".*_shoulder_pitch_joint": 5.0,
                    ".*_shoulder_roll_joint": 5.0,
                    ".*_shoulder_yaw_joint": 2.0,
                    ".*_elbow_joint": 2.0,
                    ".*_wrist_.*_joint": 2.0,
                },
                armature={".*_shoulder_.*": 0.03, ".*_elbow_.*": 0.03, ".*_wrist_.*_joint": 0.03},
                friction=0.03,
            ),
            # NOTE(peterd, 9/25/2025): The follow hand joint values are tested and working with Leapmotion and Mimic
            "hands": IdealPDActuatorCfg(
                joint_names_expr=[
                    ".*_hand_.*",
                ],
                effort_limit=5.0,
                velocity_limit=10.0,
                stiffness=4.0,
                damping=0.5,
                armature=0.03,
                friction=0.03,
            ),
        },
    )


@configclass
class G1CameraCfg:
    """Configuration for cameras."""

    robot_head_cam: CameraCfg | TiledCameraCfg = MISSING

    def __post_init__(self):
        # Get configuration from private attributes set by embodiment constructor
        # These use getattr with defaults to avoid scene parser treating them as assets
        is_tiled_camera = getattr(self, "_is_tiled_camera", True)
        camera_offset = getattr(self, "_camera_offset", _DEFAULT_G1_CAMERA_OFFSET)

        CameraClass = TiledCameraCfg if is_tiled_camera else CameraCfg
        OffsetClass = CameraClass.OffsetCfg

        common_kwargs = dict(
            prim_path="{ENV_REGEX_NS}/Robot/head_link/RobotHeadCam",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=0.169,  # 1.69 mm; FOV preserved via apertures
                horizontal_aperture=0.693,  # preserves ~128° horiz FOV
                vertical_aperture=0.284,  # preserves ~80° vert FOV
                clipping_range=(0.1, 5),
            ),
        )
        offset = OffsetClass(
            pos=camera_offset.position_xyz,
            rot=camera_offset.rotation_wxyz,
            convention="ros",
        )

        self.robot_head_cam = CameraClass(offset=offset, **common_kwargs)


@configclass
class G1WBCJointObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_joint_vel = ObsTerm(
            func=base_mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        right_wrist_pose_pelvis_frame = ObsTerm(
            func=transforms_terms.transform_pose_from_world_to_target_frame,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "target_link_name": "right_wrist_yaw_link",
                "target_frame_name": "pelvis",
            },
        )
        left_wrist_pose_pelvis_frame = ObsTerm(
            func=transforms_terms.transform_pose_from_world_to_target_frame,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "target_link_name": "left_wrist_yaw_link",
                "target_frame_name": "pelvis",
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class WBCObsCfg(ObsGroup):
        """Observations for WBC policy group with state values."""

        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_joint_vel = ObsTerm(
            func=base_mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        right_wrist_pose_pelvis_frame = ObsTerm(
            func=transforms_terms.transform_pose_from_world_to_target_frame,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "target_link_name": "right_wrist_yaw_link",
                "target_frame_name": "pelvis",
            },
        )
        left_wrist_pose_pelvis_frame = ObsTerm(
            func=transforms_terms.transform_pose_from_world_to_target_frame,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "target_link_name": "left_wrist_yaw_link",
                "target_frame_name": "pelvis",
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    wbc: WBCObsCfg = WBCObsCfg()


@configclass
class G1WBCPinkObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_joint_vel = ObsTerm(
            func=base_mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        right_wrist_pose_pelvis_frame = ObsTerm(
            func=transforms_terms.transform_pose_from_world_to_target_frame,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "target_link_name": "right_wrist_yaw_link",
                "target_frame_name": "pelvis",
            },
        )
        left_wrist_pose_pelvis_frame = ObsTerm(
            func=transforms_terms.transform_pose_from_world_to_target_frame,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "target_link_name": "left_wrist_yaw_link",
                "target_frame_name": "pelvis",
            },
        )
        # Mimic required observations
        left_eef_pos = ObsTerm(
            func=transforms_terms.get_target_link_position_in_target_frame,
            params={"target_link_name": "left_wrist_yaw_link"},
        )
        left_eef_quat = ObsTerm(
            func=transforms_terms.get_target_link_quaternion_in_target_frame,
            params={"target_link_name": "left_wrist_yaw_link"},
        )
        right_eef_pos = ObsTerm(
            func=transforms_terms.get_target_link_position_in_target_frame,
            params={"target_link_name": "right_wrist_yaw_link"},
        )
        right_eef_quat = ObsTerm(
            func=transforms_terms.get_target_link_quaternion_in_target_frame,
            params={"target_link_name": "right_wrist_yaw_link"},
        )
        # Body eefs are not used for transforms so values are not important,
        # but they must be present for datagen to run since "body" is considered an eef
        body_eef_pos = ObsTerm(
            func=transforms_terms.get_target_link_position_in_target_frame,
            params={"target_link_name": "right_wrist_yaw_link"},
        )
        body_eef_quat = ObsTerm(
            func=transforms_terms.get_target_link_quaternion_in_target_frame,
            params={"target_link_name": "right_wrist_yaw_link"},
        )
        robot_pos = ObsTerm(
            func=transforms_terms.get_asset_position,
        )
        robot_quat = ObsTerm(
            func=transforms_terms.get_asset_quaternion,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class WBCObsCfg(ObsGroup):
        """Observations for WBC policy group with state values."""

        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_joint_vel = ObsTerm(
            func=base_mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        right_wrist_pose_pelvis_frame = ObsTerm(
            func=transforms_terms.transform_pose_from_world_to_target_frame,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "target_link_name": "right_wrist_yaw_link",
                "target_frame_name": "pelvis",
            },
        )
        left_wrist_pose_pelvis_frame = ObsTerm(
            func=transforms_terms.transform_pose_from_world_to_target_frame,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "target_link_name": "left_wrist_yaw_link",
                "target_frame_name": "pelvis",
            },
        )
        is_navigating = ObsTerm(
            func=g1_observations_mdp.is_navigating,
        )
        navigation_goal_reached = ObsTerm(
            func=g1_observations_mdp.navigation_goal_reached,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class ActionLowerBodyCfg(ObsGroup):
        """Observations for post step policy group"""

        left_eef_pos = ObsTerm(
            func=g1_observations_mdp.extract_action_components,
            params={"mode": g1_observations_mdp.ActionComponentMode.LEFT_EEF_POS},
        )
        left_eef_quat = ObsTerm(
            func=g1_observations_mdp.extract_action_components,
            params={"mode": g1_observations_mdp.ActionComponentMode.LEFT_EEF_QUAT},
        )
        right_eef_pos = ObsTerm(
            func=g1_observations_mdp.extract_action_components,
            params={"mode": g1_observations_mdp.ActionComponentMode.RIGHT_EEF_POS},
        )
        right_eef_quat = ObsTerm(
            func=g1_observations_mdp.extract_action_components,
            params={"mode": g1_observations_mdp.ActionComponentMode.RIGHT_EEF_QUAT},
        )
        navigate_cmd = ObsTerm(
            func=g1_observations_mdp.get_navigate_cmd,
        )
        base_height_cmd = ObsTerm(
            func=g1_observations_mdp.extract_action_components,
            params={"mode": g1_observations_mdp.ActionComponentMode.BASE_HEIGHT_CMD},
        )
        torso_orientation_rpy_cmd = ObsTerm(
            func=g1_observations_mdp.extract_action_components,
            params={"mode": g1_observations_mdp.ActionComponentMode.TORSO_ORIENTATION_RPY_CMD},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    wbc: WBCObsCfg = WBCObsCfg()
    action: ActionLowerBodyCfg = ActionLowerBodyCfg()


@configclass
class G1WBCJointActionCfg:
    """Action specifications for the MDP, for G1 WBC action."""

    g1_action: ActionTermCfg = G1DecoupledWBCJointActionCfg(asset_name="robot", joint_names=[".*"])


@configclass
class G1WBCPinkActionCfg:
    """Action specifications for the MDP, for G1 WBC action."""

    g1_action: ActionTermCfg = G1DecoupledWBCPinkActionCfg(asset_name="robot", joint_names=[".*"])


@configclass
class G1WBCJointEventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=reset_all_articulation_joints, mode="reset")
    reset_wbc_policy = EventTerm(func=g1_events_mdp.reset_decoupled_wbc_joint_policy, mode="reset")


@configclass
class G1WBCPinkEventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=reset_all_articulation_joints, mode="reset")
    reset_wbc_policy = EventTerm(func=g1_events_mdp.reset_decoupled_wbc_pink_policy, mode="reset")


class G1MimicEnv(ManagerBasedRLMimicEnv):
    """Configuration for G1 Mimic."""

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
        Takes a target pose and gripper action for the end effector controller and returns
        an environment action to try and achieve that target pose.
        Noise is added to the target pose action if specified.

        Args:
            target_eef_pose_dict: Dictionary of 4x4 target eef pose for each end-effector.
            gripper_action_dict: Dictionary of gripper actions for each end-effector.
            noise: Noise to add to the action. If None, no noise is added.
            env_id: Environment index to get the action for.

        Returns:
            An action torch.Tensor that's compatible with env.step().
        """
        target_left_eef_pos, left_target_rot = PoseUtils.unmake_pose(target_eef_pose_dict["left"].clone())
        target_right_eef_pos, right_target_rot = PoseUtils.unmake_pose(target_eef_pose_dict["right"].clone())

        target_left_eef_rot_quat = PoseUtils.quat_from_matrix(left_target_rot)
        target_right_eef_rot_quat = PoseUtils.quat_from_matrix(right_target_rot)

        # gripper actions
        left_gripper_action = gripper_action_dict["left"].unsqueeze(0)
        right_gripper_action = gripper_action_dict["right"].unsqueeze(0)

        # body gripper action is lower body control commands (nav_cmd, base_height_cmd, torso_orientation_rpy_cmd)
        body_gripper_action = gripper_action_dict["body"]

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
                left_gripper_action,
                right_gripper_action,
                target_left_eef_pos,
                target_left_eef_rot_quat,
                target_right_eef_pos,
                target_right_eef_rot_quat,
                body_gripper_action,
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

        target_left_wrist_position = action[:, 2:5]
        target_left_rot_mat = PoseUtils.matrix_from_quat(action[:, 5:9])
        target_pose_left = PoseUtils.make_pose(target_left_wrist_position, target_left_rot_mat)
        target_poses["left"] = target_pose_left

        target_right_wrist_position = action[:, 9:12]
        target_right_rot_mat = PoseUtils.matrix_from_quat(action[:, 12:16])
        target_pose_right = PoseUtils.make_pose(target_right_wrist_position, target_right_rot_mat)
        target_poses["right"] = target_pose_right

        target_poses["body"] = torch.zeros_like(target_pose_left)

        return target_poses

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Extracts the gripper actuation part from a sequence of env actions (compatible with env.step).

        Args:
            actions: environment actions. The shape is (num_envs, num steps in a demo, action_dim).

        Returns:
            A dictionary of torch.Tensor gripper actions. Key to each dict is an eef_name.
        """

        """
        Shape of actions:
            left_gripper_action shape: (1,)
            right_gripper_action shape: (1,)
            left_wrist_pos shape: (3,)
            left_wrist_quat shape: (4,)
            right_wrist_pos shape: (3,)
            right_wrist_quat shape: (4,)
            navigate_cmd shape: (3,)
            base_height_cmd shape: (1,)
            torso_orientation_rpy_cmd shape: (3,)
        """
        return {"left": actions[:, 0], "right": actions[:, 1], "body": actions[:, -7:]}

    def get_object_poses(self, env_ids: Sequence[int] | None = None):
        """
        Gets the pose of each object relevant to Isaac Lab Mimic data generation in the current scene.

        Args:
            env_ids: Environment indices to get the pose for. If None, all envs are considered.

        Returns:
            A dictionary that maps object names to object pose matrix in pelvis frame (4x4 torch.Tensor)
        """
        if env_ids is None:
            env_ids = slice(None)

        # Get pelvis inverse transform to convert from world to pelvis frame
        pelvis_pose_w = self.scene["robot"].data.body_link_state_w[
            :, self.scene["robot"].data.body_names.index("pelvis"), :
        ]
        pelvis_position_w = pelvis_pose_w[:, :3] - self.scene.env_origins
        pelvis_rot_mat_w = PoseUtils.matrix_from_quat(pelvis_pose_w[:, 3:7])
        pelvis_pose_mat_w = PoseUtils.make_pose(pelvis_position_w, pelvis_rot_mat_w)
        pelvis_pose_inv = PoseUtils.pose_inv(pelvis_pose_mat_w)

        rigid_object_states = self.scene.get_state(is_relative=True)["rigid_object"]
        object_pose_matrix = dict()
        for obj_name, obj_state in rigid_object_states.items():
            object_pose_mat_w = PoseUtils.make_pose(
                obj_state["root_pose"][env_ids, :3], PoseUtils.matrix_from_quat(obj_state["root_pose"][env_ids, 3:7])
            )
            object_pose_pelvis_frame = torch.matmul(pelvis_pose_inv, object_pose_mat_w)
            object_pose_matrix[obj_name] = object_pose_pelvis_frame

        return object_pose_matrix

# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import tempfile
import torch
from collections.abc import Sequence
from dataclasses import MISSING, field
from typing import TYPE_CHECKING

import isaaclab.controllers.utils as ControllerUtils
import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils  # noqa: F401
import isaaclab.utils.math as PoseUtils
import isaaclab_tasks.manager_based.manipulation.pick_place.mdp as mdp
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLMimicEnv
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import ActionTermCfg
from isaaclab.managers import ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, TiledCameraCfg  # noqa: F401
from isaaclab.utils import configclass
from isaaclab_assets.robots.fourier import GR1T2_CFG
from isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_gr1t2_env_cfg import ActionsCfg as GR1T2ActionsCfg

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.common.mimic_utils import get_rigid_and_articulated_object_poses
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.utils.isaaclab_utils.resets import reset_all_articulation_joints
from isaaclab_arena.utils.pose import Pose

if TYPE_CHECKING:
    from isaaclab_arena.embodiments.gr1t2.gr1t2 import GR1T2UnifiedJointPositionActionCfg

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

# Joint names list for gr1_unified format (36 joints + 3 waist = 39 total)
# This is the order expected by the unified action processor
UNIFIED_TARGET_JOINT_NAMES_LIST = [
    # arm joints (interleaved L/R) - 14 joints
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
    # left hand proximal - 5 joints (reordered: index, middle, pinky, ring, thumb_yaw)
    "L_index_proximal_joint",
    "L_middle_proximal_joint",
    "L_pinky_proximal_joint",
    "L_ring_proximal_joint",
    "L_thumb_proximal_yaw_joint",
    # right hand proximal - 5 joints
    "R_index_proximal_joint",
    "R_middle_proximal_joint",
    "R_pinky_proximal_joint",
    "R_ring_proximal_joint",
    "R_thumb_proximal_yaw_joint",
    # left hand intermediate - 5 joints (coupled from proximal)
    "L_index_intermediate_joint",
    "L_middle_intermediate_joint",
    "L_pinky_intermediate_joint",
    "L_ring_intermediate_joint",
    "L_thumb_proximal_pitch_joint",
    # right hand intermediate - 5 joints
    "R_index_intermediate_joint",
    "R_middle_intermediate_joint",
    "R_pinky_intermediate_joint",
    "R_ring_intermediate_joint",
    "R_thumb_proximal_pitch_joint",
    # thumb distal - 2 joints
    "L_thumb_distal_joint",
    "R_thumb_distal_joint",
    # waist - 3 joints
    "waist_yaw_joint",
    "waist_pitch_joint",
    "waist_roll_joint",
]

# Default camera offset pose
_DEFAULT_CAMERA_OFFSET = Pose(position_xyz=(0.12515, 0.0, 0.06776), rotation_wxyz=(0.62, 0.32, -0.32, -0.63))


class GR1T2UnifiedJointPositionAction(ActionTerm):
    """Joint action term that converts 29D gr1_unified actions to 39D joint position commands.
    
    This action term handles the mapping from gr1_unified format (29D):
    - left_arm: 7 joints [0:7]
    - right_arm: 7 joints [7:14]
    - left_hand: 6 joints (proximal only) [14:20]
    - right_hand: 6 joints (proximal only) [20:26]
    - waist: 3 joints [26:29]
    
    To the full robot joint space (39D):
    - 14 arm joints (interleaved L/R)
    - 22 hand joints (6 proximal -> 11 per hand via coupling)
    - 3 waist joints
    """

    cfg: "GR1T2UnifiedJointPositionActionCfg"
    """The configuration of the action term."""

    def __init__(self, cfg: "GR1T2UnifiedJointPositionActionCfg", env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        # Get the articulation asset
        self._asset = env.scene[cfg.asset_name]

        # Resolve the target joint IDs for the 39 joints we control
        self._joint_ids, self._joint_names = self._asset.find_joints(
            UNIFIED_TARGET_JOINT_NAMES_LIST, preserve_order=True
        )
        self._num_target_joints = len(self._joint_ids)

        # Input is 29D (gr1_unified format)
        self._input_dim = 29
        
        # Create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self._input_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._num_target_joints, device=self.device)

        # Get default joint positions for offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        else:
            self._offset = torch.zeros(self.num_envs, self._num_target_joints, device=self.device)

        self._scale = cfg.scale

    @property
    def action_dim(self) -> int:
        """Dimension of the action space (29D gr1_unified format)."""
        return self._input_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        """Raw actions received from the policy."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Processed actions sent to the articulation (39D)."""
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        """Process 29D gr1_unified actions to 39D joint positions.
        
        gr1_unified format (29D):
        - [0:7]   left_arm
        - [7:14]  right_arm
        - [14:20] left_hand (6 proximal: index, middle, ring, pinky, thumb_yaw, thumb_pitch)
        - [20:26] right_hand (6 proximal)
        - [26:29] waist (yaw, pitch, roll)
        
        Target format (39D):
        - [0:14]  arm joints (interleaved L/R)
        - [14:19] left hand proximal (reordered: index, middle, pinky, ring, thumb_yaw)
        - [19:24] right hand proximal
        - [24:29] left hand intermediate (coupled from proximal)
        - [29:34] right hand intermediate
        - [34:36] thumb distal (L, R)
        - [36:39] waist
        """
        # Store raw actions
        self._raw_actions[:] = actions

        # Extract components from 29D input
        left_arm = actions[:, 0:7]    # 7 joints
        right_arm = actions[:, 7:14]  # 7 joints
        left_hand = actions[:, 14:20]  # 6 joints: idx, mid, ring, pinky, thumb_yaw, thumb_pitch
        right_hand = actions[:, 20:26] # 6 joints
        waist = actions[:, 26:29]      # 3 joints

        # Initialize output tensor
        output = torch.zeros(self.num_envs, self._num_target_joints, device=self.device)

        # 1. ARM JOINTS [0:14] - Interleave left and right
        # Input order: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_pitch, wrist_yaw, wrist_roll, wrist_pitch
        for i in range(7):
            output[:, 2*i] = left_arm[:, i]      # Left at even indices
            output[:, 2*i + 1] = right_arm[:, i]  # Right at odd indices

        # 2. HAND JOINTS [14:36] - Apply coupling from 6D to 11D per hand
        # Input hand format: [idx_prox, mid_prox, ring_prox, pinky_prox, thumb_yaw, thumb_pitch]
        
        # Left hand processing
        l_idx_prox = left_hand[:, 0]
        l_mid_prox = left_hand[:, 1]
        l_ring_prox = left_hand[:, 2]
        l_pinky_prox = left_hand[:, 3]
        l_thumb_yaw = left_hand[:, 4]
        l_thumb_pitch = left_hand[:, 5]
        
        # Right hand processing
        r_idx_prox = right_hand[:, 0]
        r_mid_prox = right_hand[:, 1]
        r_ring_prox = right_hand[:, 2]
        r_pinky_prox = right_hand[:, 3]
        r_thumb_yaw = right_hand[:, 4]
        r_thumb_pitch = right_hand[:, 5]

        # Left hand proximal [14:19] - reordered: index, middle, pinky, ring, thumb_yaw
        output[:, 14] = l_idx_prox
        output[:, 15] = l_mid_prox
        output[:, 16] = l_pinky_prox  # pinky before ring in target order
        output[:, 17] = l_ring_prox
        output[:, 18] = l_thumb_yaw

        # Right hand proximal [19:24]
        output[:, 19] = r_idx_prox
        output[:, 20] = r_mid_prox
        output[:, 21] = r_pinky_prox
        output[:, 22] = r_ring_prox
        output[:, 23] = r_thumb_yaw

        # Left hand intermediate [24:29] - coupled from proximal (same values)
        output[:, 24] = l_idx_prox   # intermediate follows proximal
        output[:, 25] = l_mid_prox
        output[:, 26] = l_pinky_prox
        output[:, 27] = l_ring_prox
        output[:, 28] = l_thumb_pitch  # thumb_proximal_pitch

        # Right hand intermediate [29:34]
        output[:, 29] = r_idx_prox
        output[:, 30] = r_mid_prox
        output[:, 31] = r_pinky_prox
        output[:, 32] = r_ring_prox
        output[:, 33] = r_thumb_pitch

        # Thumb distal [34:36] - use thumb_pitch
        output[:, 34] = l_thumb_pitch
        output[:, 35] = r_thumb_pitch

        # 3. WAIST JOINTS [36:39]
        output[:, 36:39] = waist

        # Apply scale and offset
        self._processed_actions[:] = self._offset + self._scale * output

    def apply_actions(self):
        """Apply the processed actions to the articulation."""
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int]) -> None:
        """Reset the action term for the specified environments."""
        self._raw_actions[env_ids] = 0.0


@configclass
class GR1T2UnifiedJointPositionActionCfg(ActionTermCfg):
    """Configuration for the GR1T2 unified joint position action term.
    
    This action term accepts 29D gr1_unified format actions and converts them
    to 39D joint position commands for the full robot.
    """
    
    class_type: type[ActionTerm] = GR1T2UnifiedJointPositionAction

    asset_name: str = "robot"
    """Name of the articulation asset in the scene."""

    scale: float = 1.0
    """Scaling factor applied to input actions. Default is 1.0."""

    use_default_offset: bool = False
    """Whether to use default joint positions as offset. Default is False."""


@register_asset
class GR1T2EmbodimentBase(EmbodimentBase):
    """Embodiment for the GR1T2 robot."""

    name = "gr1"
    default_arm_mode = ArmMode.RIGHT

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
    ):
        super().__init__(enable_cameras, initial_pose, concatenate_observation_terms, arm_mode)
        # Configuration structs
        self.scene_config = GR1T2SceneCfg()
        self.observation_config = GR1T2ObservationsCfg()
        self.observation_config.policy.concatenate_terms = self.concatenate_observation_terms
        self.event_config = GR1T2EventCfg()
        self.mimic_env = GR1T2MimicEnv
        self.action_config = MISSING
        self.camera_config = GR1T2CameraCfg()

        # XR settings
        # This unfortunately works wrt to global coordinates, so its ideal if the robot is at the origin.
        self.xr: XrCfg = XrCfg(
            anchor_pos=(-0.5, 0.0, -1.0),
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
        concatenate_observation_terms: bool = False,
    ):
        super().__init__(enable_cameras, initial_pose, concatenate_observation_terms)
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
        concatenate_observation_terms: bool = False,
    ):
        super().__init__(enable_cameras, initial_pose, concatenate_observation_terms)
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

        # Set the URDF and mesh paths for the IK controller
        self.action_config.upper_body_ik.controller.urdf_path = temp_urdf_output_path
        self.action_config.upper_body_ik.controller.mesh_path = temp_urdf_meshes_output_path


@register_asset
class GR1T2UnifiedEmbodiment(GR1T2EmbodimentBase):
    """Embodiment for the GR1T2 robot with gr1_unified action format (29D).

    This embodiment accepts 29D actions in the gr1_unified format:
    - left_arm: 7 DOF (shoulder_pitch/roll/yaw, elbow_pitch, wrist_yaw/roll/pitch)
    - right_arm: 7 DOF
    - left_hand: 6 DOF (index/middle/ring/pinky proximal, thumb_yaw, thumb_pitch)
    - right_hand: 6 DOF
    - waist: 3 DOF (yaw, pitch, roll)

    The action term internally handles:
    - Interleaving arm joints (L/R)
    - Coupling hand proximal joints to intermediate/distal joints
    - Waist control

    This is compatible with gr00t VLA policy's gr1_unified embodiment output.
    By default uses tiled camera for efficient parallel evaluation.
    """

    name = "gr1_unified"

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        camera_offset: Pose | None = _DEFAULT_CAMERA_OFFSET,
        use_tiled_camera: bool = True,  # Default to tiled for parallel evaluation
        concatenate_observation_terms: bool = False,
    ):
        super().__init__(enable_cameras, initial_pose, concatenate_observation_terms)
        # Use the unified action config that accepts 29D input
        self.action_config = GR1T2UnifiedActionCfg()
        # Tuned arm joints pd gains, smoother motions and less oscillations
        self.scene_config = GR1T2HighPDSceneCfg()
        # Create camera config with private attributes to avoid scene parser issues
        self.camera_config._is_tiled_camera = use_tiled_camera
        self.camera_config._camera_offset = camera_offset


@configclass
class GR1T2UnifiedActionCfg:
    """Configuration for the GR1T2 unified action (29D gr1_unified format)."""

    joint_pos = GR1T2UnifiedJointPositionActionCfg(
        asset_name="robot", scale=1.0, use_default_offset=False
    )


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
    """Observation specifications for the MDP.
    
    Keys are named to be compatible with gr00t VLA policy (gr1_unified embodiment):
    - left_arm, right_arm: 7 DOF arm joint positions each
    - left_hand, right_hand: 6 DOF hand joint positions each (controllable joints only)
    - waist: 3 DOF waist joint positions
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values.
        
        gr00t-compatible observation keys for the gr1_unified embodiment.
        """

        # gr00t expected state keys
        left_arm = ObsTerm(
            func=mdp.get_robot_joint_state,
            params={"joint_names": [
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_pitch_joint",
                "left_wrist_yaw_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
            ]},
        )
        right_arm = ObsTerm(
            func=mdp.get_robot_joint_state,
            params={"joint_names": [
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_pitch_joint",
                "right_wrist_yaw_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
            ]},
        )
        # Only the 6 controllable joints per hand (not intermediate/distal coupled joints)
        # to match gr1_unified model expectations
        left_hand = ObsTerm(
            func=mdp.get_robot_joint_state,
            params={"joint_names": [
                "L_index_proximal_joint",
                "L_middle_proximal_joint",
                "L_ring_proximal_joint",
                "L_pinky_proximal_joint",
                "L_thumb_proximal_yaw_joint",
                "L_thumb_proximal_pitch_joint",
            ]},
        )
        right_hand = ObsTerm(
            func=mdp.get_robot_joint_state,
            params={"joint_names": [
                "R_index_proximal_joint",
                "R_middle_proximal_joint",
                "R_ring_proximal_joint",
                "R_pinky_proximal_joint",
                "R_thumb_proximal_yaw_joint",
                "R_thumb_proximal_pitch_joint",
            ]},
        )
        waist = ObsTerm(
            func=mdp.get_robot_joint_state,
            params={"joint_names": ["waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"]},
        )

        # Additional observations (not used by gr00t but useful for debugging/logging)
        actions = ObsTerm(func=mdp.last_action)
        robot_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})
        left_eef_pos = ObsTerm(func=mdp.get_eef_pos, params={"link_name": "left_hand_roll_link"})
        left_eef_quat = ObsTerm(func=mdp.get_eef_quat, params={"link_name": "left_hand_roll_link"})
        right_eef_pos = ObsTerm(func=mdp.get_eef_pos, params={"link_name": "right_hand_roll_link"})
        right_eef_quat = ObsTerm(func=mdp.get_eef_quat, params={"link_name": "right_hand_roll_link"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

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

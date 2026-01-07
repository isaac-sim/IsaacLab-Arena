from collections.abc import Sequence
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.assets.register import register_asset

from isaaclab_arena.utils.pose import Pose
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLMimicEnv 
import torch
import isaaclab.utils.math as PoseUtils
import numpy as np
import torch
import os
import math
import tempfile
from pathlib import Path

from .swerve_ik import swerve_isosceles_ik
import isaaclab.controllers.utils as ControllerUtils

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as base_mdp

from isaaclab.controllers.pink_ik.local_frame_task import LocalFrameTask
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.assets.articulation import Articulation
from isaaclab_tasks.manager_based.manipulation.pick_place import mdp as manip_mdp

from isaaclab.controllers.pink_ik.pink_ik_cfg import PinkIKControllerCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.envs import ManagerBasedEnv

from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from isaaclab.sensors import FrameTransformer, FrameTransformerCfg, OffsetCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection

def object_grasped(
    env,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    diff_threshold: float = 0.06,
    gripper_open_val: float = 0.04,      # 4cm open (adjust after testing)
    gripper_threshold = 0.015 
) -> torch.Tensor:
    """Check if an object is grasped by the specified robot."""

    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos = object.data.root_pos_w


    any_grasped = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    if hasattr(env.cfg.actions.upper_body_ik, "hand_joint_names"):
        for i in range(len(env.cfg.actions.upper_body_ik.hand_joint_names)//2):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.actions.upper_body_ik.hand_joint_names[i*2:i*2+2])
            assert len(gripper_joint_ids) == 2, "Observations only support parallel gripper for now"
            end_effector_pos = ee_frame.data.target_pos_w[:, i*2, :]
            pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)
            grasped = torch.logical_and(
                pose_diff < diff_threshold,
                torch.abs(
                    robot.data.joint_pos[:, gripper_joint_ids[0]]
                    - torch.tensor(gripper_open_val, dtype=torch.float32).to(env.device)
                )
                > gripper_threshold,
            )
            end_effector_pos = ee_frame.data.target_pos_w[:, i*2+1, :]
            pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)
            grasped = torch.logical_and(
                grasped,
                pose_diff < diff_threshold,
            )
            grasped = torch.logical_and(
                grasped,
                torch.abs(
                    robot.data.joint_pos[:, gripper_joint_ids[1]]
                    - torch.tensor(gripper_open_val, dtype=torch.float32).to(env.device)
                )
                > gripper_threshold,
            )
            any_grasped = any_grasped or grasped
    return any_grasped

@register_asset
class FiiEmbodiment(EmbodimentBase):
    """
    Embodiment for the FII robot.
    """

    name = "fii"

    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None):
        super().__init__(enable_cameras, initial_pose)
        self.scene_config = FiiSceneCfg()
        self.action_config = FiiActionsCfg()
        self.observation_config = FiiObservationsCfg()
        self.event_config = FiiEventCfg()
        self.mimic_env = FiiMimicEnv
        
        # Convert USD to URDF for Pink IK controller
        self.temp_urdf_dir = tempfile.gettempdir()
        temp_urdf_output_path, temp_urdf_meshes_output_path = ControllerUtils.convert_usd_to_urdf(
            self.scene_config.robot.spawn.usd_path, self.temp_urdf_dir, force_conversion=False
        )
        
        # Set the URDF and mesh paths for the IK controller
        self.action_config.upper_body_ik.controller.urdf_path = temp_urdf_output_path
        self.action_config.upper_body_ik.controller.mesh_path = temp_urdf_meshes_output_path

#=======================================================================
#   SCENE
#=======================================================================
@configclass
class FiiSceneCfg:
    """Scene configuration for the FII embodiment."""

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), 
            rot=(0.7071068, 0.0, 0.0, 0.7071068),
            joint_pos={
                "jack_joint": 0.7,
                "left_1_joint": 0.0,
                "left_2_joint": 0.785398,
                "left_3_joint": 0.0,
                "left_4_joint": 1.570796,
                "left_5_joint": 0.0,
                "left_6_joint": -0.785398,
                "left_7_joint": 0.0,
                "right_1_joint": 0.0,
                "right_2_joint": 0.785398,
                "right_3_joint": 0.0,
                "right_4_joint": 1.570796,
                "right_5_joint": 0.0,
                "right_6_joint": -0.785398,
                "right_7_joint": 0.0,
            }
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(Path(__file__).parent.parent / "embodiment_library" / "Fiibot_W_1_V2_251016_Modified.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            )
        ),
        actuators={
            "jack_joint": ImplicitActuatorCfg(
                joint_names_expr=["jack_joint"], 
                damping=5000., 
                stiffness=500000.
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=["left_[0-9]_joint", "right_[0-9]_joint", "left_hand_.*", "right_hand_.*"],
                damping=None,
                stiffness=None
            ),
            "base": ImplicitActuatorCfg(
                joint_names_expr=["walk_.*_joint", ".*_wheel_joint"],
                damping=None,
                stiffness=None
            ),
            "torso_and_head": ImplicitActuatorCfg(
                joint_names_expr=["waist_joint", "head_.*_joint.*"],
                damping=None,
                stiffness=None
            ),
        },
    )

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/robot/Fiibot_W_2_V2/base_link",
        debug_vis=False,
        target_frames=[
            # Left hand end-effector
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/robot/Fiibot_W_2_V2/left_hand_grip1_link",
                name="left_hand_grip1",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/robot/Fiibot_W_2_V2/left_hand_grip2_link",
                name="left_hand_grip2",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
            ),
            # Right end-effector
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/robot/Fiibot_W_2_V2/right_hand_grip1_link",
                name="right_hand_grip1",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/robot/Fiibot_W_2_V2/right_hand_grip2_link",
                name="right_hand_grip2",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
            ),
        ],
    )

#=======================================================================
#   ACTIONS
#=======================================================================


class FiibotLowerBodyAction(ActionTerm):
    """Action term that is based on Agile lower body RL policy."""

    cfg: "FiibotLowerBodyActionCfg"
    """The configuration of the action term."""

    _asset: Articulation
    """The articulation asset to which the action term is applied."""

    def __init__(self, cfg: "FiibotLowerBodyActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        
        self._env = env

        self._joint_names = [
            "walk_mid_top_joint",
            "walk_left_bottom_joint",
            "walk_right_bottom_joint",
            "jack_joint",
            "front_wheel_joint",
            "left_wheel_joint",
            "right_wheel_joint"
        ]

        self._joint_ids = [
            self._asset.data.joint_names.index(joint_name)
            for joint_name in self._joint_names
        ]

        self._joint_pos_target = torch.zeros(self.num_envs, 7, device=self.device)
        self._joint_vel_target = torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def action_dim(self) -> int:
        """Lower Body Action: [vx, vy, wz, jack_joint_height]"""
        return 4

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._joint_pos_target

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._joint_pos_target
    
    def process_actions(self, actions: torch.Tensor):

        ik_out = swerve_isosceles_ik(
            vx=float(actions[0, 0]),
            vy=float(actions[0, 1]),
            wz=float(actions[0, 2]),
            L1=0.30438,
            d=0.17362,
            w=0.25,
            R=0.06
        )

        self._joint_pos_target[:, 0] = ik_out['wheel1']['angle_rad']
        self._joint_pos_target[:, 1] = ik_out['wheel2']['angle_rad']
        self._joint_pos_target[:, 2] = ik_out['wheel3']['angle_rad']
        self._joint_pos_target[:, 3] = float(actions[0, 3])

        self._joint_vel_target[:, 0] = ik_out['wheel1']['omega']
        self._joint_vel_target[:, 1] = ik_out['wheel2']['omega']
        self._joint_vel_target[:, 2] = ik_out['wheel3']['omega']

    def apply_actions(self):

        self._joint_pos_target[:, 4:] = self._joint_pos_target[:, 4:] + self._env.physics_dt * self._joint_vel_target

        self._asset.set_joint_position_target(
            target=self._joint_pos_target,
            joint_ids=self._joint_ids
        )



@configclass
class FiibotLowerBodyActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = FiibotLowerBodyAction


@configclass
class FiiActionsCfg:
    """Action specifications for the MDP."""

    upper_body_ik = PinkInverseKinematicsActionCfg(
        pink_controlled_joint_names=[
            # "waist_joint",
            "left_1_joint",
            "left_2_joint",
            "left_3_joint",
            "left_4_joint",
            "left_5_joint",
            "left_6_joint",
            "left_7_joint",
            "right_1_joint",
            "right_2_joint",
            "right_3_joint",
            "right_4_joint",
            "right_5_joint",
            "right_6_joint",
            "right_7_joint"
        ],
        hand_joint_names=[
            "left_hand_grip1_joint",
            "left_hand_grip2_joint",
            "right_hand_grip1_joint",
            "right_hand_grip2_joint"
        ],
        target_eef_link_names={
            "left_wrist": "Fiibot_W_2_V2_left_7_Link",
            "right_wrist": "Fiibot_W_2_V2_right_7_Link",
        },
        asset_name="robot",
        controller=PinkIKControllerCfg(
            articulation_name="robot",
            base_link_name="base_link",
            num_hand_joints=4,
            show_ik_warnings=True,
            fail_on_joint_limit_violation=False,
            variable_input_tasks=[
                LocalFrameTask(
                    "Fiibot_W_2_V2_left_7_Link",
                    base_link_frame_name="Root",
                    position_cost=1.0,
                    orientation_cost=1.0,
                    lm_damping=10,
                    gain=0.1,
                ),
                LocalFrameTask(
                    "Fiibot_W_2_V2_right_7_Link",
                    base_link_frame_name="Root",
                    position_cost=1.0,
                    orientation_cost=1.0,
                    lm_damping=10,
                    gain=0.1,
                )
            ],
            fixed_input_tasks=[],
        )
    )

    lower_body_ik = FiibotLowerBodyActionCfg(
        asset_name="robot"
    )


#=======================================================================
#   OBSERVATIONS
#=======================================================================
@configclass
class FiiEventCfg:
    """Configuration for FII events."""
    
    reset_robot_joints = EventTermCfg(
        func=base_mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )


@configclass
class FiiObservationsCfg:
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=manip_mdp.last_action)
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})
        object_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
        object_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object")})
        robot_links_state = ObsTerm(func=manip_mdp.get_all_robot_link_state)

        left_eef_pos = ObsTerm(func=manip_mdp.get_eef_pos, params={"link_name": "left_7_Link"})
        left_eef_quat = ObsTerm(func=manip_mdp.get_eef_quat, params={"link_name": "left_7_Link"})
        right_eef_pos = ObsTerm(func=manip_mdp.get_eef_pos, params={"link_name": "right_7_Link"})
        right_eef_quat = ObsTerm(func=manip_mdp.get_eef_quat, params={"link_name": "right_7_Link"})

        hand_joint_state = ObsTerm(func=manip_mdp.get_robot_joint_state, params={"joint_names": [
            "left_hand_grip1_joint",
            "left_hand_grip2_joint",
            "right_hand_grip1_joint",
            "right_hand_grip2_joint"
        ]})

        # Note: object_obs function hardcodes env.scene["object"], which doesn't exist in our scene
        # We already have object_pos and object_rot observations above, so this is redundant
        # object = ObsTerm(
        #     func=manip_mdp.object_obs,
        #     params={"left_eef_link_name": "left_7_Link", "right_eef_link_name": "right_7_Link"},
        # )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        grasp = ObsTerm(
            func=object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),  # ← Now this works!
                "object_cfg": SceneEntityCfg("object"),
                "gripper_open_val": 0.04,
                "gripper_threshold": 0.015,
            },
        )
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False
    
    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()

class FiiMimicEnv(ManagerBasedRLMimicEnv):
    """Configuration for Fii Mimic."""
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
                torch.zeros(3),
                torch.ones(1)*0.7,
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

        """
        Shape of actions:
            left_wrist_pos shape: (3,)
            left_wrist_quat shape: (4,)
            right_wrist_pos shape: (3,)
            right_wrist_quat shape: (4,)
            left_gripper_action shape: (2,)
            right_gripper_action shape: (2,)
            navigate_cmd shape: (3,)
            base_height_cmd shape: (1,)
        """
        return {"left": actions[:, 14:16], "right": actions[:, 16:18]}

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

        # Get base link inverse transform to convert from world to base link frame
        base_link_pose_w = self.scene["robot"].data.body_link_state_w[
            :, self.scene["robot"].data.body_names.index("base_link"), :
        ]
        base_link_position_w = base_link_pose_w[:, :3] - self.scene.env_origins
        base_link_rot_mat_w = PoseUtils.matrix_from_quat(base_link_pose_w[:, 3:7])
        base_link_pose_mat_w = PoseUtils.make_pose(base_link_position_w, base_link_rot_mat_w)
        base_link_pose_inv = PoseUtils.pose_inv(base_link_pose_mat_w)

        rigid_object_states = self.scene.get_state(is_relative=True)["rigid_object"]
        object_pose_matrix = dict()
        for obj_name, obj_state in rigid_object_states.items():
            object_pose_mat_w = PoseUtils.make_pose(
                obj_state["root_pose"][env_ids, :3], PoseUtils.matrix_from_quat(obj_state["root_pose"][env_ids, 3:7])
            )
            object_pose_base_link_frame = torch.matmul(base_link_pose_inv, object_pose_mat_w)
            object_pose_matrix[obj_name] = object_pose_base_link_frame

        return object_pose_matrix

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """
        Gets a dictionary of termination signal flags for each subtask in a task. The flag is 1
        when the subtask has been completed and 0 otherwise. The implementation of this method is
        required if intending to enable automatic subtask term signal annotation when running the
        dataset annotation tool. This method can be kept unimplemented if intending to use manual
        subtask term signal annotation.

        Args:
            env_ids: Environment indices to get the termination signals for. If None, all envs are considered.

        Returns:
            A dictionary termination signal flags (False or True) for each subtask.
        """
        if env_ids is None:
            env_ids = slice(None)

        signals = dict()

        subtask_terms = self.obs_buf["subtask_terms"]
        if "grasp" in subtask_terms:
            signals["grasp"] = subtask_terms["grasp"][env_ids]

        # Handle multiple grasp signals
        for i in range(0, len(self.cfg.subtask_configs)):
            grasp_key = f"grasp_{i + 1}"
            if grasp_key in subtask_terms:
                signals[grasp_key] = subtask_terms[grasp_key][env_ids]
        # final subtask signal is not needed
        return signals
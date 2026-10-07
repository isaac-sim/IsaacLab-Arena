# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from collections.abc import Callable, Sequence
from dataclasses import MISSING, fields
from typing import TYPE_CHECKING, Any

import isaaclab.envs.mdp as mdp_isaac_lab
import isaaclab.sim as sim_utils
import isaaclab.utils.math as PoseUtils
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
    JointPositionActionCfg,
)
from isaaclab.managers import ActionTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG, FRANKA_PANDA_HIGH_PD_CFG
from isaaclab_tasks.contrib.stack.mdp import franka_stack_events
from isaaclab_tasks.contrib.stack.mdp.observations import ee_frame_pos, ee_frame_quat

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.common.mimic_utils import get_rigid_and_articulated_object_poses
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.embodiments.franka.observations import gripper_pos
from isaaclab_arena.embodiments.gripper import PandaGripper
from isaaclab_arena.embodiments.robot_on_stand_utils import RobotPrimSpec, StandPrimSpec, compose_on_stand_usd
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLMimicEnv
from isaaclab_arena.terms.actions import robot_action_rate_l2, robot_last_action
from isaaclab_arena.utils.cameras import ArenaCameraCfg
from isaaclab_arena.utils.pose import Pose

if TYPE_CHECKING:
    import trimesh

_DEFAULT_CAMERA_OFFSET = Pose(position_xyz=(0.11, -0.031, -0.074), rotation_xyzw=(0.0, 0.0, 0.70711, 0.70711))

_FRANKA_ROBOT_PRIM = RobotPrimSpec(
    # TODO(qianl): use FRANKA_PANDA_CFG spawn path once IsaacSim version updates to use Legacy path by default.
    robot_usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/Legacy/panda_instanceable.usd",
    root_prim_path="/panda",
    robot_base_prim_name="panda_link0",
    stand_prim_name="stand_instanceable",
)
_FRANKA_STAND_PRIM = StandPrimSpec(
    stand_usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd",
    ref_prim_path="/Stand",
    payload_child_name="Stand",
    footprint_translate_xyz=(-0.05, 0.0, 0.0),
    stand_default_footprint_xy_m=(0.3888, 0.3888),
    stand_default_height=0.8755,
)
_FRANKA_JOINT_NAMES = (
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
    "panda_finger_joint1",
    "panda_finger_joint2",
)


class FrankaEmbodimentBase(EmbodimentBase):
    """Shared Franka scene shell, observations, events, rewards, mimic env, and camera.

    The constructor derives every scene entity, frame, camera, and term name from ``instance_key``
    and builds the configurations from those names. Subclasses pass the arm articulation and a
    builder of their action terms for a given robot asset name.
    """

    default_arm_mode = ArmMode.SINGLE_ARM

    def __init__(
        self,
        robot_cfg: ArticulationCfg,
        make_action_cfg: Callable[[str], Any],
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        initial_joint_pose: list[float] | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
        instance_key: str | None = None,
    ):
        super().__init__(
            enable_cameras, initial_pose, concatenate_observation_terms, arm_mode, instance_key=instance_key
        )
        asset_name = self.get_scene_key()
        robot_path = self.get_robot_prim_path()
        ee_frame_name = self.get_instance_name("ee_frame")
        self.gripper = PandaGripper(
            asset_name=asset_name,
            frame_transformer_name=ee_frame_name,
            target_frame_name=self.get_instance_name("end_effector"),
        )
        self.scene_config = self.with_instance_names(
            FrankaSceneCfg(
                robot=_franka_robot_cfg_on_stand(robot_cfg.copy(), robot_path),
                ee_frame=_franka_ee_frame_cfg(robot_path, self.get_instance_name),
            )
        )
        self.camera_config = self.with_instance_names(
            FrankaCameraCfg(wrist_cam=_franka_wrist_cam_cfg(robot_path)), bases=(ArenaCameraCfg,)
        )
        self.action_config = self.with_instance_names(make_action_cfg(asset_name))
        # Isaac Lab's last-action and action-rate terms read every robot's actions, so a keyed robot
        # names its own action terms. Unkeyed robots keep the whole-tensor terms.
        action_names = None if instance_key is None else tuple(field.name for field in fields(self.action_config))
        self.observation_config = self.with_instance_names(
            _franka_observations_cfg(concatenate_observation_terms, asset_name, ee_frame_name, action_names)
        )
        self.event_config = self.with_instance_names(_franka_event_cfg(asset_name))
        self.reward_config = self.with_instance_names(_franka_rewards_cfg(asset_name, action_names))
        self.mimic_env = FrankaMimicEnv
        if initial_joint_pose is not None:
            self.set_initial_joint_pose(initial_joint_pose)
        self.add_camera_variations(self.camera_config)

    def get_collision_mesh(self) -> trimesh.Trimesh:
        """Return one posed box mesh for the robot and stand."""
        from isaaclab_arena.utils.usd.helpers import extract_trimesh_from_usd_at_joint_pos

        source = self.get_placement_geometry_source()
        return extract_trimesh_from_usd_at_joint_pos(source.usd_path, source.joint_pos, source.scale)

    def set_initial_joint_pose(self, initial_joint_pose: list[float]) -> None:
        """Set the spawn and reset joint positions in articulation order."""
        expected_joint_count = len(_FRANKA_JOINT_NAMES)
        assert (
            len(initial_joint_pose) == expected_joint_count
        ), f"expected {expected_joint_count} joint positions, got {len(initial_joint_pose)}"
        robot = self.get_robot_cfg()
        robot.init_state = robot.init_state.replace(joint_pos=dict(zip(_FRANKA_JOINT_NAMES, initial_joint_pose)))

    def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
        return self.get_instance_name("ee_frame")


@register_asset
class FrankaIKEmbodiment(FrankaEmbodimentBase):
    """Franka with differential IK (relative) arm control and high-PD defaults."""

    name = "franka_ik"
    tags = ["embodiment", "default"]

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        initial_joint_pose: list[float] | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
        instance_key: str | None = None,
    ):
        super().__init__(
            FRANKA_PANDA_HIGH_PD_CFG,
            _franka_ik_action_cfg,
            enable_cameras=enable_cameras,
            initial_pose=initial_pose,
            initial_joint_pose=initial_joint_pose,
            concatenate_observation_terms=concatenate_observation_terms,
            arm_mode=arm_mode,
            instance_key=instance_key,
        )

    def get_command_body_name(self) -> str:
        return getattr(self.action_config, self.get_instance_name("arm_action")).body_name


@configclass
class FrankaIKActionCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTermCfg = MISSING

    gripper_action: ActionTermCfg = MISSING


def _franka_ik_action_cfg(asset_name: str) -> FrankaIKActionCfg:
    """Return relative differential-IK arm and binary gripper actions for the named robot asset."""
    return FrankaIKActionCfg(
        arm_action=DifferentialInverseKinematicsActionCfg(
            asset_name=asset_name,
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        ),
        gripper_action=_franka_gripper_action_cfg(asset_name),
    )


@register_asset
class FrankaJointPosEmbodiment(FrankaEmbodimentBase):
    """Franka embodiment using joint-position control, matching IsaacLab's Isaac-Lift-Cube-Franka-v0.

    Uses FRANKA_PANDA_CFG (standard PD gains, gravity enabled) instead of
    FRANKA_PANDA_HIGH_PD_CFG used by :class:`FrankaIKEmbodiment`.
    """

    name = "franka_joint_pos"

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        initial_joint_pose: list[float] | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
        instance_key: str | None = None,
    ):
        super().__init__(
            FRANKA_PANDA_CFG,
            _franka_joint_pos_action_cfg,
            enable_cameras=enable_cameras,
            initial_pose=initial_pose,
            initial_joint_pose=initial_joint_pose,
            concatenate_observation_terms=concatenate_observation_terms,
            arm_mode=arm_mode,
            instance_key=instance_key,
        )

    def get_command_body_name(self) -> str:
        return "panda_hand"


@configclass
class FrankaJointPosActionsCfg:
    """Joint-position action specification matching IsaacLab's FrankaCubeLiftEnvCfg."""

    arm_action: ActionTermCfg = MISSING

    gripper_action: ActionTermCfg = MISSING


def _franka_joint_pos_action_cfg(asset_name: str) -> FrankaJointPosActionsCfg:
    """Return joint-position arm and binary gripper actions for the named robot asset."""
    return FrankaJointPosActionsCfg(
        arm_action=JointPositionActionCfg(
            asset_name=asset_name,
            joint_names=["panda_joint.*"],
            scale=0.5,
            # Actions are displacements from Isaac Lab's default Franka pose, which is the zero point
            # trained policies were fitted against. Stated rather than left to ``use_default_offset``,
            # which reads the spawn state and so would move the zero point whenever that pose changes.
            use_default_offset=False,
            offset={
                name: value
                for name, value in FRANKA_PANDA_CFG.init_state.joint_pos.items()
                if name.startswith("panda_joint")
            },
        ),
        gripper_action=_franka_gripper_action_cfg(asset_name),
    )


def _franka_gripper_action_cfg(asset_name: str) -> BinaryJointPositionActionCfg:
    """Return the binary finger action for the named robot asset."""
    return BinaryJointPositionActionCfg(
        asset_name=asset_name,
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


@configclass
class FrankaSceneCfg:
    """Additions to the scene configuration coming from the Franka embodiment."""

    robot: ArticulationCfg | None = None

    ee_frame: FrameTransformerCfg = MISSING
    """The end-effector frame marker."""


def _franka_ee_frame_cfg(robot_path: str, frame_name: Callable[[str], str]) -> FrameTransformerCfg:
    """Return the end-effector frame transformer of the robot at ``robot_path``.

    Args:
        robot_path: Root prim path of the robot articulation.
        frame_name: Maps each unkeyed target-frame name to the robot instance's name.
    """
    # Add a marker to the end-effector frame
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"
    return FrameTransformerCfg(
        prim_path=f"{robot_path}/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{robot_path}/panda_hand",
                name=frame_name("end_effector"),
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.1034],
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{robot_path}/panda_rightfinger",
                name=frame_name("tool_rightfinger"),
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.046),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{robot_path}/panda_leftfinger",
                name=frame_name("tool_leftfinger"),
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.046),
                ),
            ),
        ],
    )


@configclass
class FrankaCameraCfg(ArenaCameraCfg):
    """Configuration for cameras."""

    wrist_cam: CameraCfg = MISSING


def _franka_wrist_cam_cfg(robot_path: str) -> CameraCfg:
    """Return the wrist camera mounted on the hand of the robot at ``robot_path``."""
    return CameraCfg(
        prim_path=f"{robot_path}/panda_hand/wrist_cam",
        update_period=0.0,
        height=84,
        width=84,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.8, focus_distance=28, horizontal_aperture=5.376, vertical_aperture=3.024
        ),
        offset=CameraCfg.OffsetCfg(
            pos=_DEFAULT_CAMERA_OFFSET.position_xyz,
            rot=_DEFAULT_CAMERA_OFFSET.rotation_xyzw,
            convention="ros",
        ),
    )


@configclass
class FrankaObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp_isaac_lab.last_action)
        joint_pos = ObsTerm(func=mdp_isaac_lab.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp_isaac_lab.joint_vel_rel)
        eef_pos = ObsTerm(func=ee_frame_pos)
        eef_quat = ObsTerm(func=ee_frame_quat)
        gripper_pos = ObsTerm(func=gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


def _franka_observations_cfg(
    concatenate_terms: bool, asset_name: str, ee_frame_name: str, action_names: tuple[str, ...] | None
) -> FrankaObservationsCfg:
    """Return the policy observations; given action names, every term names its robot explicitly.

    Args:
        concatenate_terms: Whether the policy group concatenates its terms.
        asset_name: Scene key of the robot articulation.
        ee_frame_name: Scene key of the end-effector frame transformer.
        action_names: The robot's own action terms, or None to observe the whole action tensor and
            leave the terms' default entity names unchanged.
    """
    observations = FrankaObservationsCfg()
    policy = observations.policy
    policy.concatenate_terms = concatenate_terms
    if action_names is not None:
        robot = SceneEntityCfg(asset_name)
        ee_frame = SceneEntityCfg(ee_frame_name)
        policy.actions = ObsTerm(func=robot_last_action, params={"action_names": action_names})
        policy.joint_pos.params = {"asset_cfg": robot}
        policy.joint_vel.params = {"asset_cfg": robot}
        policy.eef_pos.params = {"ee_frame_cfg": ee_frame}
        policy.eef_quat.params = {"ee_frame_cfg": ee_frame}
        policy.gripper_pos.params = {"robot_cfg": robot}
    return observations


_FRANKA_READY_POSE = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.785,
    "panda_joint3": -0.1107,
    "panda_joint4": -1.1775,
    "panda_joint5": 0.0,
    "panda_joint6": 0.785,
    "panda_joint7": 0.785,
    "panda_finger_joint.*": 0.0400,
}
"""The arm pose the Franka spawns and resets in, overridable via ``set_initial_joint_pose``."""


@configclass
class FrankaEventCfg:
    """Configuration for Franka."""

    randomize_franka_joint_state: EventTerm = MISSING


def _franka_event_cfg(asset_name: str) -> FrankaEventCfg:
    """Return the reset-time joint noise event for the named robot asset."""
    return FrankaEventCfg(
        randomize_franka_joint_state=EventTerm(
            func=franka_stack_events.randomize_joint_by_gaussian_offset,
            mode="reset",
            params={
                "mean": 0.0,
                "std": 0.02,
                "asset_cfg": SceneEntityCfg(asset_name),
            },
        )
    )


@configclass
class FrankaRewardsCfg:
    """Reward specifications for the MDP."""

    action_rate: RewardTermCfg = MISSING

    joint_vel: RewardTermCfg = MISSING


def _franka_rewards_cfg(asset_name: str, action_names: tuple[str, ...] | None) -> FrankaRewardsCfg:
    """Return the regularization rewards of the named robot asset.

    Args:
        asset_name: Scene key of the robot articulation.
        action_names: The robot's own action terms, or None to penalize the whole action tensor.
    """
    if action_names is None:
        action_rate = RewardTermCfg(func=mdp_isaac_lab.action_rate_l2, weight=-0.0001)
    else:
        action_rate = RewardTermCfg(func=robot_action_rate_l2, weight=-0.0001, params={"action_names": action_names})
    return FrankaRewardsCfg(
        action_rate=action_rate,
        joint_vel=RewardTermCfg(
            func=mdp_isaac_lab.joint_vel_l2, weight=-0.0001, params={"asset_cfg": SceneEntityCfg(asset_name)}
        ),
    )


# This is copied from FrankaCubeStackIKAbsMimicEnv in isaaclab_mimic.
# We copy it as we only need a few methods from it.
# The remaining ones belong to the task.
class FrankaMimicEnv(IsaacLabArenaManagerBasedRLMimicEnv):
    """Configuration for Franka Mimic."""

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

        # Retrieve end effector pose from the observation buffer
        eef_pos = self.obs_buf["policy"]["eef_pos"][env_ids]
        eef_quat = self.obs_buf["policy"]["eef_quat"][env_ids]
        # Quaternion format is w,x,y,z
        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        noise: float | None = None,
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
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        # target position and rotation
        (target_eef_pose,) = target_eef_pose_dict.values()
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)

        # current position and rotation
        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=[env_id])[0]
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        # normalized delta position action
        delta_position = target_pos - curr_pos

        # normalized delta rotation action
        delta_rot_mat = target_rot.matmul(curr_rot.transpose(-1, -2))
        delta_quat = PoseUtils.quat_from_matrix(delta_rot_mat)
        delta_rotation = PoseUtils.axis_angle_from_quat(delta_quat)

        # get gripper action for single eef
        (gripper_action,) = gripper_action_dict.values()

        # add noise to action
        pose_action = torch.cat([delta_position, delta_rotation], dim=0)
        if noise is not None:
            noise = noise * torch.randn_like(pose_action)
            pose_action += noise
            pose_action = torch.clamp(pose_action, -1.0, 1.0)

        return torch.cat([pose_action, gripper_action], dim=0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Converts action (compatible with env.step) to a target pose for the end effector controller.
        Inverse of @target_eef_pose_to_action. Usually used to infer a sequence of target controller poses
        from a demonstration trajectory using the recorded actions.
        Args:
            action: Environment action. Shape is (num_envs, action_dim)
        Returns:
            A dictionary of eef pose torch.Tensor that @action corresponds to
        """
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        delta_position = action[:, :3]
        delta_rotation = action[:, 3:6]

        # current position and rotation
        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=None)
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        # get pose target
        target_pos = curr_pos + delta_position

        # Convert delta_rotation to axis angle form
        delta_rotation_angle = torch.linalg.norm(delta_rotation, dim=-1, keepdim=True)
        delta_rotation_axis = delta_rotation / delta_rotation_angle

        # Handle invalid division for the case when delta_rotation_angle is close to zero
        is_close_to_zero_angle = torch.isclose(delta_rotation_angle, torch.zeros_like(delta_rotation_angle)).squeeze(1)
        delta_rotation_axis[is_close_to_zero_angle] = torch.zeros_like(delta_rotation_axis)[is_close_to_zero_angle]

        delta_quat = PoseUtils.quat_from_angle_axis(delta_rotation_angle.squeeze(1), delta_rotation_axis).squeeze(0)
        delta_rot_mat = PoseUtils.matrix_from_quat(delta_quat)
        target_rot = torch.matmul(delta_rot_mat, curr_rot)

        target_poses = PoseUtils.make_pose(target_pos, target_rot).clone()

        return {eef_name: target_poses}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Extracts the gripper actuation part from a sequence of env actions (compatible with env.step).
        Args:
            actions: environment actions. The shape is (num_envs, num steps in a demo, action_dim).
        Returns:
            A dictionary of torch.Tensor gripper actions. Key to each dict is an eef_name.
        """
        # last dimension is gripper action
        return {list(self.cfg.subtask_configs.keys())[0]: actions[:, -1:]}

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


def _franka_robot_cfg_on_stand(robot_cfg: ArticulationCfg, prim_path: str) -> ArticulationCfg:
    """Copy ``robot_cfg`` onto ``prim_path`` with the composed on-stand USD."""
    cfg = robot_cfg.replace(prim_path=prim_path)
    # Arena reaches for objects on a table, so it spawns at its own ready pose rather than at the
    # pose Isaac Lab ships, which folds the elbow back.
    cfg.init_state = cfg.init_state.replace(joint_pos=_FRANKA_READY_POSE)
    cfg.spawn.usd_path = compose_on_stand_usd(
        _FRANKA_ROBOT_PRIM,
        _FRANKA_STAND_PRIM,
        stand_height_m=_FRANKA_STAND_PRIM.stand_default_height,
        output_basename="franka_panda_on_stand",
    )
    return cfg

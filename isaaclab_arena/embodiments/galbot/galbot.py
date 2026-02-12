# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import MISSING

import isaaclab.envs.mdp as mdp
from isaaclab.controllers.config.rmp_flow import GALBOT_LEFT_ARM_RMPFLOW_CFG
from isaaclab.envs.mdp.actions.rmpflow_actions_cfg import RMPFlowActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.galbot import GALBOT_ONE_CHARLIE_CFG
from isaaclab_tasks.manager_based.manipulation.pick_place.mdp import get_robot_joint_state
from isaaclab_tasks.manager_based.manipulation.stack.mdp import ee_frame_pose_in_base_frame, franka_stack_events

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.utils.pose import Pose

# PD gains are taken from the vendor-provided Galbot USD (authoritative source).
# These relatively high gains favor stiff joint tracking (pose holding / fast setpoint tracking).
GALBOT_ONE_CHARLIE_HIGH_PD_CFG = GALBOT_ONE_CHARLIE_CFG.copy()
GALBOT_ONE_CHARLIE_HIGH_PD_CFG.actuators["left_arm"].stiffness = 1745.32922 * 1e3
GALBOT_ONE_CHARLIE_HIGH_PD_CFG.actuators["left_arm"].damping = 1745
GALBOT_ONE_CHARLIE_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True


@register_asset
class GalbotEmbodiment(EmbodimentBase):
    """Embodiment for the Galbot robot with a gripper on the left arm and a suction cup on the right arm."""

    name = "galbot"
    default_arm_mode = ArmMode.LEFT

    def __init__(
        self, enable_cameras: bool = False, initial_pose: Pose | None = None, arm_mode: ArmMode | None = None
    ):
        super().__init__(enable_cameras, initial_pose, arm_mode=arm_mode)
        if self.arm_mode == ArmMode.LEFT:
            self.scene_config = GalbotLeftArmSceneCfg()
            self.action_config = GalbotLeftArmActionsCfg()
        else:
            raise NotImplementedError("Right arm (suction cup) is not supported yet.")
        self.observation_config = GalbotObservationsCfg()
        self.event_config = GalbotEventCfg()


@configclass
class GalbotSceneCfg:
    """Scene configuration for the Galbot."""

    robot = GALBOT_ONE_CHARLIE_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    ee_frame: FrameTransformerCfg = MISSING


@configclass
class GalbotLeftArmSceneCfg(GalbotSceneCfg):
    """Scene configuration for the Galbot left arm."""

    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/left_gripper_tcp_link",
                name="left_end_effector",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.0),
                ),
            ),
        ],
    )

    def __post_init__(self):

        # Add a marker to the end-effector frame
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.ee_frame.visualizer_cfg = marker_cfg


@configclass
class GalbotLeftArmActionsCfg:
    """Action configuration for the Galbot left arm."""

    arm_action = RMPFlowActionCfg(
        asset_name="robot",
        joint_names=["left_arm_joint.*"],
        body_name="left_gripper_tcp_link",
        controller=GALBOT_LEFT_ARM_RMPFLOW_CFG,
        scale=1.0,
        body_offset=RMPFlowActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        articulation_prim_expr="/World/envs/env_.*/Robot",
        use_relative_mode=True,
    )

    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_gripper_.*_joint"],
        open_command_expr={"left_gripper_.*_joint": 0.04},
        close_command_expr={"left_gripper_.*_joint": 0.00},
    )


@configclass
class GalbotObservationsCfg:
    """Observation configuration for the Galbot robot."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # Since the robot may not be located at the origin of the environment, we get the EEF pose in the base frame.
        eef_pos = ObsTerm(func=ee_frame_pose_in_base_frame, params={"return_key": "pos"})
        eef_quat = ObsTerm(func=ee_frame_pose_in_base_frame, params={"return_key": "quat"})
        left_gripper_pos = ObsTerm(func=get_robot_joint_state, params={"joint_names": ["left_gripper_.*_joint"]})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class GalbotEventCfg:
    """Event configuration for the Galbot robot."""

    # NOTE(lanceli, 2026.2.6): When using the RMPFlow controller, the initial joint state must
    # be consistent with the default_joint_pos defined in GALBOT_LEFT_ARM_RMPFLOW_CFG. This event
    # term ensures the embodiment's joint states are initialized to match. For the same reason,
    # we disable randomization by setting mean=0.0 and std=0.0.
    randomize_galbot_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

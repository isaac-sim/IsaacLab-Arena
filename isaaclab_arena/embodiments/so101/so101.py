# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from pathlib import Path

import isaaclab.envs.mdp as mdp_isaac_lab
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg, RelativeJointPositionActionCfg
from isaaclab.managers import ActionTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg, TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.utils.pose import Pose

_SO101_DATA_DIR = Path(__file__).parent / "data"

_SO101_FOLLOWER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(_SO101_DATA_DIR / "so101_follower.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
            fix_root_link=True,
            sleep_threshold=0.00005,
            stabilization_threshold=0.00001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-0.20, 0.40, 0.71),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": np.pi / 2,
            "wrist_roll": np.pi / 2,
            "gripper": 0.87,
        },
    ),
    actuators={
        "shoulder_pan": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan"],
            effort_limit_sim=100,
            velocity_limit_sim=1.0,
            stiffness=1000,
            damping=100,
        ),
        "shoulder_lift": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_lift"],
            effort_limit_sim=100,
            velocity_limit_sim=1.0,
            stiffness=1000,
            damping=100,
        ),
        "elbow_flex": ImplicitActuatorCfg(
            joint_names_expr=["elbow_flex"],
            effort_limit_sim=100,
            velocity_limit_sim=1.0,
            stiffness=1000,
            damping=100,
        ),
        "wrist_flex": ImplicitActuatorCfg(
            joint_names_expr=["wrist_flex"],
            effort_limit_sim=100,
            velocity_limit_sim=1.0,
            stiffness=1000,
            damping=100,
        ),
        "wrist_roll": ImplicitActuatorCfg(
            joint_names_expr=["wrist_roll"],
            effort_limit_sim=100,
            velocity_limit_sim=1.0,
            stiffness=1000,
            damping=100,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper"],
            effort_limit_sim=100.0,
            velocity_limit_sim=1.0,
            stiffness=1000,
            damping=100,
        ),
    },
    soft_joint_pos_limit_factor=1,
)

_FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
_FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)


@configclass
class SO101SceneCfg:
    robot: ArticulationCfg = _SO101_FOLLOWER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class SO101RLSceneCfg(SO101SceneCfg):
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        debug_vis=False,
        visualizer_cfg=_FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/gripper",
                name="tool_gripper",
                offset=OffsetCfg(pos=(-0.011, -0.0001, -0.0953)),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/jaw",
                name="tool_jaw",
                offset=OffsetCfg(pos=(-0.01, -0.073, 0.019)),
            ),
        ],
    )
    base_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Scene/floor*"],
    )
    left_gripper_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=[],
    )
    right_gripper_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/jaw",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=[],
    )
    gripper_table_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Scene/counter_1_front_group/top_geometry"],
    )
    jaw_table_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/jaw",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Scene/counter_1_front_group/top_geometry"],
    )


@configclass
class SO101RelJointActionsCfg:
    arm_action: ActionTermCfg = RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["shoulder.*", "elbow_flex", "wrist.*", "gripper"],
        scale={"shoulder.*": 0.05, "elbow_flex": 0.05, "wrist.*": 0.05, "gripper": 0.2},
        use_zero_offset=True,
        clip={"shoulder.*": (-1.0, 1.0), "elbow_flex": (-1.0, 1.0), "wrist.*": (-1.0, 1.0), "gripper": (-1.0, 1.0)},
    )


@configclass
class SO101AbsJointActionsCfg:
    arm_action: ActionTermCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["shoulder.*", "elbow_flex", "wrist.*"],
        scale=1,
        use_default_offset=False,
    )
    gripper_action: ActionTermCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper"],
        scale=1,
        use_default_offset=False,
    )


@configclass
class SO101ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp_isaac_lab.last_action)
        joint_pos = ObsTerm(func=mdp_isaac_lab.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=mdp_isaac_lab.joint_vel, params={"asset_cfg": SceneEntityCfg("robot")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class SO101CameraCfg:
    hand_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper/wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.001, 0.1, -0.04),
            rot=(-0.404379, -0.912179, -0.0451242, 0.0486914),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=36.5,
            focus_distance=400.0,
            horizontal_aperture=36.83,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=224,
        height=224,
        update_period=0.05,
    )
    global_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/camera_base/global_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.2, -0.65, 0.3),
            rot=(0.8, 0.5, 0.16657, 0.2414),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=40.6,
            focus_distance=400.0,
            horizontal_aperture=38.11,
            clipping_range=(0.01, 3.0),
            lock_camera=True,
        ),
        width=224,
        height=224,
        update_period=0.05,
    )


class SO101EmbodimentBase(EmbodimentBase):
    """Base class for SO101 follower arm embodiments (6-DOF LeRobot arm)."""

    default_arm_mode = ArmMode.SINGLE_ARM

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
    ):
        super().__init__(enable_cameras, initial_pose, concatenate_observation_terms, arm_mode)
        self.scene_config = SO101SceneCfg()
        self.camera_config = SO101CameraCfg()
        self.observation_config = SO101ObservationsCfg()
        self.action_config = None

    def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
        return "tool_gripper"


@register_asset
class SO101RelJointEmbodiment(SO101EmbodimentBase):
    """SO101 with relative joint position actions."""

    name = "so101_rel_joint"

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
    ):
        super().__init__(enable_cameras, initial_pose, concatenate_observation_terms, arm_mode)
        self.action_config = SO101RelJointActionsCfg()


@register_asset
class SO101AbsJointEmbodiment(SO101EmbodimentBase):
    """SO101 with absolute joint position actions (arm + gripper separately)."""

    name = "so101_abs_joint"

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
    ):
        super().__init__(enable_cameras, initial_pose, concatenate_observation_terms, arm_mode)
        self.action_config = SO101AbsJointActionsCfg()


@register_asset
class SO101RLEmbodiment(SO101EmbodimentBase):
    """SO101 with RL-oriented scene (EE frame + contact sensors) and relative joint actions."""

    name = "so101_rl"

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
    ):
        super().__init__(enable_cameras, initial_pose, concatenate_observation_terms, arm_mode)
        self.scene_config = SO101RLSceneCfg()
        self.action_config = SO101RelJointActionsCfg()
